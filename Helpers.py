import torch
import math
import torch.nn as nn
import librosa as li
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
import os

#### Helper functions ####

def UpsampleTime(x, fs=16000): #upsample time dimension after using encoders
  batch = x[:,0,0].shape[0]
  timesteps = x[0,:,0].shape[0]
  upsampled_timesteps = 4000#fs
  channels = x[0,0,:].shape[0]
  upsampled = torch.zeros((batch, upsampled_timesteps, channels))

  j = 1

  for i in range(upsampled_timesteps):
    if i == 0:
      upsampled[:, i, :] = x[:, i, :]
    elif i % (upsampled_timesteps/timesteps) == 0:
      upsampled[:, i, :] = x[:, j, :]
      j += 1
    else:
      upsampled[:, i, :] = upsampled[:, i-1, :]

  return upsampled


### Acids code - interpolate https://github.com/acids-ircam/ddsp_pytorch/tree/master by caillonantoine
def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)



#############################################################
##############  DATA LOADING AND MANIPULATION  ##############
#############################################################

### Data manipulation helpers ###

def round_down(samples, multiple):
    return multiple * math.floor(samples / multiple)


def slice_audio(x, step):
  slices = []
  for i in range(step, len(x)+1, step):
    slices += x[i-step:i].unsqueeze(0)
  slices = torch.stack(slices)
  return slices


def CutAudioLen(track, sample_rate):
    x_original, audio_fs = li.load(track, sr=sample_rate)
    x = torch.tensor(x_original)
    floored_length = round_down(len(x), audio_fs) #cut the end of the audio so the future audio slices will all be of the same length
    x = x[:floored_length]
    return x, audio_fs


def MakeMelsTensor(sliced_x, audio_fs, mel_tensor,n_fft=128, hop=32, n_mels=229):
  for clip in sliced_x:
    clip = np.array(clip)
    mel = li.feature.melspectrogram(y=clip, sr=audio_fs, n_fft=n_fft, hop_length=hop, n_mels=n_mels)#n_fft=2048, hop_length=128, n_mels=229)
    mel_db = torch.tensor(li.power_to_db(mel, ref=np.max)).T.unsqueeze(0) #channel, time, frequency
    mel_tensor += mel_db
  return mel_tensor


def NormaliseMels(MELS):
  mean = torch.mean(MELS, dim=0).unsqueeze(0)
  std = torch.std(MELS, dim=0).unsqueeze(0)
  MELS_norm = (MELS - mean) / std
  return MELS_norm


### Main data loader ###
def LoadAudio(audio_dir, sample_rate):
  audio_tracks = os.listdir(audio_dir)
  Y = []
  X = []
  MELS = []
  for idx, track in enumerate(audio_tracks):
    if idx > 15:
      break
    audio_path = audio_dir + track
    print('INDEX', idx)
    x, audio_fs = CutAudioLen(audio_path, sample_rate)
    
    step = audio_fs//4
    sliced_x = slice_audio(x, step) #slice the audio into 16000 sample long sections
    X += sliced_x

    ### to mel specs ###
    MELS = MakeMelsTensor(sliced_x, audio_fs, MELS)
    
  # Stack
  MELS = torch.stack(MELS).unsqueeze(1)
  Y = torch.stack(X)

  ### Normalise the mel specs ###
  MELS_norm = NormaliseMels(MELS)

  return MELS_norm, Y


### Create a custom dataset class ###
class CustomDataset(Dataset):
  def __init__(self, mels, targets):
    self.targets = targets
    self.mels = mels
  def __len__(self):
    return len(self.targets)
  def __getitem__(self, idx):
    y = self.targets[idx]
    x = self.mels[idx]
    return x, y


### Define a training testing split for the dataset ###
#https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/4 - Manpreet Singh (msminhas93) May '20
def train_val_dataset(dataset, val_split=0.25):
    train_idx, val_idx = train_test_split(list(range(len(dataset))), test_size=val_split)
    datasets = {}
    datasets['train'] = Subset(dataset, train_idx)
    datasets['val'] = Subset(dataset, val_idx)
    return datasets


def ReadCSV(csv):
  csv_data = []
  times = []

  for line in csv.readlines():
    line = line.rstrip('\n').split(',')
    for i in range(len(line)):
      line[i] = float(line[i])
    csv_data.append(line[1])
    times.append(line[0])

  return csv_data, times


##############################################################
################  HARMONIC ENCODER HELPERS ###################
##############################################################

# transcribed from magenta's tensorflow implementation
def hz_to_midi(hz):
  print('deeper', type(hz))
  notes = 12 * (torch.log2(hz) - torch.log2(torch.tensor(440.0))) + 69.0
  condition = torch.le(hz, 0.0)
  notes = torch.where(condition, 0.0, notes)
  return notes


# transcribed from magenta's tensorflow implementation
def unit_scale_hz(hz, hz_min=torch.tensor(0.0), hz_max=torch.tensor(8000)): #hz_maz is 8000 because we're using sr=16000, and Nyquist states max f can be only sr/2
  print(type(hz))
  midi_notes = hz_to_midi(hz)
  midi_min = hz_to_midi(hz_min)
  midi_max = hz_to_midi(hz_max)
  unit = (midi_notes - midi_min) / (midi_max - midi_min)
  return unit


# transcribed from magenta's tensorflow implementation
def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
  """Exponentiated Sigmoid pointwise nonlinearity.

  Bounds input to [threshold, max_value] with slope given by exponent.
"""
  with torch.no_grad():
    exponentiated = max_value * torch.sigmoid(x)**torch.log(torch.tensor(exponent)) + threshold
  return exponentiated


# transcribed from magenta's tensorflow implementation
def get_harmonic_frequencies(f0):
  """Create integer multiples of the fundamental frequency.

  Args:
    frequencies: Fundamental frequencies (Hz). Shape [batch_size, :, 1].
    n_harmonics: Number of harmonics.

  Returns:
    harmonic_frequencies: Oscillator frequencies (Hz).
      Shape [batch_size, :, n_harmonics].
  """
  f_ratios = torch.linspace(1.0, 100.0, 100)
  f_ratios = f_ratios.unsqueeze(0).unsqueeze(0)
  harmonic_frequencies = f0 * f_ratios
  return harmonic_frequencies


# transcribed from magenta's tensorflow implementation
def remove_above_nyquist(harmonics, cn, sr=16000):
    condition = torch.ge(harmonics, sr/2)
    cn = torch.where(condition, torch.zeros_like(cn), cn)
    return cn


# transcribed from magenta's tensorflow implementation
def safe_divide(numerator, denominator, eps=1e-7):
  """Avoid dividing by zero by adding a small epsilon."""
  safe_denominator = torch.where(denominator == 0.0, eps, denominator)
  return numerator / safe_denominator