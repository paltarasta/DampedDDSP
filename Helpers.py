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
  upsampled_timesteps = fs
  channels = x[0,0,:].shape[0]
  upsampled = torch.zeros((batch, upsampled_timesteps, channels))#.cuda()

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


def MakeMelsTensor(sliced_x, audio_fs, mel_tensor, is_synthetic,n_fft=128, hop=32, n_mels=229):
  for clip in sliced_x:
    if is_synthetic == False:
      clip = np.array(clip)
    else:
      clip = np.squeeze(clip)
    mel = li.feature.melspectrogram(y=clip, sr=audio_fs, n_fft=2048, hop_length=128, n_mels=229)
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
    if idx > 200:
      break
    audio_path = audio_dir + track
    print('INDEX', idx)
    x, audio_fs = CutAudioLen(audio_path, sample_rate)
    
    step = audio_fs
    sliced_x = slice_audio(x, step) #slice the audio into 16000 sample long sections
    X += sliced_x

    ### to mel specs ###
    MELS = MakeMelsTensor(sliced_x, audio_fs, MELS, False)
    
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

### Create dataset class for synthetic data and controls ###
class SyntheticDataset(Dataset):
  def __init__(self,
               mels,
               audio_target,
               harm_amp_target,
               harm_dist_target,
               f0_target,
               sin_amps_target,
               sin_freqs_target):
    self.mels = mels
    self.audio_target = audio_target
    self.harm_amp_target = harm_amp_target
    self.harm_dist_target = harm_dist_target
    self.f0_target = f0_target
    self.sin_amps_target = sin_amps_target
    self.sin_freqs_target = sin_freqs_target

  def __len__(self):
    return len(self.audio_target)
  
  def __getitem__(self, idx):
    mels = self.mels[idx]
    audio_target = self.audio_target[idx]
    harm_amp_target = self.harm_amp_target[idx]
    harm_dist_target = self.harm_dist_target[idx]
    f0_target = self.f0_target[idx]
    sin_amps_target = self.sin_amps_target[idx]
    sin_freqs_target = self.sin_freqs_target[idx]
    return mels, audio_target, harm_amp_target, harm_dist_target, f0_target, sin_amps_target, sin_freqs_target

### Define a training testing split for the dataset ###
#https://discuss.pytorch.org/t/how-to-split-dataset-into-test-and-validation-sets/33987/4 - Manpreet Singh (msminhas93) May '20
def train_val_dataset(dataset, val_split=0.20):
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

### from here https://github.com/pytorch/pytorch/issues/71409
def shufflerow(tensor1, tensor2, axis):
    row_perm = torch.rand(tensor1.shape[:axis+1]).argsort(axis)  # get permutation indices
    row_perm_2 = torch.clone(row_perm)
    for _ in range(tensor1.ndim-axis-1): row_perm.unsqueeze_(-1)
    for _ in range(tensor2.ndim-axis-1): row_perm_2.unsqueeze_(-1)
    row_perm = row_perm.repeat(*[1 for _ in range(axis+1)], *(tensor1.shape[axis+1:]))  # reformat this for the gather operation
    row_perm_2 = row_perm_2.repeat(*[1 for _ in range(axis+1)], *(tensor2.shape[axis+1:]))
    tensor_a = tensor1.gather(axis, row_perm)
    tensor_b = tensor2.gather(axis, row_perm_2)
    return tensor_a, tensor_b


##############################################################
################  HARMONIC ENCODER HELPERS ###################
##############################################################

# transcribed from magenta's tensorflow implementation
def hz_to_midi(hz):
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
  f_ratios = torch.linspace(1.0, 100.0, 100)#.cuda()
  f_ratios = f_ratios.unsqueeze(0).unsqueeze(0)
  harmonic_frequencies = f0 * f_ratios
  return harmonic_frequencies


# transcribed from magenta's tensorflow implementation
def remove_above_nyquist(harmonics, cn, sr=16000):
    condition = torch.ge(harmonics, sr/2)#.cuda()
    cn = torch.where(condition, torch.zeros_like(cn), cn)#.cuda()
    return cn


# transcribed from magenta's tensorflow implementation
def safe_divide(numerator, denominator, eps=1e-7):
  """Avoid dividing by zero by adding a small epsilon."""
  safe_denominator = torch.where(denominator == 0.0, eps, denominator)#.cuda()
  return numerator / safe_denominator


#From Mohammed Ataaur Rahman
def print_model_stats(model):

    print(f"Model parameters: {sum([param.nelement() for param in model.parameters()])}")