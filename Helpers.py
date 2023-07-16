import torch
import math
import torch.nn as nn
import librosa as li
import numpy as np
from torch.utils.data import Dataset, Subset
from sklearn.model_selection import train_test_split
import os

#### Helper functions ####
def round_down(samples, multiple):
    return multiple * math.floor(samples / multiple)


def slice_audio(x, step):
  slices = []
  for i in range(step, len(x)+1, step):
    slices += x[i-step:i].unsqueeze(0)
  slices = torch.stack(slices)
  return slices

def UpsampleTime(x, fs=16000):
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


#acids code
def upsample(signal, factor):
    signal = signal.permute(0, 2, 1)
    signal = nn.functional.interpolate(signal, size=signal.shape[-1] * factor)
    return signal.permute(0, 2, 1)


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


### Load data ###
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


