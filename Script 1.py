import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
import librosa as li
import matplotlib.pyplot as plt
from tqdm import trange
from IPython.display import Audio
import re
import einops
from einops import rearrange
import math
from scipy import signal
import os
from torch.utils.data import Dataset, DataLoader


os.environ["CUDA_VISIBLE_DEVICES"] = "1]"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on device: {device}")


#### Helper functions ####
def round_down(samples, multiple):
    return multiple * math.floor(samples / multiple)


def slice_audio(x, step):
  slices = []
  for i in range(step, len(x)+1, step):
    slices.append(x[i-step:i])

  return slices


def UpsampleCSV(csv, fs=16000):
  timesteps = csv.shape[0]
  upsampled_timesteps = 4 * fs
  upsampled = torch.zeros((1,upsampled_timesteps))

  j = 1

  for i in range(upsampled_timesteps):
    if j == timesteps and i < upsampled_timesteps - 1:
      upsampled[:, i] = upsampled[:, i-1]
    if i == 0:
      upsampled[:, i] = csv[i]
    elif i % np.floor(upsampled_timesteps/timesteps) == 0 and j < timesteps:
      upsampled[:, i] = csv[j]
      j += 1
    else:
      upsampled[:, i] = upsampled[:, i-1]

  return upsampled


def UpsampleTime(x, fs=16000):
  batch = x[:,0,0].shape[0]
  timesteps = x[0,:,0].shape[0]
  upsampled_timesteps = 4 * fs
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
  
audio_dir = 'C:/Users/eat_m/Documents/QMUL/Summer_Project/MDB-stem-synth/audio_stems/'
annot_dir = 'C:/Users/eat_m/Documents/QMUL/Summer_Project/MDB-stem-synth/annotation_stems/'

audio_tracks = os.listdir(audio_dir)

X = []
Y = []
MELS = []

for track in audio_tracks:
  audio_path = audio_dir + track
  annotation_path = annot_dir + track.strip('wav') + 'csv'
  if 'AClassicEducation_NightOwl' in track:
    ### audio ###
    x_original, audio_fs = li.load(audio_path, sr=16000)
    x = torch.tensor(x_original)
    floored_length = round_down(len(x), 4*audio_fs)

    x = x[:floored_length]

    step = 4*audio_fs
    sliced_x = slice_audio(x_original, step)

    ### to mel specs ###
    for clip in sliced_x:
      target = clip
      mel = li.feature.melspectrogram(y=target, sr=audio_fs, n_fft=2048, hop_length=514, n_mels=229)
      mel_db = torch.tensor(li.power_to_db(mel, ref=np.max)).T.unsqueeze(0) #channel, time, frequency
      MELS.append(mel_db)

    X += sliced_x

    ### annotation ###

    csv = open(annotation_path, 'r')
    csv_data, times = ReadCSV(csv)
    audio_len = len(x)

    csv_fs = int(np.floor(len(csv_data)/times[-1]))
    duration = audio_len/audio_fs
    csv_len = int((csv_fs * audio_len) / audio_fs)
    csv_data = torch.tensor(csv_data[:csv_len])
    csv_step = 4*csv_fs

    sliced_csv = slice_audio(csv_data, csv_step)

    for snippet in sliced_csv:
      upsampled_csv = UpsampleCSV(snippet)
      Y += upsampled_csv

myDataset = CustomDataset(MELS, Y)
DL_DS = DataLoader(myDataset, batch_size=16, shuffle=True)


class ResidualBlock(nn.Module):

    def __init__(self, in_channels, out_channels, time, freq, stride):
        super(ResidualBlock, self).__init__()
        self.first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)
        self.other_first_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=(1, stride), padding=(0,1))
        self.layer_norm1 = nn.LayerNorm([in_channels, time, freq])
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels//4, kernel_size=1, stride=1) #kernel = 1 means no spatial reduction

        self.layer_norm2 = nn.LayerNorm([out_channels//4, time, freq])
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels//4, out_channels//4, kernel_size=(1,3), stride=(1,stride))

        self.layer_norm3 = nn.LayerNorm([out_channels//4, time, freq-2])
        self.layer_norm4 = nn.LayerNorm([out_channels//4, time, ((freq-3)//stride)+1])
        self.relu3 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels//4, out_channels, kernel_size=1, stride=1, padding=(0,1))

        self.residual = nn.Identity()
        self.stride = stride

    def forward(self, x):
      if self.stride == 2:
        res = self.other_first_conv(x)
      else:
        res = self.first_conv(x)

      residual = self.residual(res)

      out = self.layer_norm1(x)
      out = self.relu1(out)
      out = self.conv1(out)
      print('conv1, ', out.shape)

      out = self.layer_norm2(out)
      out = self.relu2(out)
      out = self.conv2(out)
      print('conv2, ', out.shape)
      if self.stride == 2:
        out = self.layer_norm4(out)
      else:
        out = self.layer_norm3(out)
      out = self.relu3(out)
      out = self.conv3(out)
      print('conv3, ', out.shape)

      out += residual

      return out


class ResNet(nn.Module):
    """Residual network."""

    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Conv2d(1, 64, kernel_size=7, stride=(1,2), padding=3)
        self.maxpool = nn.MaxPool2d((1,3), stride=(1,2))

        self.residual_block1 = ResidualBlock(64, 128, 125, 57, 1)
        self.residual_block2 = ResidualBlock(128, 128, 125, 57, 1)
        self.residual_block3 = ResidualBlock(128, 256, 125, 57, 2)
        self.residual_block4 = ResidualBlock(256, 256, 125, 30, 1)
        self.residual_block5 = ResidualBlock(256, 256, 125, 30, 1)
        self.residual_block6 = ResidualBlock(256, 512, 125, 30, 2)
        self.residual_block7 = ResidualBlock(512, 512, 125, 16, 1)
        self.residual_block8 = ResidualBlock(512, 512, 125, 16, 1)
        self.residual_block9 = ResidualBlock(512, 512, 125, 16, 1)
        self.residual_block10 = ResidualBlock(512, 1024, 125, 16, 2)
        self.residual_block11 = ResidualBlock(1024, 1024, 125, 9, 1)
        self.residual_block12 = ResidualBlock(1024, 1024, 125, 9, 1)


    def forward(self, x):
        print('in,', x.shape)
        out = self.conv(x)
        print('post conv, ', out.shape)
        out = self.maxpool(out)
        print('post maxpool, ', out.shape)

        out = self.residual_block1(out)
        print('res1, ', out.shape)
        #out = self.residual_block2(out)
        out = self.residual_block3(out)
        #out = self.residual_block4(out)
        #out = self.residual_block5(out)
        print('res5, ', out.shape)
        out = self.residual_block6(out)
        #out = self.residual_block7(out)
        #out = self.residual_block8(out)
        #out = self.residual_block9(out)
        out = self.residual_block10(out)
        #out = self.residual_block11(out)
        out = self.residual_block12(out)
        print('res12, ', out.shape)

        #out = F.normalize(out, dim=2)
        #print('post norm', out)

        return out
    

class MapToFrequency(nn.Module):

  def __init__(self):

    super(MapToFrequency, self).__init__()

    self.resnet = ResNet()
    self.dense = nn.Linear(9*1024, 64)
    self.register_buffer("scale", torch.logspace(20, 8000, 64, base=2.0))
    self.softmax = nn.Softmax(dim=2)

  def forward(self, x):
    out = self.resnet(x)
    #print(out.shape)
    out = rearrange(out, 'z a b c -> z b c a')
    print(out.shape)
    out = torch.reshape(out, (out.shape[0], 125, 9*1024))
    print('reshape', out.shape)

    out = self.dense(out)
    #print('dense', out.shape)
    out = UpsampleTime(out)
    #print('upsample', out.shape)
    out = self.softmax(out)
    #print('softmax',out.shape)
    out = torch.sum(out * self.scale, dim=-1)
    #print('scaled;, out.shape')


    return out 
  

model = MapToFrequency()
criterion = nn.MSELoss()

# Instantiate the optimizer (e.g., Adam optimizer) and specify learning rate
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
num_epochs = 3
for epoch in range(num_epochs):
    running_loss = []

    for inputs, targets in DL_DS:

      optimizer.zero_grad()

      y_pred = model(inputs)

      y_target = targets

      loss = criterion(y_pred, y_target)

      loss.backward()
      optimizer.step()

      running_loss.append(loss.item())
      print('end of loop')

plt.plot(running_loss)