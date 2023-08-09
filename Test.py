import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import Helpers as h
import Nets as n
import auraloss as al
from tqdm import tqdm, trange
import Synths as s
from einops import rearrange
import matplotlib.pyplot as plt
import os
import Losses as l
from torch.utils.tensorboard import SummaryWriter

# mixed dataset loader
'''
MELS = torch.load('SavedTensors/meltensor.pt')
MELS = h.shufflerow(MELS,0)
synthMELS = torch.load('SavedTensors/melsynth.pt')
synthMELS = h.shufflerow(synthMELS, 0)

total_mels = torch.cat((MELS[:7000], synthMELS[:7000]),dim=0)
total_mels = h.shufflerow(total_mels, 0)
print(total_mels.shape)
'''

a = torch.tensor([[1,2,3],[4,5,6],[7,8,9],[10,11,12]])
b = torch.tensor([[10,20,30],[40,50,60],[70,80,90],[100,110,120]])
print(a.shape)

x, y = h.shufflerow(a, b, 0)
print(x)
print(y)
