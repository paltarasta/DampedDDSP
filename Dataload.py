import torch
import torch.nn as nn
from Synths import damped_synth
import Helpers as h
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
import os
from torch.utils.data import DataLoader
### Load data ###

if 'Y' not in globals():

    audio_dir = "C:/Users/eat_m/Documents/QMUL/Summer_Project/MDB-stem-synth/audio_stems/"

    audio_path = os.listdir(audio_dir)

    MELS_norm, Y = h.LoadAudio(audio_dir, 16000)
else:
    print('HI SCR')

print(Y.shape)
print(MELS_norm.shape)

torch.save(MELS_norm, 'meltensor.pt')
torch.save(Y, 'y.pt')