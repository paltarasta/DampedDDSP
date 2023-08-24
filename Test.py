import torch
import Helpers as h
import os
import matplotlib.pyplot as plt

audio = torch.load('SavedTensors\y_125.pt')
print(audio[0].shape)

print(torch.max(audio))