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

pitch = torch.rand(1, 125, 100)*500
interpolated = h.UpsampleTime(pitch, 16000)
pitch = interpolated

amplitudes = torch.rand(1, 125, 100)
amplitudes = h.UpsampleTime(amplitudes, 16000)

fs = 16000

indices = torch.arange(pitch.size(1)).unsqueeze(0).unsqueeze(-1)
damping = torch.zeros(1, 125, 100)
target = s.damped_synth(pitch, amplitudes, damping, fs, 16000//125)

pred_amplitudes = torch.rand((1, 125, 100))
pred_pitches = torch.rand((1, 125, 100))
pred_damping = torch.randn(1, 125, 100, requires_grad=True)

upsampled_pitches = h.UpsampleTime(pred_pitches, 16000).requires_grad_(True)
upsampled_amplitudes = h.UpsampleTime(pred_amplitudes, 16000).requires_grad_(True)

optimiser = torch.optim.Adam([upsampled_amplitudes, upsampled_pitches, pred_damping], lr = 0.001)

criterion = nn.MSELoss()
steps = 5000
mainloss = []

t = trange(steps)
for step in t:
  prediction = s.damped_synth(upsampled_pitches, upsampled_amplitudes, pred_damping, fs, 16000//125)
  loss = criterion(target, prediction)
  optimiser.zero_grad()
  loss.backward()
  optimiser.step()
  
  mainloss.append(loss.item())
  t.set_description_str(f"{step}, loss = {loss.item():.4f}")

plt.figure(1)
plt.plot(mainloss)
plt.show()
plt.figure(2)
plt.plot(target.squeeze())
plt.plot(prediction.squeeze().detach())
plt.show()