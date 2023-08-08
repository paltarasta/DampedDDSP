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

pitches = torch.tensor([[[100.0, 200],[200.0, 300]]])
amplitudes = torch.linspace(1, 0.5, 2)[None, None]
amplitudes = amplitudes.repeat(1, 2, 1)
damping = torch.zeros(1, 2, 2)
#damping = torch.tensor([[[0.1,1], [0.2,2]]])

signal = s.damped_synth(pitches, amplitudes, damping, 16000)


pred_amplitudes = torch.rand((1, 2, 2), requires_grad=True)
pred_pitches = torch.rand((1, 2, 2)) * 500
pred_pitches.requires_grad_(True)
pred_damping = torch.randn(1, 2, 2, requires_grad=True)

fs = 16000

optimiser = torch.optim.Adam([pred_amplitudes, pred_damping, pred_pitches], lr = 0.001)

criterion = nn.MSELoss()
steps = 1000
mainloss = []

t = trange(steps)
for step in t:
  prediction = s.damped_synth(pred_pitches, pred_amplitudes, pred_damping, fs)
  loss = criterion(signal, prediction)
  optimiser.zero_grad()
  loss.backward()
  optimiser.step()

  mainloss.append(loss.item())
  t.set_description_str(f"{step}, loss = {loss.item():.4f}")

plt.plot(mainloss)
plt.figure()
plt.plot(signal.squeeze())
plt.figure()
plt.plot(prediction.squeeze().detach())