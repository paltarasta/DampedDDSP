import torch
import Helpers as h
import os
import matplotlib.pyplot as plt

harm = torch.load('Outputs/harm_signal_exp_10_400.pt', map_location=torch.device('cpu'))
print(harm.squeeze().shape)

plt.figure(1)
plt.plot(harm.squeeze().detach())
plt.show()

sin = torch.load('Outputs/damp_sin_signal_exp_10_400.pt', map_location=torch.device('cpu'))
print(sin.shape)
plt.figure(2)
plt.plot(sin.squeeze().detach())
plt.show()
