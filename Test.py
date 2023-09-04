import torch
import Helpers as h
import os
import matplotlib.pyplot as plt
import numpy as np
import Synths as s

harm_signal = torch.load('Normal/Outputs/harm_signal_normal_0_225.pt', map_location=torch.device('cpu'))
damp_sin_signal = torch.load('Normal/Outputs/damp_sin_signal_normal_0_225.pt', map_location=torch.device('cpu'))
audio = torch.load('Normal/Outputs/audio_normal0_225.pt', map_location=torch.device('cpu'))
harm_amps = torch.load('Normal/Outputs/harm_amps_normal_0_225.pt', map_location=torch.device('cpu'))
sin_amps = torch.load('Normal/Outputs/sin_amps_normal_0_225.pt', map_location=torch.device('cpu'))
sin_damps = torch.load('Normal/Outputs/sin_damps_normal_0_225.pt', map_location=torch.device('cpu'))
sin_freqs = torch.load('Normal/Outputs/sin_freqs_normal_0_225.pt', map_location=torch.device('cpu'))

exponent = torch.load('Normal/Outputs/exponent_normal_225.pt', map_location=torch.device('cpu'))

print(torch.max(sin_amps), torch.max(sin_damps), torch.max(exponent), torch.max(harm_amps), torch.max(damp_sin_signal), torch.max(harm_signal), torch.max(audio))

plt.figure()
x = np.linspace(0, 16000, 16000)


print('shapes', sin_freqs.shape, sin_amps.shape)
sin = s.sinusoidal_synth(sin_freqs[0].unsqueeze(0), sin_amps[0].unsqueeze(0), 16000)
print(sin.shape)
plt.plot(x[:256], sin.squeeze().detach()[:256], label='Sinusoidal synth')

plt.plot(x[:256], damp_sin_signal[0].squeeze().detach()[:256], label='Damped synth')
plt.ylabel('Amplitude')
plt.xlabel('Samples')
plt.legend(loc="upper left")
plt.show()