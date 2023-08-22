import torch
import Helpers as h
import os
import matplotlib.pyplot as plt
'''
audio = torch.load('Outputs/audio_exp2_0_4000.pt', map_location=torch.device('cpu'))
print('audio', audio.shape)

sin = torch.load('Outputs/damp_sin_signal_exp3_0_4000.pt', map_location=torch.device('cpu'))
print('sin', sin.shape)

harm = torch.load('Outputs/harm_signal_exp3_0_4000.pt', map_location=torch.device('cpu'))

plt.figure(1)
plt.plot(audio.squeeze())
plt.title('audio')
plt.show()
plt.figure(2)
plt.plot(sin.squeeze().detach())
plt.title('sin synth')
plt.show()
plt.figure(3)
plt.plot(harm.squeeze().detach())
plt.title('harm synth')
plt.show()
'''
sin = torch.load('Outputs/damp_sin_signal_exp2_1_1000.pt', map_location=torch.device('cpu')).squeeze().detach()
sin1 = torch.load('Outputs/damp_sin_signal_exp2_1_1500.pt', map_location=torch.device('cpu')).squeeze().detach()
sin2 = torch.load('Outputs/damp_sin_signal_exp2_1_2000.pt', map_location=torch.device('cpu')).squeeze().detach()
sin3 = torch.load('Outputs/damp_sin_signal_exp2_0_9000.pt', map_location=torch.device('cpu')).squeeze().detach()
sin4 = torch.load('Outputs/damp_sin_signal_exp2_0_9500.pt', map_location=torch.device('cpu')).squeeze().detach()
sin5 = torch.load('Outputs/damp_sin_signal_exp2_0_10000.pt', map_location=torch.device('cpu')).squeeze().detach()


plt.subplot(2,3,1)
plt.plot(sin)
plt.subplot(2,3,2)
plt.plot(sin1)
plt.subplot(2,3,3)
plt.plot(sin2)
plt.subplot(2,3,4)
plt.plot(sin3)
plt.subplot(2,3,5)
plt.plot(sin4)
plt.subplot(2,3,6)
plt.plot(sin5)
plt.show()


sin_freqs = torch.load('Outputs/sin_freqs_exp2_0_10000.pt', map_location=torch.device('cpu'))
sin_freqs2 = torch.load('Outputs/sin_freqs_exp2_1_1000.pt', map_location=torch.device('cpu'))
sin_freqs3 = torch.load('Outputs/sin_freqs_exp2_0_8000.pt', map_location=torch.device('cpu'))

print('One', sin_freqs)
print('Two', sin_freqs2)
print('Three', sin_freqs3)
'''

sin_amps = torch.load('Outputs/sin_amps_exp2_0_10000.pt', map_location=torch.device('cpu'))
sin_amps2 = torch.load('Outputs/sin_amps_exp2_1_1000.pt', map_location=torch.device('cpu'))
sin_amps3 = torch.load('Outputs/sin_amps_exp2_0_8000.pt', map_location=torch.device('cpu'))

print('One', sin_amps[:,0,:])
print('Two', sin_amps2)
print('Three', sin_amps3)

harmonics = torch.load('Outputs/harmonics_exp2_0_10000.pt', map_location=torch.device('cpu'))
harmonics2 = torch.load('Outputs/harmonics_exp2_1_1000.pt', map_location=torch.device('cpu'))
harmonics3 = torch.load('Outputs/harmonics_exp2_0_8000.pt', map_location=torch.device('cpu'))

print('One', harmonics)
print('Two', harmonics2)
print('Three', harmonics3)


harm_amps = torch.load('Outputs/harm_amps_exp2_0_10000.pt', map_location=torch.device('cpu'))
harm_amps2 = torch.load('Outputs/harm_amps_exp2_1_1000.pt', map_location=torch.device('cpu'))
harm_amps3 = torch.load('Outputs/harm_amps_exp2_0_8000.pt', map_location=torch.device('cpu'))

print('One', harm_amps)
print('Two', harm_amps2)
print('Three', harm_amps3)
'''