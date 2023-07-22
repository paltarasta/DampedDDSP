import torch
import Helpers as h
import Nets as n
import Synths as s
import matplotlib.pyplot as plt
from einops import rearrange
'''
test = torch.rand(1,126,100)
test2 = torch.rand(1,126,100)
test2 = h.unit_scale_hz(test2)
comb = torch.cat((test, test2), -1)


model = n.SinToHarmEncoder()
harm_a, c, f0, d = model(comb)
#print(harm_a.shape, c.shape, f0.shape)

harmonics = h.get_harmonic_frequencies(f0) #need this to then do the sin synth - creates a bank of 100 sinusoids

cn = h.remove_above_nyquist(harmonics, c) #only keep the frequencies which are below the nyquist criterion, set amplitudes of other freqs to 0
cn = h.safe_divide(cn, torch.sum(cn, dim=-1, keepdim=True)) #normalise
print('cn', cn.shape)
#End of encoder as in magenta


#Synth
amplitudes = harm_a * cn
print(amplitudes.shape)
print(harmonics.shape)
#Then using harm_a, cn, f0 it is possible to generate a harmonic sound. But you don't need this to continue DDSP-inv
#Use the sinusoidal synthesiser to generate audio
harmonics = h.UpsampleTime(harmonics)
amplitudes = h.UpsampleTime(amplitudes)

signal = s.sinusoidal_synth(harmonics, amplitudes, 16000)
plt.plot(signal.squeeze())
plt.show()'''

sin_encoder = n.SinMapToFrequency()
harm_encoder = n.SinToHarmEncoder()

inputs = torch.rand(1,1,126,229)
#Sinusoisal encoder
sins, amps = sin_encoder(inputs)

#Sinusoidal synthesiser
#damped_signal = damped_synth(sins, amps, damps, 16000)
sin_signal = s.sinusoidal_synth(sins, amps, 16000)
sin_signal = rearrange(sin_signal, 'a b c -> a c b')

#First reconstruction loss
#sin_recon_loss = criterion(sin_signal.to(device), targets.unsqueeze(0).to(device))

#Harmonic encoder
harm_input = torch.cat((sins, amps), -1)
global_amps, amp_coeffs, f0s, harm_damps = harm_encoder(harm_input)

harmonics = h.get_harmonic_frequencies(f0s) #need this to then do the sin synth - creates a bank of 100 sinusoids
amp_coeffs = h.remove_above_nyquist(harmonics, amp_coeffs) #only keep the frequencies which are below the nyquist criterion, set amplitudes of other freqs to 0
amp_coeffs = h.safe_divide(amp_coeffs, torch.sum(amp_coeffs, dim=-1, keepdim=True)) #normalise
harm_amps = global_amps * amp_coeffs

#harm_signal = damped_synth(harmonics, harm_amps, harm_damps, 16000)
harm_signal = s.sinusoidal_synth(harmonics, harm_amps, 16000)