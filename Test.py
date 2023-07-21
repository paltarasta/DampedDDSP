import torch
import Helpers as h
import Nets as n
import Synths as s
import matplotlib.pyplot as plt

test = torch.rand(1,126,100)
test2 = torch.rand(1,126,100)
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