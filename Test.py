import torch
import torch.nn as nn
import Helpers as h
import Nets as n
import Synths as s
import matplotlib.pyplot as plt
from einops import rearrange
import Losses as l
import auraloss as al

print('start')
sin_encoder = n.SinMapToFrequency()
harm_encoder = n.SinToHarmEncoder()

sin_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512])#.cuda()
harm_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512])#.cuda()

params = list(sin_encoder.parameters()) + list(harm_encoder.parameters())
optimizer = torch.optim.Adam(params, lr=0.03)

print('model stats')
h.print_model_stats(harm_encoder)

inputs = torch.rand(4,1,126,229)
targets = torch.rand(4, 1, 4000)

for i in range(4):
    #Sinusoisal encoder

    sins, amps = sin_encoder(inputs[i].unsqueeze(0))

    print('post sin encoder')

    #Sinusoidal synthesiser
    #damped_signal = damped_synth(sins, amps, damps, 16000)
    sin_signal = s.sinusoidal_synth(sins, amps, 16000)
    sin_signal = rearrange(sin_signal, 'a b c -> a c b')
    print('post sin synth')

    #First reconstruction loss
    sin_recon_loss = sin_criterion(sin_signal, targets[i].unsqueeze(0))

    #Harmonic encoder
    sins = sins.detach() #detach gradients before they go into the harmonic encoder
    amps = amps.detach()

    harmonics, harm_amps, harm_damps = harm_encoder(sins, amps)
    print('post harm encoder')

    #harm_signal = damped_synth(harmonics, harm_amps, harm_damps, 16000)
    harm_signal = s.sinusoidal_synth(harmonics, harm_amps, 16000)
    harm_signal = rearrange(harm_signal, 'a b c -> a c b')
    
    #Second reconstruction loss
    harm_recon_loss = harm_criterion(harm_signal, targets[i].unsqueeze(0))
    print('post harm loss')

    consistency_loss = l.KDEConsistencyLoss(harm_amps, harmonics, amps, sins)
    print('consistency loss', consistency_loss)
    
    total_loss = sin_recon_loss + harm_recon_loss + consistency_loss
    print('after losses all summed', total_loss)
    
    print('starting backwards')
    optimizer.zero_grad()
    total_loss.backward()
    print('backwards done')
    optimizer.step()
    print('finished')
