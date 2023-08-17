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

if __name__ == "__main__":

  ### Load tensors and create dataloader ###
  MELS = torch.load('SavedTensors/melsynth_125.pt')[:100]
  Y = torch.load('SavedTensors/ysynth_125.pt')[:100]

  mixed_dataset = h.CustomDataset(MELS, Y)
  datasets = h.train_val_dataset(mixed_dataset)
  print('Train', len(datasets['train']))
  print('Val', len(datasets['val']))

  DL_DS = {x:DataLoader(datasets[x], 1, shuffle=True, num_workers=2) for x in ['train','val']} #num_worker should = 4 * num_GPU
  
  ### Set up ###
  damp_encoder = n.DampingMapping()#.cuda()
  damp_harm_encoder = n.DampSinToHarmEncoder()#.cuda()

  ### Each "window" containing one damping parameter is 128 samples long (upsampling from 125 to 16000), so
  damp_sin_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[256, 256, 256], win_lengths=[256, 256, 256])#.cuda()
  harm_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[256, 256, 256], win_lengths=[256, 256, 256])#.cuda()
  consistency_criterion = l.KDEConsistencyLoss

  params = list(damp_encoder.parameters()) + list(damp_harm_encoder.parameters())
  optimizer = torch.optim.Adam(params, lr=0.0001)

  # Training loop
  num_epochs = 1
  i = 0
  j = 0

  for epoch in range(num_epochs):
    print('Epoch', epoch)

    with tqdm(DL_DS['train'], unit='batch') as tepoch:
      running_loss = []
      sin_recon_running_loss = []
      harm_recon_running_loss = []
      consistency_running_loss = []

      for mels, audio in tepoch:
        if i > 1:
          break
        print(audio.shape, 'THE TARGETS SHAPE')
        print(mels.shape, 'THE INPUTS SHAPE')

        mels = mels#.cuda()
        audio = audio#.cuda()
            
        #Damped sinusoisal encoder
        sin_freqs, sin_amps, sin_damps = damp_encoder(mels)
        print(sin_freqs)

        #Damped sinusoidal synthesiser
        fs = 16000
        old_fs = sin_freqs.size(1)
        factor = fs // old_fs
        i += 1






























'''

        upsampled_sin_freqs = h.UpsampleTime(sin_freqs, fs)
        upsampled_sin_amps = h.UpsampleTime(sin_amps, fs)
        upsampled_sin_damps = h.upsample_to_damper(sin_damps, factor)

        damp_sin_signal = s.damped_synth(upsampled_sin_freqs, upsampled_sin_amps, upsampled_sin_damps, fs)
        damp_sin_signal = rearrange(damp_sin_signal, 'a b c -> a c b')

        #Harmonic encoder
        sin_freqs = sin_freqs.detach() #detach gradients before they go into the harmonic encoder
        sin_amps = sin_amps.detach()
        sin_damps = sin_damps.detach() #do we need to do this?
        glob_amp, harm_dist, f0 = damp_harm_encoder(sin_freqs, sin_amps, sin_damps)

        #Reconstruct audio from harmonic encoder results
        harmonics = h.get_harmonic_frequencies(f0) #need this to then do the sin synth - creates a bank of 100 sinusoids
        harm_dist = h.remove_above_nyquist(harmonics, harm_dist) #only keep the frequencies which are below the nyquist criterion, set amplitudes of other freqs to 0
        harm_dist = h.safe_divide(harm_dist, torch.sum(harm_dist, dim=-1, keepdim=True)) #normalise
        harm_amps = glob_amp * harm_dist

        upsampled_harmonics = h.UpsampleTime(harmonics, fs)
        upsampled_harm_amps = h.UpsampleTime(harm_amps, fs)

        harm_signal = s.sinusoidal_synth(upsampled_harmonics, upsampled_harm_amps, fs) #if we use the sinusoidal synth it defeats the purpose?
        harm_signal = rearrange(harm_signal, 'a b c -> a c b')
'''
       