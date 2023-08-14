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

writer = SummaryWriter()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"Running on device: {device}")

#################################################################################################
######################################## DAMPED TRAINING ########################################
#################################################################################################
if __name__ == "__main__":

  ### Load tensors and create dataloader ###
  MELS_synth = torch.load('SavedTensors/melsynth_125.pt')
  MELS_real = torch.load('SavedTensors/meltensor_125.pt')
  Y_synth = torch.load('SavedTensors/ysynth_125.pt')
  Y_real = torch.load('SavedTensors/y_125.pt').unsqueeze(1)
  print(MELS_synth.shape)
  print(MELS_real.shape)
  print(Y_synth.shape)
  print(Y_real.shape)

  MELS_synth, Y_synth = h.shufflerow(MELS_synth, Y_synth, 0)
  MELS_real, Y_real = h.shufflerow(MELS_real, Y_real, 0)

  MELS = torch.cat((MELS_synth[:7000], MELS_real[:7000]), dim=0)
  Y = torch.cat((Y_synth[:7000], Y_real[:7000]), dim=0)
  MELS, Y = h.shufflerow(MELS, Y, 0)

  mixed_dataset = h.CustomDataset(MELS, Y)
  datasets = h.train_val_dataset(mixed_dataset)
  print('Train', len(datasets['train']))
  print('Val', len(datasets['val']))

  DL_DS = {x:DataLoader(datasets[x], 1, shuffle=True, num_workers=2) for x in ['train','val']} #num_worker should = 4 * num_GPU
  
  ### Set up ###
  damp_encoder = n.DampingMapping()#.cuda()
  damp_harm_encoder = n.DampSinToHarmEncoder()#.cuda()

  ### Each "window" containing one damping parameter is 128 samples long (upsampling from 125 to 16000), so
  ### the multi res loss will have windows which are multiples of 128.
  ### Experiment with different window sizes e.g. all 3 the same?
  damp_sin_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[256, 512, 128], win_lengths=[256, 512, 128])#.cuda()
  harm_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[256, 512, 128], win_lengths=[256, 512, 128])#.cuda()
  consistency_criterion = l.KDEConsistencyLoss#.cuda()
# mate you forgot to scale the consistency loss man
  params = list(damp_encoder.parameters()) + list(damp_harm_encoder.parameters())
  optimizer = torch.optim.Adam(params, lr=0.0003)

  # Training loop
  num_epochs = 2
  i = 0

  for epoch in range(num_epochs):
    print('Epoch', epoch)

    with tqdm(DL_DS['train'], unit='batch') as tepoch:
      running_loss = []

      for mels, audio in tepoch:
        print(audio.shape, 'THE TARGETS SHAPE')
        print(mels.shape, 'THE INPUTS SHAPE')

        #mels = mels.cuda()
        #audio = audio.cuda()

        if i > 4:
          break
            
        #Damped sinusoisal encoder
        sin_freqs, sin_amps, sin_damps = damp_encoder(mels)
        print('post damp encoder')

        #Damped sinusoidal synthesiser
        fs = 16000
        old_fs = sin_freqs.size(1)
        factor = fs // old_fs

        upsampled_sin_freqs = h.UpsampleTime(sin_freqs, fs)
        upsampled_sin_amps = h.UpsampleTime(sin_amps, fs)
        upsampled_sin_damps = h.upsample_to_damper(sin_damps, factor)

        damp_sin_signal = s.damped_synth(upsampled_sin_freqs, upsampled_sin_amps, upsampled_sin_damps, fs)
        damp_sin_signal = rearrange(damp_sin_signal, 'a b c -> a c b')
        print('post sin synth', damp_sin_signal.shape)

        #Sinusoidal reconstruction loss
        sin_recon_loss = damp_sin_criterion(damp_sin_signal, audio.unsqueeze(0))

        #Harmonic encoder
        sin_freqs = sin_freqs.detach() #detach gradients before they go into the harmonic encoder
        sin_amps = sin_amps.detach()
        sin_damps = sin_damps.detach() #do we need to do this?
        glob_amp, harm_dist, f0 = damp_harm_encoder(sin_freqs, sin_amps, sin_damps)
        print('post harm encoder')

        #Reconstruct audio from harmonic encoder results
        harmonics = h.get_harmonic_frequencies(f0) #need this to then do the sin synth - creates a bank of 100 sinusoids
        harm_dist = h.remove_above_nyquist(harmonics, harm_dist) #only keep the frequencies which are below the nyquist criterion, set amplitudes of other freqs to 0
        harm_dist = h.safe_divide(harm_dist, torch.sum(harm_dist, dim=-1, keepdim=True)) #normalise
        harm_amps = glob_amp * harm_dist
        #when you add the damped output as well, make sure to remove the nyquist damping amplitudes too
        #i.e. damps = h.remove_above_nyquist(harmonics, damps) but check how this works first
        #also how are we generating the damping? i suppose it's in the harm_dst shape right?

        upsampled_harmonics = h.UpsampleTime(harmonics, fs)
        upsampled_harm_amps = h.UpsampleTime(harm_amps, fs)

        harm_signal = s.damped_synth(upsampled_harmonics, upsampled_harm_amps, upsampled_sin_damps, fs) #if we use the sinusoidal synth it defeats the purpose?
        harm_signal = rearrange(harm_signal, 'a b c -> a c b')

        #Harmonic reconstruction loss
        harm_recon_loss = harm_criterion(harm_signal, audio.unsqueeze(0))
        print('post harm loss')

        #Consistency loss
        consistency_loss = consistency_criterion(harm_amps, harmonics, sin_amps, sin_freqs)
        print('sin loss', sin_recon_loss)
        print('harm loss', harm_recon_loss)
        print('consistency loss', consistency_loss)

        #Total loss
        total_loss = sin_recon_loss + harm_recon_loss + (0.1 * consistency_loss)
        print('after losses all summed', total_loss, type(total_loss), total_loss.shape)

        writer.add_scalar('Loss/train', total_loss, i)

        print('starting backwards')
        optimizer.zero_grad()
        total_loss.backward()
        print('backwards done')
        optimizer.step()
        print('finished')

        running_loss.append(total_loss.item())
        tepoch.set_description_str(f"{epoch}, loss = {total_loss.item():.4f}", refresh=True)

        i += 1
        break

      # Save a checkpoint
      torch.save(damp_encoder.state_dict(), f'Checkpoints/damp_encoder_ckpt_exp1_{epoch}.pt')
      torch.save(damp_harm_encoder.state_dict(), f'Checkpoints/damp_harm_encoder_ckpt_exp_1{epoch}.pt')

  writer.flush()
  writer.close()
