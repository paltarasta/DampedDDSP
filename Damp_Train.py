import torch
from torch.utils.data import DataLoader
import Helpers as h
import Nets as n
import auraloss as al
from tqdm import tqdm
import Synths as s
from einops import rearrange
import os
import Losses as l
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('trainval')
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

  MELS = torch.cat((MELS_synth[:7], MELS_real[:7]), dim=0)
  Y = torch.cat((Y_synth[:7], Y_real[:7]), dim=0)
  MELS, Y = h.shufflerow(MELS, Y, 0)

  mixed_dataset = h.CustomDataset(MELS, Y)
  datasets = h.train_val_dataset(mixed_dataset)
  print('Train', len(datasets['train']))
  print('Val', len(datasets['val']))

  DL_DS = {x:DataLoader(datasets[x], 1, shuffle=True, num_workers=2) for x in ['train','val']} #num_worker should = 4 * num_GPU
  
  ### Set up ###
  damp_encoder = n.DampingMapping().cuda()
  damp_harm_encoder = n.DampSinToHarmEncoder().cuda()

  ### Each "window" containing one damping parameter is 128 samples long (upsampling from 125 to 16000), so
  ### Experiment with different window sizes e.g. all 3 the same?
  damp_sin_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[256, 256, 256], win_lengths=[256, 256, 256]).cuda()
  harm_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[256, 256, 256], win_lengths=[256, 256, 256]).cuda()
  consistency_criterion = l.KDEConsistencyLoss
# mate you forgot to scale the consistency loss man
  params = list(damp_encoder.parameters()) + list(damp_harm_encoder.parameters())
  optimizer = torch.optim.Adam(params, lr=0.0003)

  # Training loop
  num_epochs = 5
  i = 0

  for epoch in range(num_epochs):
    print('Epoch', epoch)

    with tqdm(DL_DS['train'], unit='batch') as tepoch:
      running_loss = []

      for mels, audio in tepoch:
        print(audio.shape, 'THE TARGETS SHAPE')
        print(mels.shape, 'THE INPUTS SHAPE')

        mels = mels.cuda()
        audio = audio.cuda()
            
        #Damped sinusoisal encoder
        sin_freqs, sin_amps, sin_damps = damp_encoder(mels)

        #Damped sinusoidal synthesiser
        fs = 16000
        old_fs = sin_freqs.size(1)
        factor = fs // old_fs

        upsampled_sin_freqs = h.UpsampleTime(sin_freqs, fs)
        upsampled_sin_amps = h.UpsampleTime(sin_amps, fs)
        upsampled_sin_damps = h.upsample_to_damper(sin_damps, factor)

        damp_sin_signal = s.damped_synth(upsampled_sin_freqs, upsampled_sin_amps, upsampled_sin_damps, fs)
        damp_sin_signal = rearrange(damp_sin_signal, 'a b c -> a c b')

        #Sinusoidal reconstruction loss
        sin_recon_loss = damp_sin_criterion(damp_sin_signal, audio.unsqueeze(0))
        writer.add_scalar('Sinusoidal Recon loss/train', sin_recon_loss, i)

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

        #Harmonic reconstruction loss
        harm_recon_loss = harm_criterion(harm_signal, audio.unsqueeze(0))
        writer.add_scalar('Harmonic Recon loss/train', harm_recon_loss, i)

        #Consistency loss
        consistency_loss = consistency_criterion(harm_amps, harmonics, sin_amps, sin_freqs)
        writer.add_scalar('Consistency loss/train', consistency_loss, i)

        #Total loss
        total_loss = sin_recon_loss + harm_recon_loss + (0.1 * consistency_loss)
        writer.add_scalar('Loss/train', total_loss, i)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss.append(total_loss.item())
        tepoch.set_description_str(f"{epoch}, loss = {total_loss.item():.4f}", refresh=True)

        i += 1

        # Save a checkpoint

        if i%100 == 0:
          torch.save(damp_encoder.state_dict(), f'Checkpoints/damp_encoder_ckpt_exp1_{epoch}_{i}.pt')
          torch.save(damp_harm_encoder.state_dict(), f'Checkpoints/damp_harm_encoder_ckpt_exp_1{epoch}_{i}.pt')
          torch.save(damp_sin_signal, f'Outputs/damp_sin_signal_exp_1{epoch}_{i}.pt')
          torch.save(harm_signal, f'Outputs/harm_signal_exp_1{epoch}_{i}.pt')

    ##################
    ### Validation ###
    ##################

    with torch.no_grad():
      val_loss = []

      for mels_val, audio_val in DL_DS['val']:
        mels_val = mels_val.cuda()
        audio_val = audio_val.cuda()
        
        #Damped sinusoisal encoder
        sin_freqs_val, sin_amps_val, sin_damps_val = damp_encoder(mels_val)

        #Damped sinusoidal synthesiser
        fs_val = 16000
        old_fs_val = sin_freqs_val.size(1)
        factor_val = fs_val // old_fs_val

        upsampled_sin_freqs_val = h.UpsampleTime(sin_freqs_val, fs_val)
        upsampled_sin_amps_val = h.UpsampleTime(sin_amps_val, fs_val)
        upsampled_sin_damps_val = h.upsample_to_damper(sin_damps_val, factor_val)

        damp_sin_signal_val = s.damped_synth(upsampled_sin_freqs_val, upsampled_sin_amps_val, upsampled_sin_damps_val, fs_val)
        damp_sin_signal_val = rearrange(damp_sin_signal_val, 'a b c -> a c b')

        #Sinusoidal reconstruction loss
        sin_recon_loss_val = damp_sin_criterion(damp_sin_signal_val, audio_val.unsqueeze(0))
        writer.add_scalar('Sinusoidal Recon loss/val', sin_recon_loss_val, i)

        #Harmonic encoder
        sin_freqs_val = sin_freqs_val.detach() #detach gradients before they go into the harmonic encoder
        sin_amps_val = sin_amps_val.detach()
        sin_damps_val = sin_damps_val.detach() #do we need to do this?
        glob_amp_val, harm_dist_val, f0_val, harm_damps_val = damp_harm_encoder(sin_freqs_val, sin_amps_val, sin_damps_val)

        #Reconstruct audio from harmonic encoder results
        harmonics_val = h.get_harmonic_frequencies(f0_val) #need this to then do the sin synth - creates a bank of 100 sinusoids
        harm_dist_val = h.remove_above_nyquist(harmonics_val, harm_dist_val) #only keep the frequencies which are below the nyquist criterion, set amplitudes of other freqs to 0
        harm_dist_val = h.safe_divide(harm_dist_val, torch.sum(harm_dist_val, dim=-1, keepdim=True)) #normalise
        harm_amps_val = glob_amp_val * harm_dist_val

        upsampled_harmonics_val = h.UpsampleTime(harmonics_val, fs_val)
        upsampled_harm_amps_val = h.UpsampleTime(harm_amps_val, fs_val)

        harm_signal_val = s.sinusoidal_synth(upsampled_harmonics_val, upsampled_harm_amps_val, fs_val) #if we use the sinusoidal synth it defeats the purpose?
        harm_signal_val = rearrange(harm_signal_val, 'a b c -> a c b')

        #Harmonic reconstruction loss
        harm_recon_loss_val = harm_criterion(harm_signal_val, audio_val.unsqueeze(0))
        writer.add_scalar('Harmonic Recon loss/val', harm_recon_loss_val, i)

        #Consistency loss
        consistency_loss_val = consistency_criterion(harm_amps_val, harmonics_val, sin_amps_val, sin_freqs_val)
        writer.add_scalar('Consistency loss/train', consistency_loss_val, i)

        #Total loss
        total_loss_val = sin_recon_loss_val + harm_recon_loss_val + (0.1 * consistency_loss_val)
        writer.add_scalar('Loss/val', total_loss_val, i)

        val_loss.append(total_loss_val)


  writer.flush()
  writer.close()
