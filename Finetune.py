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

writer = SummaryWriter('finetune')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

#################################################################################################
########################################## FINETUNING ##########################################
#################################################################################################
if __name__ == "__main__":


  ### Load tensors and create dataloader ###
  MELS_synth = torch.load('SavedTensors/melsynth.pt')
  MELS_real = torch.load('SavedTensors/meltensor.pt')
  Y_synth = torch.load('SavedTensors/ysynth.pt')
  Y_real = torch.load('SavedTensors/y.pt').unsqueeze(1)
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
  
  DL_DS = {x:DataLoader(datasets[x], 16, shuffle=True, num_workers=2) for x in ['train','val']} #num_worker should = 4 * num_GPU

  ### Set up ###
  sin_encoder = n.SinMapToFrequency().cuda()
  harm_encoder = n.SinToHarmEncoder().cuda()

  ### Load pretrained weights ###
  sin_encoder.load_state_dict(torch.load("Pretrain\Checkpoints\sin_encoder_ckpt_1.pt"))
  harm_encoder.load_state_dict(torch.load("Pretrain\Checkpoints\harm_encoder_ckpt_1.pt"))

  sin_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512],
                                                  hop_sizes=[1024, 2048, 512],
                                                  win_lengths=[1024, 2048, 512],
                                                  mag_distance="L2",
                                                  w_log_mag=0.0,
                                                  w_lin_mag=1.0).cuda()
  harm_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512],
                                                  hop_sizes=[1024, 2048, 512],
                                                  win_lengths=[1024, 2048, 512],
                                                  mag_distance="L2",
                                                  w_log_mag=0.0,
                                                  w_lin_mag=1.0).cuda()
  consistency_criterion = l.KDEConsistencyLoss

  params = list(sin_encoder.parameters()) + list(harm_encoder.parameters())
  optimizer = torch.optim.Adam(params, lr=0.0003)

  # Training loop
  num_epochs = 2
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

        mels = mels.cuda()
        audio = audio.cuda()
            
        #Sinusoisal encoder
        sin_freqs, sin_amps = sin_encoder(mels)

        #Sinusoidal synthesiser
        sin_signal = s.sinusoidal_synth(sin_freqs, sin_amps, 16000)
        sin_signal = rearrange(sin_signal, 'a b c -> a c b')

        #Sinusoidal reconstruction loss
        sin_recon_loss = sin_criterion(sin_signal, audio.unsqueeze(0))
        writer.add_scalar('Sin recon Loss/train', sin_recon_loss, i)

        #Harmonic encoder
        sin_freqs = sin_freqs.detach() #detach gradients before they go into the harmonic encoder
        sin_amps = sin_amps.detach()
        glob_amp, harm_dist, f0 = harm_encoder(sin_freqs, sin_amps)

        #Reconstruct audio from harmonic encoder results
        harmonics = h.get_harmonic_frequencies(f0) #need this to then do the sin synth - creates a bank of 100 sinusoids
        harm_dist = h.remove_above_nyquist(harmonics, harm_dist) #only keep the frequencies which are below the nyquist criterion, set amplitudes of other freqs to 0
        harm_dist = h.safe_divide(harm_dist, torch.sum(harm_dist, dim=-1, keepdim=True)) #normalise
        harm_amps = glob_amp * harm_dist

        harm_signal = s.sinusoidal_synth(harmonics, harm_amps, 16000)
        harm_signal = rearrange(harm_signal, 'a b c -> a c b')

        #Harmonic reconstruction loss
        harm_recon_loss = harm_criterion(harm_signal, audio.unsqueeze(0))
        writer.add_scalar('Harm recon Loss/train', harm_recon_loss, i)

        #Consistency loss
        consistency_loss = consistency_criterion(harm_amps, harmonics, sin_amps, sin_freqs)
        writer.add_scalar('Consitency Loss/train', consistency_loss, i)

        #Total loss
        total_loss = sin_recon_loss + harm_recon_loss + (0.1 * consistency_loss)
        writer.add_scalar('Loss/train', total_loss, i)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss.append(total_loss.item())
        sin_recon_running_loss.append(sin_recon_loss.item())
        harm_recon_running_loss.append(harm_recon_loss.item())
        consistency_running_loss.append(consistency_loss.item())

        tepoch.set_description_str(f"{epoch}, loss = {total_loss.item():.4f}", refresh=True)

        i += 1
        
        if i%75 == 0:
          # Save a checkpoint
          torch.save(sin_encoder.state_dict(), f'Finetune/Checkpoints/finetune_sin_encoder_ckpt_{epoch}_{i}.pt')
          torch.save(harm_encoder.state_dict(), f'Finetune/Checkpoints/finetune_harm_encoder_ckpt_{epoch}_{i}.pt')
          sin_loss = torch.tensor(sin_recon_running_loss)
          harm_loss = torch.tensor(harm_recon_running_loss)
          consis_loss = torch.tensor(consistency_running_loss)
          tot_loss = torch.tensor(running_loss)
          torch.save(sin_loss, f'Finetune/Losses/sin_recon_loss_finetune_{epoch}_{i}.pt')
          torch.save(harm_loss, f'Finetune/Losses/harm_recon_loss_finetune_{epoch}_{i}.pt')
          torch.save(consis_loss, f'Finetune/Losses/consistency_loss_finetune_{epoch}_{i}.pt')
          torch.save(tot_loss, f'Finetune/Losses/total_loss_finetune_{epoch}_{i}.pt')          
    
    ##################
    ### Validation ###
    ##################

    with torch.no_grad():

      val_loss = []

      for mels_val, audio_val in DL_DS['val']:

        mels_val = mels_val.cuda()
        audio_val = audio_val.cuda()

        #Sinusoisal encoder
        sin_freqs_val, sin_amps_val = sin_encoder(mels_val)

        #Sinusoidal synthesiser
        sin_signal_val = s.sinusoidal_synth(sin_freqs_val, sin_amps_val, 16000)
        sin_signal_val = rearrange(sin_signal_val, 'a b c -> a c b')

        #Sinusoidal reconstruction loss
        sin_recon_loss_val = sin_criterion(sin_signal_val, audio_val.unsqueeze(0))
        writer.add_scalar('Sin recon Loss/val', sin_recon_loss_val, j)

        #Harmonic encoder
        sin_freqs_val = sin_freqs_val.detach() #detach gradients before they go into the harmonic encoder
        sin_amps_val = sin_amps_val.detach()
        glob_amp_val, harm_dist_val, f0_val = harm_encoder(sin_freqs_val, sin_amps_val)

        #Reconstruct audio from harmonic encoder results
        harmonics_val = h.get_harmonic_frequencies(f0_val) #need this to then do the sin synth - creates a bank of 100 sinusoids
        harm_dist_val = h.remove_above_nyquist(harmonics_val, harm_dist_val) #only keep the frequencies which are below the nyquist criterion, set amplitudes of other freqs to 0
        harm_dist_val = h.safe_divide(harm_dist_val, torch.sum(harm_dist_val, dim=-1, keepdim=True)) #normalise
        harm_amps_val = glob_amp_val * harm_dist_val

        harm_signal_val = s.sinusoidal_synth(harmonics_val, harm_amps_val, 16000)
        harm_signal_val = rearrange(harm_signal_val, 'a b c -> a c b')

        #Harmonic reconstruction loss
        harm_recon_loss_val = harm_criterion(harm_signal_val, audio_val.unsqueeze(0))
        writer.add_scalar('Harm recon Loss/val', harm_recon_loss_val, j)

        #Consistency loss
        consistency_loss_val = consistency_criterion(harm_amps_val, harmonics_val, sin_amps_val, sin_freqs_val)
        writer.add_scalar('Consitency Loss/val', consistency_loss_val, j)

        #Total loss
        total_loss_val = sin_recon_loss_val + harm_recon_loss_val + (0.1 * consistency_loss_val)
        writer.add_scalar('Loss/val', total_loss_val, j)
        val_loss.append(total_loss_val.item())

        if j%75 == 0:
          tot_loss_val = torch.tensor(val_loss)
          torch.save(tot_loss_val, f'Finetune/Losses/total_loss_finetune_val_{epoch}_{j}.pt')          

        j += 1

  writer.flush()
  writer.close()
