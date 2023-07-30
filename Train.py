import torch
from torch.utils.data import DataLoader
import Helpers as h
import Nets as n
import auraloss as al
from tqdm import tqdm
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
########################################## PRETRAINING ##########################################
#################################################################################################

### Load tensors and create dataloader ###
MELS = torch.load('SavedTensors/melsynth.pt')
Y = torch.load('SavedTensors/ysynth.pt')
sin_amps_target = torch.load('SavedTensors/sin_amps.pt')
sin_freqs_target = torch.load('SavedTensors/sin_freqs.pt')
harm_amp_target = torch.load('SavedTensors/harm_amp.pt')
harm_dist_target = torch.load('SavedTensors/harm_dist.pt')
f0_hz_target = torch.load('SavedTensors/f0_hz.pt')

if __name__ == "__main__":
  synth_dataset = h.SyntheticDataset(MELS, Y, harm_amp_target, harm_dist_target, f0_hz_target, sin_amps_target, sin_freqs_target)
  datasets = h.train_val_dataset(synth_dataset)
  print('Train', len(datasets['train']))
  print('Val', len(datasets['val']))

  DL_DS = {x:DataLoader(datasets[x], 1, shuffle=True, num_workers=2) for x in ['train','val']} #num_worker should = 4 * num_GPU

  ### Set up ###
  sin_encoder = n.SinMapToFrequency()#.cuda()
  harm_encoder = n.SinToHarmEncoder()#.cuda()

  sin_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512])#.cuda()
  harm_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512])#.cuda()
  consistency_criterion = l.KDEConsistencyLoss#.cuda()
  self_supervision_criterion = l.HarmonicConsistencyLoss#.cuda()

  params = list(sin_encoder.parameters()) + list(harm_encoder.parameters())
  optimizer = torch.optim.Adam(params, lr=0.0003)

  # Training loop
  num_epochs = 2
  i = 0

  for epoch in range(num_epochs):
    print('Epoch', epoch)

    with tqdm(DL_DS['train'], unit='batch') as tepoch:
      running_loss = []

      for mels, audio, glob_amp_t, harm_dist_t, f0_hz_t, sin_amps_t, sin_freqs_t in tepoch:
        print(audio.shape, 'THE TARGETS SHAPE')
        print(mels.shape, 'THE INPUTS SHAPE')

        #mels = mels.cuda()
        #audio = audio.cuda()
        #harm_amp_t = harm_amp_t.cuda()
        #harm_dist_t = harm_dist_t.cuda()
        #f0_hz_t = f0_hz_t.cuda()
        #sin_amps_t = sin_amps_t.cuda()
        #sin_freqs_t = sin_freqs_t.cuda()

        if i > 4:
          break
            
        #Sinusoisal encoder
        sin_freqs, sin_amps = sin_encoder(mels)
        print('post sin encoder')

        #Sinusoidal synthesiser
        sin_signal = s.sinusoidal_synth(sin_freqs, sin_amps, 16000)
        sin_signal = rearrange(sin_signal, 'a b c -> a c b')
        print('post sin synth')

        #Sinusoidal reconstruction loss
        sin_recon_loss = sin_criterion(sin_signal, audio.unsqueeze(0))

        #Harmonic encoder
        sin_freqs = sin_freqs.detach() #detach gradients before they go into the harmonic encoder
        sin_amps = sin_amps.detach()
        glob_amp, harm_dist, f0 = harm_encoder(sin_freqs, sin_amps)
        print('post harm encoder')

        #Reconstruct audio from harmonic encoder results
        harmonics = h.get_harmonic_frequencies(f0) #need this to then do the sin synth - creates a bank of 100 sinusoids
        harm_dist = h.remove_above_nyquist(harmonics, harm_dist) #only keep the frequencies which are below the nyquist criterion, set amplitudes of other freqs to 0
        harm_dist = h.safe_divide(harm_dist, torch.sum(harm_dist, dim=-1, keepdim=True)) #normalise
        harm_amps = glob_amp * harm_dist

        harm_signal = s.sinusoidal_synth(harmonics, harm_amps, 16000)
        harm_signal = rearrange(harm_signal, 'a b c -> a c b')

        #Harmonic reconstruction loss
        harm_recon_loss = harm_criterion(harm_signal, audio.unsqueeze(0))
        print('post harm loss')

        #Consistency loss
        consistency_loss = consistency_criterion(harm_amps, harmonics, sin_amps, sin_freqs)
        print('sin loss', sin_recon_loss)
        print('harm loss', harm_recon_loss)
        print('consistency loss', consistency_loss)

        #Self-supervision loss
        ss_loss = self_supervision_criterion(glob_amp,
                                             glob_amp_t,
                                             harm_dist,
                                             harm_dist_t,
                                             f0,
                                             f0_hz_t,
                                             sin_amps,
                                             sin_amps_t,
                                             sin_freqs,
                                             sin_freqs_t
                                             )
        print('self supervision', ss_loss)

        #Total loss
        total_loss = sin_recon_loss + harm_recon_loss + consistency_loss + ss_loss
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

      # Save a checkpoint
      torch.save(sin_encoder.state_dict(), f'sin_encoder_ckpt_{epoch}.pt')
      torch.save(harm_encoder.state_dict(), f'harm_encoder_ckpt_{epoch}.pt')

  writer.flush()
  writer.close()





'''
### Load tensors and create dataloader ###
MELS_norm = torch.load('SavedTensors/melsynth.pt')
Y = torch.load('SavedTensors/ysynth.pt')

if __name__ == "__main__":
  myDataset = h.CustomDataset(MELS_norm, Y)
  datasets = h.train_val_dataset(myDataset)
  print('Train', len(datasets['train']))
  print('Val', len(datasets['val']))

  DL_DS = {x:DataLoader(datasets[x], 1, shuffle=True, num_workers=2) for x in ['train','val']} #num_worker should = 4 * num_GPU


  ### Training Loop ###
  sin_encoder = n.SinMapToFrequency()#.cuda()
  harm_encoder = n.SinToHarmEncoder()#.cuda()

  sin_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512])#.cuda()
  harm_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512])#.cuda()
  consistency_criterion = l.KDEConsistencyLoss#.cuda()

  params = list(sin_encoder.parameters()) + list(harm_encoder.parameters())
  optimizer = torch.optim.Adam(params, lr=0.0003)
  
  # Training loop
  num_epochs = 2

  average_loss = []
  i = 0

  for epoch in range(num_epochs):
    print('Epoch', epoch)

    with tqdm(DL_DS['train'], unit='batch') as tepoch:
      running_loss = []

      for inputs, targets in tepoch:
        print(targets.shape, 'THE TARGETS SHAPE')
        print(inputs.shape, 'THE INPUTS SHAPE')

        #inputs = inputs.cuda()
        #targets = targets.cuda()

        if i > 2:
          break
            
        #Sinusoisal encoder

        sins, amps = sin_encoder(inputs)

        print('post sin encoder')

        #Sinusoidal synthesiser
        #damped_signal = damped_synth(sins, amps, damps, 16000)
        sin_signal = s.sinusoidal_synth(sins, amps, 16000)
        sin_signal = rearrange(sin_signal, 'a b c -> a c b')
        print('post sin synth')

        #First reconstruction loss
        sin_recon_loss = sin_criterion(sin_signal, targets.unsqueeze(0))

        #Harmonic encoder
        sins = sins.detach() #detach gradients before they go into the harmonic encoder
        amps = amps.detach()

        harmonics, harm_amps, a, b, c = harm_encoder(sins, amps)
        print('post harm encoder')

        #harm_signal = damped_synth(harmonics, harm_amps, harm_damps, 16000)
        harm_signal = s.sinusoidal_synth(harmonics, harm_amps, 16000)
        harm_signal = rearrange(harm_signal, 'a b c -> a c b')
        
        #Second reconstruction loss
        harm_recon_loss = harm_criterion(harm_signal, targets.unsqueeze(0))
        print('post harm loss')

        consistency_loss = consistency_criterion(harm_amps, harmonics, amps, sins)
        print('sin loss', sin_recon_loss)
        print('harm loss', harm_recon_loss)
        print('consistency loss', consistency_loss)
        
        total_loss = sin_recon_loss + harm_recon_loss + consistency_loss
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

      #average_loss.append(sum(running_loss)/len(running_loss))

      #print('AVERAGE LOSS', average_loss[epoch-1])
      # Save a checkpoint
      torch.save(sin_encoder.state_dict(), f'sin_encoder_ckpt_{epoch}.pt')

  #print('LOSS', running_loss)
  #plt.plot(running_loss)
  #plt.show()
'''