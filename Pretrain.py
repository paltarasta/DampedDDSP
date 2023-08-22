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

writer = SummaryWriter('pretrain')
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"Running on device: {device}")

#################################################################################################
########################################## PRETRAINING ##########################################
#################################################################################################

### Load tensors and create dataloader ###
MELS = torch.load('SavedTensors/melsynth.pt')[:,:,:125,:]
Y = torch.load('SavedTensors/ysynth.pt')
sin_amps_target = torch.load('SavedTensors/sin_amps.pt')[:,:125,:]
sin_freqs_target = torch.load('SavedTensors/sin_freqs.pt')[:,:125,:]
harm_amp_target = torch.load('SavedTensors/harm_amp.pt')[:,:125,:]
harm_dist_target = torch.load('SavedTensors/harm_dist.pt')[:,:125,:]
f0_hz_target = torch.load('SavedTensors/f0_hz.pt')[:,:125,:]

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
  consistency_criterion = l.KDEConsistencyLoss
  self_supervision_criterion = l.HarmonicConsistencyLoss

  params = list(sin_encoder.parameters()) + list(harm_encoder.parameters())
  optimizer = torch.optim.Adam(params, lr=0.0003)

  # Training loop
  num_epochs = 5
  i = 0

  for epoch in range(num_epochs):
    print('Epoch', epoch)

    with tqdm(DL_DS['train'], unit='batch') as tepoch:
      running_loss = []

      for mels, audio, glob_amp_t, harm_dist_t, f0_hz_t, sin_amps_t, sin_freqs_t in tepoch:
        print(audio.shape, 'THE TARGETS SHAPE')
        print(mels.shape, 'THE INPUTS SHAPE')

        mels = mels#.cuda()
        audio = audio#.cuda()
        glob_amp_t = glob_amp_t#.cuda()
        harm_dist_t = harm_dist_t#.cuda()
        f0_hz_t = f0_hz_t#.cuda()
        sin_amps_t = sin_amps_t#.cuda()
        sin_freqs_t = sin_freqs_t#.cuda()

        #Sinusoisal encoder
        sin_freqs, sin_amps = sin_encoder(mels)

        #Sinusoidal synthesiser
        sin_signal = s.sinusoidal_synth(sin_freqs, sin_amps, 16000)
        sin_signal = rearrange(sin_signal, 'a b c -> a c b')

        #Sinusoidal reconstruction loss
        sin_recon_loss = sin_criterion(sin_signal, audio.unsqueeze(0))
        writer.add_scalar('Sinusoidal Recon Loss/Pretrain', sin_recon_loss, i)

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
        writer.add_scalar('Harmonic Recon Loss/Pretrain', harm_recon_loss, i)

        #Consistency loss
        consistency_loss = consistency_criterion(harm_amps, harmonics, sin_amps, sin_freqs) #do i need to include the damping here?
        writer.add_scalar('Consistency Loss/Pretrain', consistency_loss, i)

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
        writer.add_scalar('SSLoss/Pretrain', ss_loss, i)

        #Total loss
        total_loss = sin_recon_loss + harm_recon_loss + (0.1 * consistency_loss) + ss_loss

        writer.add_scalar('Loss/Pretrain', total_loss, i)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss.append(total_loss.item())
        tepoch.set_description_str(f"{epoch}, loss = {total_loss.item():.4f}", refresh=True)

        if i%100 == 0:
          # Save a checkpoint
          torch.save(sin_encoder.state_dict(), f'Checkpoints/sin_encoder_ckpt_{epoch}_{i}.pt')
          torch.save(harm_encoder.state_dict(), f'Checkpoints/harm_encoder_ckpt_{epoch}_{i}.pt')

        i += 1


  writer.flush()
  writer.close()
