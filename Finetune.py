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

writer = SummaryWriter()
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"Running on device: {device}")

#################################################################################################
########################################## FINETUNING ##########################################
#################################################################################################

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

if __name__ == "__main__":
  mixed_dataset = h.CustomDataset(MELS, Y)
  datasets = h.train_val_dataset(mixed_dataset)
  print('Train', len(datasets['train']))
  print('Val', len(datasets['val']))

  DL_DS = {x:DataLoader(datasets[x], 1, shuffle=True, num_workers=2) for x in ['train','val']} #num_worker should = 4 * num_GPU

  ### Set up ###
  sin_encoder = n.SinMapToFrequency()#.cuda()
  harm_encoder = n.SinToHarmEncoder()#.cuda()

  ### Load pretrained weights ###
  sin_encoder.load_state_dict(torch.load("Checkpoints\sin_encoder_ckpt_1.pt"))
  harm_encoder.load_state_dict(torch.load("Checkpoints\harm_encoder_ckpt_1.pt"))

  sin_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512])#.cuda()
  harm_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512])#.cuda()
  consistency_criterion = l.KDEConsistencyLoss#.cuda()

  params = list(sin_encoder.parameters()) + list(harm_encoder.parameters())
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

        #Total loss
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

      # Save a checkpoint
      torch.save(sin_encoder.state_dict(), f'Checkpoints/finetune_sin_encoder_ckpt_{epoch}.pt')
      torch.save(harm_encoder.state_dict(), f'Checkpoints/finetune_harm_encoder_ckpt_{epoch}.pt')

  writer.flush()
  writer.close()
