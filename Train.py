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

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print(f"Running on device: {device}")

### Load tensors and create dataloader ###
MELS_norm = torch.load('meltensor.pt')
Y = torch.load('y.pt')

if __name__ == "__main__":
  myDataset = h.CustomDataset(MELS_norm, Y)
  datasets = h.train_val_dataset(myDataset)
  print(len(datasets['train']))
  print(len(datasets['val']))

  DL_DS = {x:DataLoader(datasets[x], 1, shuffle=True, num_workers=2) for x in ['train','val']}


  ### Training Loop ###
  sin_encoder = n.SinMapToFrequency()
  harm_encoder = n.SinToHarmEncoder()

  sin_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512])#.cuda()
  harm_criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512])#.cuda()

  params = list(sin_encoder.parameters()) + list(harm_encoder.parameters())
  optimizer = torch.optim.Adam(params, lr=0.03)
  
  # Training loop
  num_epochs = 2

  average_loss = []
  i = 0

  for epoch in range(num_epochs):
    with tqdm(DL_DS['train'], unit='batch') as tepoch:
      print('Epoch', epoch)
      running_loss = []
      for inputs, targets in tepoch:
        print(targets.shape, 'THE TARGETS SHAPE')
        print(inputs.shape, 'THE INPUTS SHAPE')
        if i > 50:
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

        harmonics, harm_amps, harm_damps = harm_encoder(sins, amps)
        print('post harm encoder')

        #harm_signal = damped_synth(harmonics, harm_amps, harm_damps, 16000)
        harm_signal = s.sinusoidal_synth(harmonics, harm_amps, 16000)
        harm_signal = rearrange(harm_signal, 'a b c -> a c b')
        
        #Second reconstruction loss
        harm_recon_loss = harm_criterion(harm_signal, targets.unsqueeze(0))
        print('post harm loss')

        consistency_loss = l.KDEConsistencyLoss(harm_amps, harmonics, amps, sins)
        print('sin loss', sin_recon_loss)
        print('harm loss', harm_recon_loss)
        print('consistency loss', consistency_loss)
        
        total_loss = sin_recon_loss + harm_recon_loss + consistency_loss
        print('after losses all summed', total_loss)
        
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
  torch.save(sin_encoder.state_dict(), 'sin_weights.pt')
  #print('LOSS', running_loss)
  #plt.plot(running_loss)
  #plt.show()