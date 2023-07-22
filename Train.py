import torch
from torch.utils.data import DataLoader
import Helpers as h
import Nets as n
import auraloss as al
from tqdm import tqdm
from Synths import damped_synth, sinusoidal_synth
from einops import rearrange
import matplotlib.pyplot as plt
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Running on device: {device}")

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
  #sin_encoder = sin_encoder.cuda()
  criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512])#.cuda()
  optimizer = torch.optim.Adam(sin_encoder.parameters(), lr=0.001)

  harm_encoder = n.SinToHarmEncoder()
  #harm_encoder = harm_encoder.cuda()
  # Training loop
  num_epochs = 5

  average_loss = []
  i = 0

  for epoch in range(num_epochs):
    with tqdm(DL_DS['train'], unit='batch') as tepoch:
      print('Epoch', epoch)
      running_loss = []
      for inputs, targets in tepoch:
        if i > 1:
          break
        
        #Sinusoisal encoder
        #sins, amps, damps = sin_encoder(inputs)
        sins, amps = sin_encoder(inputs)

        #Sinusoidal synthesiser
        #damped_signal = damped_synth(sins, amps, damps, 16000)
        sin_signal = sinusoidal_synth(sins, amps, 16000)
        sin_signal = rearrange(sin_signal, 'a b c -> a c b')

        #First reconstruction loss
        sin_recon_loss = criterion(sin_signal.to(device), targets.unsqueeze(0).to(device))
        print('sin recon loss', sin_recon_loss)
        
        #Harmonic encoder
        sins = sins.detach() #detach gradients before they go into the harmonic encoder
        amps = amps.detach()
        harmonics, harm_amps, harm_damps = harm_encoder(sins, amps)

        #harm_signal = damped_synth(harmonics, harm_amps, harm_damps, 16000)
        harm_signal = sinusoidal_synth(harmonics, harm_amps, 16000)
        
        optimizer.zero_grad()
        sin_recon_loss.backward()
        optimizer.step()

        running_loss.append(sin_recon_loss.item())
        tepoch.set_description_str(f"{epoch}, loss = {sin_recon_loss.item():.4f}", refresh=True)

        i += 1

      #average_loss.append(sum(running_loss)/len(running_loss))

      #print('AVERAGE LOSS', average_loss[epoch-1])
  torch.save(sin_encoder.state_dict(), 'sin_weights.pt')
  #print('LOSS', running_loss)
  #plt.plot(running_loss)
  #plt.show()