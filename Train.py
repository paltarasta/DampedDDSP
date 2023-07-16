import torch
from torch.utils.data import DataLoader
import Helpers as h
import Nets as n
import auraloss as al
from tqdm import tqdm
from Synths import damped_synth
from einops import rearrange
import matplotlib.pyplot as plt

### Load tensors and create dataloader ###
MELS_norm = torch.load('meltensor.pt')
Y = torch.load('y.pt')

if __name__ == "__main__":
  myDataset = h.CustomDataset(MELS_norm[:4], Y[:4])
  datasets = h.train_val_dataset(myDataset)
  print(len(datasets['train']))
  print(len(datasets['val']))

  DL_DS = {x:DataLoader(datasets[x], 1, shuffle=True, num_workers=2) for x in ['train','val']}


  ### Training Loop ###
  model = n.MapToFrequency()
  criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512])
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # Training loop
  num_epochs = 2

  average_loss = []

  for epoch in range(num_epochs):
    with tqdm(DL_DS['train'], unit='batch') as tepoch:
      print('Epoch', epoch)
      running_loss = []
      for inputs, targets in tepoch:

        f, a, d = model(inputs)

        print('f', f.shape,'a', a.shape, 'd', d.shape)
        f_upsampled = h.UpsampleTime(f)
        a_upsampled = h.UpsampleTime(a)
        d_upsampled = h.UpsampleTime(d)

        prediction = damped_synth(f_upsampled, a_upsampled, d_upsampled, 16000)
        prediction = rearrange(prediction, 'a b c -> a c b')
        print('prediction', prediction.shape)
        print('target', targets.unsqueeze(0).shape)

        loss = criterion(prediction, targets.unsqueeze(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        tepoch.set_description_str(f"{epoch}, loss = {loss.item():.4f}")

      average_loss.append(sum(running_loss)/len(running_loss))

      print('AVERAGE LOSS', average_loss[epoch-1])

  print('LOSS', running_loss)
  plt.plot(running_loss)
  plt.show()