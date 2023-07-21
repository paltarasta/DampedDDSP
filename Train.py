import torch
from torch.utils.data import DataLoader
import Helpers as h
import Nets as n
import auraloss as al
from tqdm import tqdm
from Synths import damped_synth
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
  model = n.MapToFrequency()
  model = model.cuda()
  criterion = al.freq.MultiResolutionSTFTLoss(fft_sizes=[1024, 2048, 512]).cuda()
  optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

  # Training loop
  num_epochs = 1000

  average_loss = []

  for epoch in range(num_epochs):
    with tqdm(DL_DS['train'], unit='batch') as tepoch:
      print('Epoch', epoch)
      running_loss = []
      for inputs, targets in tepoch:

        f, a, d = model(inputs)

        f_upsampled = h.UpsampleTime(f)
        a_upsampled = h.UpsampleTime(a)
        d_upsampled = h.UpsampleTime(d)

        prediction = damped_synth(f_upsampled, a_upsampled, d_upsampled, 16000)
        prediction = rearrange(prediction, 'a b c -> a c b')

        loss = criterion(prediction.to(device), targets.unsqueeze(0).to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        tepoch.set_description_str(f"{epoch}, loss = {loss.item():.4f}", refresh=True)

      average_loss.append(sum(running_loss)/len(running_loss))

      print('AVERAGE LOSS', average_loss[epoch-1])
  torch.save(model.state_dict(), 'damped_weights.pt')
  #print('LOSS', running_loss)
  plt.plot(running_loss)
  plt.show()