import torch

MELS_synth = torch.load('SavedTensors/melsynth_125.pt')
MELS_real = torch.load('SavedTensors/meltensor_125.pt')
Y_synth = torch.load('SavedTensors/ysynth_125.pt')
Y_real = torch.load('SavedTensors/y_125.pt').unsqueeze(1)
print(MELS_synth.shape)
print(MELS_real.shape)
print(Y_synth.shape)
print(Y_real.shape)