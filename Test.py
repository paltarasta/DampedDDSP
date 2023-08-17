import torch
import Helpers as h
import os
import matplotlib.pyplot as plt

harmonics = torch.load('Outputs/harmonics_exp1_0_2000.pt', map_location=torch.device('cpu'))
print('harmonics', harmonics.shape)
print(harmonics.to(torch.int32))

sin_freqs = torch.load('Outputs/sin_freqs_exp1_0_2000.pt', map_location=torch.device('cpu'))
print('sinfreqs')
print(sin_freqs.to(torch.int32))

harmonics = torch.load('Outputs/harmonics_exp1_0_100.pt', map_location=torch.device('cpu'))
print(harmonics.shape)
print(harmonics)

sin_freqs = torch.load('Outputs/sin_freqs_exp1_0_100.pt', map_location=torch.device('cpu'))
print(sin_freqs)

'''

harm_amps = harmonics = torch.load('Outputs/harm_amps_exp1_0_2000.pt', map_location=torch.device('cpu'))
print('harm amps', harm_amps)

sin_amps = torch.load('Outputs/sin_amps_exp1_0_2000.pt', map_location=torch.device('cpu'))
print('sin amps', sin_amps)
'''