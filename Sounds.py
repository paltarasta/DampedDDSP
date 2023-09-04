import torch
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np

synthetics = torch.load('SavedTensors/ysynth_125.pt')[:10]
synthetics = synthetics.view(1, -1)

synthetics = np.array(synthetics[0])
print(synthetics.shape)

plt.figure()
plt.plot(synthetics)
plt.show()
fs = 16000
file_name = "output.wav"
wavfile.write(file_name, fs, synthetics)

