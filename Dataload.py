import Helpers as h
import os
import torch
### Load data ###

audio_dir = "C:/Users/eat_m/Documents/QMUL/Summer_Project/MDB-stem-synth/audio_stems/"
annotation_dir = "C:/Users/eat_m/Documents/QMUL/Summer_Project/MDB-stem-synth/annotation_stems/"

audio_path = os.listdir(audio_dir)

MELS_norm, Y, annotations = h.LoadAudio(audio_dir, annotation_dir, 16000)


print(Y.shape)
print(MELS_norm.shape)
MELS_norm = MELS_norm[:,:,:125, :]
torch.save(MELS_norm, 'meltensor_125.pt')
torch.save(Y, 'y_125.pt')
torch.save(annotations, 'annotations_125.pt')