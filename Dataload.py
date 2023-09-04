import Helpers as h
import os
import torch
### Load data ###

audio_dir = "C:/Users/eat_m/Documents/QMUL/Summer_Project/MDB-stem-synth/3_audio/"
annotation_dir = "C:/Users/eat_m/Documents/QMUL/Summer_Project/MDB-stem-synth/3_annotation/"

audio_path = os.listdir(audio_dir)

MELS_norm, Y, annotations = h.LoadAudio(audio_dir, annotation_dir, 16000)


print(Y.shape, annotations[:2000, :].shape)
print(MELS_norm[:2000,:,:125,:].shape)
MELS_norm = MELS_norm[:2000,:,:125, :]
Y = Y[:2000, :]
annotations = annotations[:2000, :]
torch.save(MELS_norm, 'SavedTensors/meltensor_eval.pt')
torch.save(Y, 'SavedTensors/y_eval.pt')
torch.save(annotations, 'SavedTensors/annotations_eval.pt')