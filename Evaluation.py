import mir_eval.melody as me
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import Nets as n
import Helpers as h
from tqdm import tqdm
import multiprocessing

multiprocessing.freeze_support()

experiment = input("Enter an experiment number between 1 and 5: ")
experiment = int(experiment)
checkpoint_sin = ''
checkpoint_harm = ''
if experiment == 1:
  checkpoint_sin = 'Normal\Checkpoints\damp_encoder_ckpt_normal_1_1425.pt'
  checkpoint_harm = 'Normal\Checkpoints\damp_harm_encoder_ckpt_normal_1_1425.pt'
elif experiment == 2:
  checkpoint_sin = 'Scaled\Checkpoints\damp_encoder_ckpt_scaled_1_1425.pt'
  checkpoint_harm = 'Scaled\Checkpoints\damp_harm_encoder_ckpt_scaled_1_1425.pt'
elif experiment == 3:
  checkpoint_sin = '3_Windows\Checkpoints\damp_encoder_ckpt_windows_1_1425.pt'
  checkpoint_harm = '3_Windows\Checkpoints\damp_harm_encoder_ckpt_windows_1_1425.pt'
elif experiment == 4:
  checkpoint_sin = 'Compressed\Checkpoints\damp_encoder_ckpt_compressed_1_1425.pt'
  checkpoint_harm = 'Compressed\Checkpoints\damp_harm_encoder_ckpt_compressed_1_1425.pt'
elif experiment == 5:
  checkpoint_sin = 'Final\Checkpoints\damp_encoder_ckpt_Final_1_1425.pt'
  checkpoint_harm = 'Final\Checkpoints\damp_harm_encoder_ckpt_Final_1_1425.pt'
elif experiment > 5 or experiment < 1 or type(experiment) != int:
  print('Defaulting to experiment 5.')
  checkpoint_sin = 'Final\Checkpoints\damp_encoder_ckpt_Final_1_1425.pt'
  checkpoint_harm = 'Final\Checkpoints\damp_harm_encoder_ckpt_Final_1_1425.pt'



### Load tensors and create dataloader ###
print('Loading test data...')
MELS = torch.load('SavedTensors/meltensor_eval.pt')
Y = torch.load('SavedTensors/y_eval.pt').unsqueeze(1)
ANNOTS = torch.load('SavedTensors/annotations_eval.pt')

eval_dataset = h.EvalDataset(MELS, Y, ANNOTS)
datasets = h.train_val_dataset(eval_dataset)
print('Train', len(datasets['train']))
print('Val', len(datasets['val']))

DL_DS = {x:DataLoader(datasets[x], 1, shuffle=True, num_workers=2) for x in ['train','val']}
print('Done')

### Set up ###
damp_encoder = n.DampingMapping()#.cuda()
#sin_encoder = n.SinMapToFrequency()
damp_harm_encoder = n.DampSinToHarmEncoder()#.cuda()
#harm_encoder = n.SinToHarmEncoder()
print('Loading checkpoints...')
damp_encoder.load_state_dict(torch.load('Final/Checkpoints/damp_encoder_ckpt_Final_1_1425.pt', map_location=torch.device('cpu')))
damp_harm_encoder.load_state_dict(torch.load('Final/Checkpoints/damp_harm_encoder_ckpt_Final_1_1425.pt', map_location=torch.device('cpu')))
#sin_encoder.load_state_dict(torch.load('Finetune/Checkpoints/sin_encoder_ckpt_finetune_1_1425.pt', map_location=torch.device('cpu')))
#harm_encoder.load_state_dict(torch.load('Finetune/Checkpoints/damp_harm_encoder_ckpt_finetune_1_1425.pt', map_location=torch.device('cpu')))
print('Done')

i = 1
rpa_sum = 0
rca_sum = 0
# Evaluation loop

with torch.no_grad():

  with tqdm(DL_DS['train'], unit='batch') as tepoch:

    for mels, audio, annot in tepoch:

      annot = annot.squeeze()
      fs = 16000
          
      #Sinusoisal encoder
      sin_freqs, sin_amps, sin_damps = damp_encoder(mels)
      #sin_freqs, sin_amps = sin_encoder(mels)

      glob_amp, harm_dist, f0 = damp_harm_encoder(sin_freqs, sin_amps, sin_damps)
      #glob_amp, harm_dist, f0 = harm_encoder(sin_freqs, sin_amps)


      f0 = h.UpsampleTime(f0, fs).squeeze()

      ### RPA ### 

      f0 = f0.numpy()
      annot = annot.numpy()
      reftime = np.linspace(0, 1, fs)
      (ref_v, ref_c, est_v, est_c) = me.to_cent_voicing(reftime, annot, reftime, f0)
      rpa = me.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=50)
      rca = me.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c, cent_tolerance=50)
      rpa_sum += rpa
      rca_sum += rca

      tepoch.set_description_str(f"Sample {i}", refresh=True)
      i += 1

rpa_av = rpa_sum/i
rca_av = rca_sum/i
print('Average rpa', rpa_av)
print('Average rca', rca_av)

a = input('Press any key to close.')
