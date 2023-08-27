import mir_eval.melody as me
import torch
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import Nets as n
import Helpers as h
from tqdm import tqdm


if __name__ == "__main__":

  ### Load tensors and create dataloader ###
  MELS = torch.load('SavedTensors/meltensor_125.pt')[:10]
  Y = torch.load('SavedTensors/y_125.pt').unsqueeze(1)[:10]
  ANNOTS = torch.load('SavedTensors/annotations_125.pt')[:10]

  eval_dataset = h.EvalDataset(MELS, Y, ANNOTS)
  datasets = h.train_val_dataset(eval_dataset)
  print('Train', len(datasets['train']))
  print('Val', len(datasets['val']))

  DL_DS = {x:DataLoader(datasets[x], 1, shuffle=True, num_workers=2) for x in ['train','val']} #num_worker should = 4 * num_GPU
  
  ### Set up ###
  damp_encoder = n.DampingMapping()#.cuda()
  damp_harm_encoder = n.DampSinToHarmEncoder()#.cuda()
  #damp_encoder.load_state_dict(torch.load('Normal\Checkpoints\damp_encoder_ckpt_normal_0_225.pt', map_location=torch.device('cpu')))
  #damp_harm_encoder.load_state_dict(torch.load('Normal\Checkpoints\damp_harm_encoder_ckpt_normal_0_225.pt', map_location=torch.device('cpu')))

  i = 1
  rpa_sum = 0
  rca_sum = 0
  # Evaluation loop

  with torch.no_grad():

    with tqdm(DL_DS['train'], unit='batch') as tepoch:

      for mels, audio, annot in tepoch:

        mels = mels#.cuda()
        audio = audio#.cuda()
        annot = annot.squeeze()#.cuda()
        fs = 16000
            
        #Sinusoisal encoder
        sin_freqs, sin_amps, sin_damps = damp_encoder(mels)

        glob_amp, harm_dist, f0 = damp_harm_encoder(sin_freqs, sin_amps, sin_damps)

        f0 = h.UpsampleTime(f0, fs).squeeze()

        ### RPA ### 

        f0 = f0.numpy()
        annot = annot.numpy()
        reftime = np.linspace(0, 1, fs)
        (ref_v, ref_c, est_v, est_c) = me.to_cent_voicing(reftime, annot, reftime, f0)
        rpa = me.raw_pitch_accuracy(ref_v, ref_c, est_v, est_c)
        rca = me.raw_chroma_accuracy(ref_v, ref_c, est_v, est_c)
        rpa_sum += rpa
        rca_sum += rca
        print('rpa sum', rpa_sum)

        tepoch.set_description_str(f"Sample {i}", refresh=True)
        i += 1

  rpa_av = rpa_sum/i
  rca_av = rca_sum/i
  print('Average rpa', rpa_av)
  print('Average rca', rca_sum)