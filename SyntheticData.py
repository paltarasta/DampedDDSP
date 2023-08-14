import tensorflow as tf
import torch
from einops import rearrange
import Synths as s
import Helpers as h
import numpy as np

filepath = "C:/Users/eat_m/Documents/QMUL/Summer_Project/Synthetic_data/synth_dataset_125-00000-of-00001"

#def parse_tfrecord(filepath):
#    return [tf.train.Example.FromString(record.numpy()) for record in tf.data.TFRecordDataset(filepath)]

#beta = parse_tfrecord(filepath)

feature_description = {'harm_amp': tf.io.FixedLenSequenceFeature([], dtype=float, allow_missing=True),
              'harm_dist': tf.io.FixedLenSequenceFeature([], dtype=float, allow_missing=True),
              'f0_hz': tf.io.FixedLenSequenceFeature([], dtype=float, allow_missing=True),
              'sin_amps': tf.io.FixedLenSequenceFeature([], dtype=float, allow_missing=True),
              'sin_freqs': tf.io.FixedLenSequenceFeature([], dtype=float, allow_missing=True),
              'noise_magnitudes': tf.io.FixedLenSequenceFeature([], dtype=float, allow_missing=True)}

raw_dataset = tf.data.TFRecordDataset(filepath)

alpha = []
MELS_synth = []
sin_amps = []
sin_freqs = []
harm_amp = []
harm_dist = []
f0_hz = []
i = 0

for raw_record in raw_dataset:
    print(i, end='\r')

    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    serialised_example = example.SerializeToString()
    parsed_example = tf.io.parse_single_example(serialised_example, feature_description)

    harm_amp_tensor = parsed_example['harm_amp']
    harm_dist_tensor = parsed_example['harm_dist']
    f0_hz_tensor = parsed_example['f0_hz']
    sin_amps_tensor = parsed_example['sin_amps']
    sin_freqs_tensor = parsed_example['sin_freqs']
    noise_magnitudes_tensor = parsed_example['noise_magnitudes']

    #Convert to numpy

    harm_amp_numpy = harm_amp_tensor.numpy()
    harm_dist_numpy = harm_dist_tensor.numpy()
    f0_hz_numpy = f0_hz_tensor.numpy()
    sin_amps_numpy = sin_amps_tensor.numpy()
    sin_freqs_numpy = sin_freqs_tensor.numpy()
    noise_magnitudes_numpy = noise_magnitudes_tensor.numpy()
    
    # Convert to torch tensor

    harm_amp_torch = torch.tensor(harm_amp_numpy)
    harm_dist_torch = torch.tensor(harm_dist_numpy)
    f0_hz_torch = torch.tensor(f0_hz_numpy)
    sin_amps_torch = torch.tensor(sin_amps_numpy)
    sin_freqs_torch = torch.tensor(sin_freqs_numpy)
    noise_magnitudes_torch = torch.tensor(noise_magnitudes_numpy)

    # Reshape for further processing (batch x time x feature = 1 x 126 x 1 or 100)
    harm_amp_torch = rearrange(harm_amp_torch, '(a b) -> a b', b=1)
    harm_dist_torch = rearrange(harm_dist_torch, '(a b) -> a b', a=125)
    f0_hz_torch = rearrange(f0_hz_torch, '(a b) -> a b', b=1).unsqueeze(0)
    amplitudes = (harm_amp_torch * harm_dist_torch).unsqueeze(0)
    sin_amps_torch = rearrange(sin_amps_torch, '(a b) -> a b', a=125)
    sin_freqs_torch = rearrange(sin_freqs_torch, '(a b) -> a b', a=125)

    sin_amps.append(sin_amps_torch)
    sin_freqs.append(sin_freqs_torch)
    harm_amp.append(harm_amp_torch)
    harm_dist.append(harm_dist_torch)
    f0_hz.append(f0_hz_torch.squeeze(0))

    # Upsample time

    amplitudes = h.UpsampleTime(amplitudes)
    f0_hz_torch = h.UpsampleTime(f0_hz_torch)

    # Synthesise sound - not sure about this bit

    signal = s.harmonic_synth(f0_hz_torch, amplitudes, 16000).squeeze(0).numpy()
    alpha.append(signal)

    i += 1

MELS_synth = h.MakeMelsTensor(alpha, 16000, MELS_synth, True)
MELS_synth = torch.stack(MELS_synth).unsqueeze(1)

#before normalsing, make sure it is in batch, 1, 126, 229 format
#for the audio and mels, and for the controls it should be in batch, 126, 1 slash 100

MELS_synth_norm = h.NormaliseMels(MELS_synth)
MELS_synth_norm = MELS_synth_norm[:,:,:125, :]

alpha = torch.tensor(np.array(alpha))
alpha = rearrange(alpha, 'a b c -> a c b')

#sin_amps = torch.stack(sin_amps)
#sin_freqs = torch.stack(sin_freqs)
#harm_amp = torch.stack(harm_amp)
#harm_dist = torch.stack(harm_dist)
#f0_hz = torch.stack(f0_hz)

print('after stacking', MELS_synth_norm.shape, alpha.shape)#, sin_amps.shape, sin_freqs.shape, harm_amp.shape, harm_dist.shape, f0_hz.shape)

torch.save(MELS_synth_norm, 'SavedTensors/melsynth_125.pt')
torch.save(alpha, 'SavedTensors/ysynth_125.pt')
#torch.save(sin_amps, 'SavedTensors/sin_amps.pt')
#torch.save(sin_freqs, 'SavedTensors/sin_freqs.pt')
#torch.save(harm_amp, 'SavedTensors/harm_amp.pt')
#torch.save(harm_dist, 'SavedTensors/harm_dist.pt')
#torch.save(f0_hz, 'SavedTensors/f0_hz.pt')

print('all done!')