import tensorflow as tf
import os
import torch
from einops import rearrange
import Synths as s
import Helpers as h
import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd

filepath = "C:/Users/eat_m/Documents/QMUL/Summer_Project/Synthetic_data/synth_dataset-00000-of-00001"

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

    #Convert to numpy

    harm_amp_numpy = harm_amp_tensor.numpy()
    harm_dist_numpy = harm_dist_tensor.numpy()
    f0_hz_numpy = f0_hz_tensor.numpy()

    # Convert to torch tensor

    harm_amp_torch = torch.tensor(harm_amp_numpy)
    harm_dist_torch = torch.tensor(harm_dist_numpy)
    f0_hz_torch = torch.tensor(f0_hz_numpy)

    # Reshape for further processing (batch x time x feature = 1 x 126 x 1 or 100)

    harm_amp_torch = rearrange(harm_amp_torch, '(a b) -> a b', b=1)
    harm_dist_torch = rearrange(harm_dist_torch, '(a b) -> a b', a=126)
    f0_hz_torch = rearrange(f0_hz_torch, '(a b) -> a b', b=1).unsqueeze(0)
    amplitudes = (harm_amp_torch * harm_dist_torch).unsqueeze(0)

    # Upsample time

    amplitudes = h.UpsampleTime(amplitudes)
    f0_hz_torch = h.UpsampleTime(f0_hz_torch)

    # Synthesise sound - not sure about this bit

    signal = s.harmonic_synth(f0_hz_torch, amplitudes, 16000).squeeze(0).numpy()
    alpha.append(signal)

    i += 1

print(len(alpha[0]), len(alpha), type(alpha))

MELS_synth = h.MakeMelsTensor(alpha, 16000, MELS_synth, True)
MELS_synth = torch.stack(MELS_synth).unsqueeze(1)
print(type(MELS_synth), MELS_synth.shape, 'all about mels synth')

#before normalsing, make sure it is in batch, 1, 126, 229 format

MELS_synth_norm = h.NormaliseMels(MELS_synth)

alpha = torch.tensor(np.array(alpha))
alpha = rearrange(alpha, 'a b c -> a c b')
print(alpha.shape)

torch.save(MELS_synth_norm, 'melsynth.pt')
torch.save(alpha, 'ysynth.pt')
print('all done!')


#sd.play(alpha[1], samplerate=16000)
#sd.wait()


#plt.figure()
#plt.plot(alpha[0])
#plt.show()
'''
for raw_record in raw_dataset.take(1):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    alpha.append(example)
#print('alpha', type(alpha[0]))
print('num examples as i see it', i)
#print(alpha[0] == alpha[1])

serialised_example = alpha[0].SerializeToString()
#print(serialised_example)


parsed_example = tf.io.parse_single_example(serialised_example, feature_description)

# Convert the features to TensorFlow tensors or NumPy arrays
harm_amp_tensor = parsed_example['harm_amp']
harm_dist_tensor = parsed_example['harm_dist']
f0_hz_tensor = parsed_example['f0_hz']
sin_amps_tensor = parsed_example['sin_amps']
sin_freqs_tensor = parsed_example['sin_freqs']
noise_magnitudes_tensor = parsed_example['noise_magnitudes']


print('########################################################################################')
#print(harm_amp_tensor)
# Convert to numpy array

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

harm_amp_torch = rearrange(harm_amp_torch, '(a b) -> a b', b=1)
print(harm_amp_torch.shape, type(harm_amp_torch))
harm_dist_torch = rearrange(harm_dist_torch, '(a b) -> a b', a=126)
print(harm_dist_torch.shape)
f0_hz_torch = rearrange(f0_hz_torch, '(a b) -> a b', b=1).unsqueeze(0)
print(f0_hz_torch.shape)

amplitudes = (harm_amp_torch * harm_dist_torch).unsqueeze(0)
print('amplitudes', amplitudes.shape)

amplitudes = h.UpsampleTime(amplitudes)
f0_hz_torch = h.UpsampleTime(f0_hz_torch)

signal = s.harmonic_synth(f0_hz_torch, amplitudes, 16000).squeeze(0).numpy()
print('signal', type(signal), signal.shape)

### now check you can run this for multiple files
### use the harmonic synth to generate a sound
### check what it sounds like
### create dataset
sd.play(signal, samplerate=16000)
sd.wait()
'''