# Copyright 2023 The DDSP Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Functions to generate self-supervised signal, EXPERIMENTAL."""

''' Modified for use in DampedDDSP-inv '''
import warnings

#import ddsp
import gin
import numpy as np
import tensorflow as tf


warnings.warn('Imported synthetic_data.py module, which is EXPERIMENTAL '
              'and likely to change.')



def upsample_with_windows(inputs: tf.Tensor,
                          n_timesteps: int,
                          add_endpoint: bool = True) -> tf.Tensor:
  """Upsample a series of frames using using overlapping hann windows.

  Good for amplitude envelopes.
  Args:
    inputs: Framewise 3-D tensor. Shape [batch_size, n_frames, n_channels].
    n_timesteps: The time resolution of the output signal.
    add_endpoint: Hold the last timestep for an additional step as the endpoint.
      Then, n_timesteps is divided evenly into n_frames segments. If false, use
      the last timestep as the endpoint, producing (n_frames - 1) segments with
      each having a length of n_timesteps / (n_frames - 1).

  Returns:
    Upsampled 3-D tensor. Shape [batch_size, n_timesteps, n_channels].

  Raises:
    ValueError: If input does not have 3 dimensions.
    ValueError: If attempting to use function for downsampling.
    ValueError: If n_timesteps is not divisible by n_frames (if add_endpoint is
      true) or n_frames - 1 (if add_endpoint is false).
  """
  inputs = tf.cast(inputs, tf.float32)
  print('upsample with windows')

  if len(inputs.shape) != 3:
    raise ValueError('Upsample_with_windows() only supports 3 dimensions, '
                     'not {}.'.format(inputs.shape))

  # Mimic behavior of tf.image.resize.
  # For forward (not endpointed), hold value for last interval.
  if add_endpoint:
    inputs = tf.concat([inputs, inputs[:, -1:, :]], axis=1)

  n_frames = int(inputs.shape[1])
  n_intervals = (n_frames - 1)

  if n_frames >= n_timesteps:
    raise ValueError('Upsample with windows cannot be used for downsampling'
                     'More input frames ({}) than output timesteps ({})'.format(
                         n_frames, n_timesteps))

  if n_timesteps % n_intervals != 0.0:
    minus_one = '' if add_endpoint else ' - 1'
    raise ValueError(
        'For upsampling, the target the number of timesteps must be divisible '
        'by the number of input frames{}. (timesteps:{}, frames:{}, '
        'add_endpoint={}).'.format(minus_one, n_timesteps, n_frames,
                                   add_endpoint))

  # Constant overlap-add, half overlapping windows.
  hop_size = n_timesteps // n_intervals
  window_length = 2 * hop_size
  window = tf.signal.hann_window(window_length)  # [window]

  # Transpose for overlap_and_add.
  x = tf.transpose(inputs, perm=[0, 2, 1])  # [batch_size, n_channels, n_frames]

  # Broadcast multiply.
  # Add dimension for windows [batch_size, n_channels, n_frames, window].
  x = x[:, :, :, tf.newaxis]
  window = window[tf.newaxis, tf.newaxis, tf.newaxis, :]
  x_windowed = (x * window)
  x = tf.signal.overlap_and_add(x_windowed, hop_size)

  # Transpose back.
  x = tf.transpose(x, perm=[0, 2, 1])  # [batch_size, n_timesteps, n_channels]

  # Trim the rise and fall of the first and last window.
  return x[:, hop_size:-hop_size, :]



def get_harmonic_frequencies(frequencies: tf.Tensor,
                             n_harmonics: int) -> tf.Tensor:
  """Create integer multiples of the fundamental frequency.

  Args:
    frequencies: Fundamental frequencies (Hz). Shape [batch_size, :, 1].
    n_harmonics: Number of harmonics.

  Returns:
    harmonic_frequencies: Oscillator frequencies (Hz).
      Shape [batch_size, :, n_harmonics].
  """
  print('getting harmonic freqs')
  frequencies = tf.cast(frequencies, tf.float32)

  f_ratios = tf.linspace(1.0, float(n_harmonics), int(n_harmonics))
  f_ratios = f_ratios[tf.newaxis, tf.newaxis, :]
  harmonic_frequencies = frequencies * f_ratios
  return harmonic_frequencies


def harmonic_to_sinusoidal(harm_amp, harm_dist, f0_hz, sample_rate=16000):
  print('harmonic to sinusoids')
  """Converts controls for a harmonic synth to those for a sinusoidal synth."""
  n_harmonics = int(harm_dist.shape[-1])
  freqs = get_harmonic_frequencies(f0_hz, n_harmonics)
  # Double check to remove anything above Nyquist.
  harm_dist = remove_above_nyquist(freqs, harm_dist, sample_rate)
  # Renormalize after removing above nyquist.
  harm_dist_sum = tf.reduce_sum(harm_dist, axis=-1, keepdims=True)
  harm_dist = safe_divide(harm_dist, harm_dist_sum)
  amps = harm_amp * harm_dist
  return amps, freqs



def safe_divide(numerator, denominator, eps=1e-7):
  """Avoid dividing by zero by adding a small epsilon."""
  safe_denominator = tf.where(denominator == 0.0, eps, denominator)
  return numerator / safe_denominator


def remove_above_nyquist(frequency_envelopes: tf.Tensor,
                         amplitude_envelopes: tf.Tensor,
                         sample_rate: int = 16000) -> tf.Tensor:
  """Set amplitudes for oscillators above nyquist to 0.

  Args:
    frequency_envelopes: Sample-wise oscillator frequencies (Hz). Shape
      [batch_size, n_samples, n_sinusoids].
    amplitude_envelopes: Sample-wise oscillator amplitude. Shape [batch_size,
      n_samples, n_sinusoids].
    sample_rate: Sample rate in samples per a second.

  Returns:
    amplitude_envelopes: Sample-wise filtered oscillator amplitude.
      Shape [batch_size, n_samples, n_sinusoids].
  """
  print('removing above nyquist')
  frequency_envelopes = tf.cast(frequency_envelopes, tf.float32)
  amplitude_envelopes = tf.cast(amplitude_envelopes, tf.float32)

  amplitude_envelopes = tf.where(
      tf.greater_equal(frequency_envelopes, sample_rate / 2.0),
      tf.zeros_like(amplitude_envelopes), amplitude_envelopes)
  return amplitude_envelopes



def midi_to_hz(notes, midi_zero_silence: bool = False):
  """TF-compatible midi_to_hz function.

  Args:
    notes: Tensor containing encoded pitch in MIDI scale.
    midi_zero_silence: Whether to output 0 hz for midi 0, which would be
      convenient when midi 0 represents silence. By defualt (False), midi 0.0
      corresponds to 8.18 Hz.

  Returns:
    hz: Frequency of MIDI in hz, same shape as input.
  """
  print('midi to hz')
  notes = tf.cast(notes, tf.float32)
  hz = 440.0 * (2.0 ** ((notes - 69.0) / 12.0))
  # Map MIDI 0 as 0 hz when MIDI 0 is silence.
  if midi_zero_silence:
    hz = tf.where(tf.equal(notes, 0.0), 0.0, hz)
  return hz


def resample(inputs: tf.Tensor,
             n_timesteps: int,
             method,
             add_endpoint: bool = True) -> tf.Tensor:
  """Interpolates a tensor from n_frames to n_timesteps.

  Args:
    inputs: Framewise 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_frames],
      [batch_size, n_frames], [batch_size, n_frames, channels], or
      [batch_size, n_frames, n_freq, channels].
    n_timesteps: Time resolution of the output signal.
    method: Type of resampling, must be in ['nearest', 'linear', 'cubic',
      'window']. Linear and cubic ar typical bilinear, bicubic interpolation.
      'window' uses overlapping windows (only for upsampling) which is smoother
      for amplitude envelopes with large frame sizes.
    add_endpoint: Hold the last timestep for an additional step as the endpoint.
      Then, n_timesteps is divided evenly into n_frames segments. If false, use
      the last timestep as the endpoint, producing (n_frames - 1) segments with
      each having a length of n_timesteps / (n_frames - 1).
  Returns:
    Interpolated 1-D, 2-D, 3-D, or 4-D Tensor. Shape [n_timesteps],
      [batch_size, n_timesteps], [batch_size, n_timesteps, channels], or
      [batch_size, n_timesteps, n_freqs, channels].

  Raises:
    ValueError: If method is 'window' and input is 4-D.
    ValueError: If method is not one of 'nearest', 'linear', 'cubic', or
      'window'.
  """
  print('resampling')
  inputs = tf.cast(inputs, tf.float32)
  is_1d = len(inputs.shape) == 1
  is_2d = len(inputs.shape) == 2
  is_4d = len(inputs.shape) == 4

  # Ensure inputs are at least 3d.
  if is_1d:
    inputs = inputs[tf.newaxis, :, tf.newaxis]
  elif is_2d:
    inputs = inputs[:, :, tf.newaxis]

  def _image_resize(method):
    """Closure around tf.image.resize."""
    # Image resize needs 4-D input. Add/remove extra axis if not 4-D.
    outputs = inputs[:, :, tf.newaxis, :] if not is_4d else inputs
    outputs = tf.compat.v1.image.resize(outputs,
                                        [n_timesteps, outputs.shape[2]],
                                        method=method,
                                        align_corners=not add_endpoint)
    return outputs[:, :, 0, :] if not is_4d else outputs

  # Perform resampling.
  if method == 'nearest':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.NEAREST_NEIGHBOR)
  elif method == 'linear':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.BILINEAR)
  elif method == 'cubic':
    outputs = _image_resize(tf.compat.v1.image.ResizeMethod.BICUBIC)
  elif method == 'window':
    outputs = upsample_with_windows(inputs, n_timesteps, add_endpoint)
  else:
    raise ValueError('Method ({}) is invalid. Must be one of {}.'.format(
        method, "['nearest', 'linear', 'cubic', 'window']"))

  # Return outputs to the same dimensionality of the inputs.
  if is_1d:
    outputs = outputs[0, :, 0]
  elif is_2d:
    outputs = outputs[:, :, 0]

  return outputs

def exp_sigmoid(x, exponent=10.0, max_value=2.0, threshold=1e-7):
  """Exponentiated Sigmoid pointwise nonlinearity.

  Bounds input to [threshold, max_value] with slope given by exponent.

  Args:
    x: Input tensor.
    exponent: In nonlinear regime (away from x=0), the output varies by this
      factor for every change of x by 1.0.
    max_value: Limiting value at x=inf.
    threshold: Limiting value at x=-inf. Stablizes training when outputs are
      pushed to 0.

  Returns:
    A tensor with pointwise nonlinearity applied.
  """
  print('exp sigmoid')
  x = tf.cast(x, tf.float32)
  #tf.cast(exponent, tf.float32)
  return max_value * tf.nn.sigmoid(x)**tf.math.log(exponent) + threshold


def flip(p=0.5):
  return np.random.uniform() <= p


def uniform_int(minval=0, maxval=10):
  return np.random.random_integers(int(minval), int(maxval))


def uniform_float(minval=0.0, maxval=10.0):
  return np.random.uniform(float(minval), float(maxval))


def uniform_generator(sample_shape, n_timesteps, minval, maxval,
                      method='linear'):
  print('uniform generator')
  """Linearly interpolates between a fixed number of uniform samples."""
  signal = np.random.uniform(minval, maxval, sample_shape)
  return resample(signal, n_timesteps, method=method)


def normal_generator(sample_shape, n_timesteps, mean, stddev, method='linear'):
  print('normal generator')
  """Linearly interpolates between a fixed number of uniform samples."""
  signal = np.random.normal(mean, stddev, sample_shape)
  return resample(signal, n_timesteps, method=method)


def modulate(signal, maxval=0.5, n_t=10, method='linear'):
  print('modulate')
  """Generate abs(normal noise) with stddev from uniform, multiply original."""
  n_batch, n_timesteps, _ = signal.shape
  signal_std = np.random.uniform(0.0, maxval, n_batch)
  mod = np.abs(np.random.normal(1.0, signal_std, [1, n_t, 1]))
  mod = np.transpose(mod, [2, 1, 0])
  mod = resample(mod, n_timesteps, method=method)
  return signal * mod


@gin.configurable
def generate_notes(n_batch,
                   n_timesteps,
                   n_harmonics=100,
                   n_mags=65,
                   get_controls=True):
  """Generate self-supervision signal of discrete notes."""
  print('generate notes')
  n_notes = uniform_int(1, 20)

  # Amplitudes.
  method = 'nearest' if flip(0.5) else 'linear'
  harm_amp = uniform_generator([n_batch, n_notes, 1], n_timesteps,
                               minval=-2, maxval=2, method=method)
  if get_controls:
    harm_amp = exp_sigmoid(harm_amp)

  # Frequencies.
  note_midi = uniform_generator([n_batch, n_notes, 1], n_timesteps,
                                minval=24.0, maxval=84.0, method='nearest')
  f0_hz = midi_to_hz(note_midi)

  # Harmonic Distribution
  method = 'nearest' if flip(0.5) else 'linear'
  n_lines = 10
  exponents = [uniform_float(1.0, 6.0) for _ in range(n_lines)]
  harm_dist_lines = [-tf.linspace(0.0, float(i), n_harmonics)**exponents[i]
                     for i in range(n_lines)]
  harm_dist_lines = tf.stack(harm_dist_lines)
  lines_dist = uniform_generator([n_batch, n_notes, n_lines], n_timesteps,
                                 minval=0.0, maxval=1.0, method=method)
  harm_dist = (lines_dist[..., tf.newaxis] *
               harm_dist_lines[tf.newaxis, tf.newaxis, :])
  harm_dist = tf.reduce_sum(harm_dist, axis=-2)

  if get_controls:
    harm_dist = exp_sigmoid(harm_dist)
    harm_dist = remove_above_nyquist(f0_hz, harm_dist)
    harm_dist = safe_divide(
        harm_dist, tf.reduce_sum(harm_dist, axis=-1, keepdims=True))

  # Noise Magnitudes.
  method = 'nearest' if flip(0.5) else 'linear'
  mags = uniform_generator([n_batch, n_notes, n_mags], n_timesteps,
                           minval=-6.0, maxval=uniform_float(-4.0, 0.0),
                           method=method)
  if get_controls:
    mags = exp_sigmoid(mags)

  sin_amps, sin_freqs = harmonic_to_sinusoidal(
      harm_amp, harm_dist, f0_hz)

  controls = {'harm_amp': harm_amp,
              'harm_dist': harm_dist,
              'f0_hz': f0_hz,
              'sin_amps': sin_amps,
              'sin_freqs': sin_freqs,
              'noise_magnitudes': mags}
  return controls


def random_blend(length, env_start=1.0, env_end=0.0, exp_max=2.0):
  print('random blending whatever that is')
  """Returns a linear mix between two values, with a random curve steepness."""
  exp = uniform_float(-exp_max, exp_max)
  v = np.linspace(1.0, 0.0, length) ** (2.0 ** exp)
  return env_start * v + env_end * (1.0 - v)


def random_harm_dist(n_harmonics=100, low_pass=True, rand_phase=0.0):
  print('random harm dist')
  """Create harmonic distribution out of sinusoidal components."""
  n_components = uniform_int(1, 20)
  smoothness = uniform_float(1.0, 10.0)
  coeffs = np.random.rand(n_components)
  freqs = np.random.rand(n_components) * n_harmonics / smoothness

  v = []
  for i in range(n_components):
    v_i = (coeffs[i] * np.cos(
        np.linspace(0.0, 2.0 * np.pi * freqs[i], n_harmonics) +
        uniform_float(0.0, np.pi * 2.0 * rand_phase)))
    v.append(v_i)

  if low_pass:
    v = [v_i * np.linspace(1.0, uniform_float(0.0, 0.5), n_harmonics) **
         uniform_float(0.5, 2.0) for v_i in v]
  harm_dist = np.sum(np.stack(v), axis=0)
  return harm_dist


@gin.configurable
def generate_notes_v2(n_batch=2,
                      n_timesteps=125,
                      n_harmonics=100,
                      n_mags=65,
                      min_note_length=5,
                      max_note_length=25,
                      p_silent=0.1,
                      p_vibrato=0.5,
                      get_controls=True):
  print('generate ntoes v2')
  """Generate more expressive self-supervision signal of discrete notes."""
  harm_amp = np.zeros([n_batch, n_timesteps, 1])
  harm_dist = np.zeros([n_batch, n_timesteps, n_harmonics])
  f0_midi = np.zeros([n_batch, n_timesteps, 1])
  mags = np.zeros([n_batch, n_timesteps, n_mags])

  for b in range(n_batch):
    t_start = 0
    while t_start < n_timesteps:
      note_length = uniform_int(min_note_length, max_note_length)
      t_end = min(t_start + note_length, n_timesteps)
      note_length = t_end - t_start

      # Silent?
      silent = flip(p_silent)
      if silent:
        # Amplitudes.
        ha_slice = harm_amp[b, t_start:t_end, :]
        ha_slice -= 10.0

      else:
        # Amplitudes.
        amp_start = uniform_float(-1.0, 3.0)
        amp_end = uniform_float(-1.0, 3.0)
        amp_blend = random_blend(note_length, amp_start, amp_end)
        ha_slice = harm_amp[b, t_start:t_end, :]
        ha_slice += amp_blend[:, np.newaxis]

        # Add some noise.
        ha_slice += uniform_float(0.0, 0.1) * np.random.randn(*ha_slice.shape)

        # Harmonic Distribution.
        low_pass = flip(0.8)
        rand_phase = uniform_float(0.0, 0.4)
        harm_dist_start = random_harm_dist(n_harmonics,
                                           low_pass=low_pass,
                                           rand_phase=rand_phase)[np.newaxis, :]
        harm_dist_end = random_harm_dist(n_harmonics,
                                         low_pass=low_pass,
                                         rand_phase=rand_phase)[np.newaxis, :]
        blend = random_blend(note_length, 1.0, 0.0)[:, np.newaxis]
        harm_dist_blend = (harm_dist_start * blend +
                           harm_dist_end * (1.0 - blend))
        hd_slice = harm_dist[b, t_start:t_end, :]
        hd_slice += harm_dist_blend

        # Add some noise.
        hd_slice += uniform_float(0.0, 0.5) * np.random.randn(*hd_slice.shape)

        # Fundamental Frequency.
        f0 = uniform_float(24.0, 84.0)
        if flip(p_vibrato):
          vib_start = uniform_float(0.0, 1.0)
          vib_end = uniform_float(0.0, 1.0)
          vib_periods = uniform_float(0.0, note_length * 2.0 / min_note_length)
          vib_blend = random_blend(note_length, vib_start, vib_end)
          vib = vib_blend * np.sin(
              np.linspace(0.0, 2.0 * np.pi * vib_periods, note_length))
          f0_note = f0 + vib
        else:
          f0_note = f0 * np.ones([note_length])

        f0_slice = f0_midi[b, t_start:t_end, :]
        f0_slice += f0_note[:, np.newaxis]

        # Add some noise.
        f0_slice += uniform_float(0.0, 0.1) * np.random.randn(*f0_slice.shape)

      # Filtered Noise.
      low_pass = flip(0.8)
      rand_phase = uniform_float(0.0, 0.4)
      mags_start = random_harm_dist(n_mags,
                                    low_pass=low_pass,
                                    rand_phase=rand_phase)[np.newaxis, :]
      mags_end = random_harm_dist(n_mags,
                                  low_pass=low_pass,
                                  rand_phase=rand_phase)[np.newaxis, :]
      blend = random_blend(note_length, 1.0, 0.0)[:, np.newaxis]
      mags_blend = mags_start * blend + mags_end * (1.0 - blend)

      mags_slice = mags[b, t_start:t_end, :]
      mags_slice += mags_blend

      # Add some noise.
      mags_slice += uniform_float(0.0, 0.2) * np.random.randn(*mags_slice.shape)

      # # Scale.
      mags_slice -= uniform_float(1.0, 10.0)

      t_start = t_end

  if get_controls:
    harm_amp = exp_sigmoid(harm_amp)
    harm_amp /= uniform_float(1.0, [2.0, uniform_float(2.0, 10.0)][flip(0.2)])

  # Frequencies.
  f0_hz = midi_to_hz(f0_midi)

  if get_controls:
    harm_dist = tf.nn.softmax(harm_dist)
    harm_dist = remove_above_nyquist(f0_hz, harm_dist)
    harm_dist = safe_divide(
        harm_dist, tf.reduce_sum(harm_dist, axis=-1, keepdims=True))

  if get_controls:
    mags = exp_sigmoid(mags)

  sin_amps, sin_freqs = harmonic_to_sinusoidal(
      harm_amp, harm_dist, f0_hz)

  controls = {'harm_amp': harm_amp,
              'harm_dist': harm_dist,
              'f0_hz': f0_hz,
              'sin_amps': sin_amps,
              'sin_freqs': sin_freqs,
              'noise_magnitudes': mags}
  return controls


x = generate_notes_v2(n_batch=1,
                      n_timesteps=125,
                      n_harmonics=100,
                      n_mags=65,
                      min_note_length=5,
                      max_note_length=25,
                      p_silent=0.1,
                      p_vibrato=0.5,
                      get_controls=True)

print(type(x))
#print(x['harm_amp'])