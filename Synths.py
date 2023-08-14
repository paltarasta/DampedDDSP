import torch
import math
import Helpers as h
from einops import rearrange


def damped_synth(pitches, amplitudes, damper, sampling_rate):
    assert pitches.shape[-1] == amplitudes.shape[-1]
    omegas = torch.cumsum(2 * math.pi * pitches / sampling_rate, 1)
    signal = (torch.cos(omegas) * amplitudes * damper).sum(-1, keepdim=True)
    return signal


def sinusoidal_synth(pitches, amplitudes, sampling_rate):
    pitches = h.UpsampleTime(pitches)
    amplitudes = h.UpsampleTime(amplitudes)
    assert pitches.shape[-1] == amplitudes.shape[-1]
    omegas = torch.cumsum(2 * math.pi * pitches / sampling_rate, 1)
    signal = (torch.cos(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal


def harmonic_synth(pitch, amplitudes, sampling_rate):
    n_harmonic = amplitudes.shape[-1]
    omega = torch.cumsum(2 * math.pi * pitch / sampling_rate, 1)
    omegas = omega * torch.arange(1, n_harmonic + 1).to(omega)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal