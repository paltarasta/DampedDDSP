import torch
import math


def damped_synth(pitches, amplitudes, damping, sampling_rate):
    assert pitches.shape[-1] == amplitudes.shape[-1]
    indices = torch.arange(pitches.size(1)).unsqueeze(0).unsqueeze(-1)
    damper = torch.exp(- damping.abs() * indices)
    omegas = torch.cumsum(2 * math.pi * pitches / sampling_rate, 1)
    signal = (torch.cos(omegas) * amplitudes * damper).sum(-1, keepdim=True)
    return signal

def sinusoidal_synth(pitches, amplitudes, sampling_rate):
    assert pitches.shape[-1] == amplitudes.shape[-1]
    omegas = torch.cumsum(2 * math.pi * pitches / sampling_rate, 1)
    signal = (torch.sin(omegas) * amplitudes).sum(-1, keepdim=True)
    return signal