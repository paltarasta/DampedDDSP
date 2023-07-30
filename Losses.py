import Helpers as h
import torch
import torch.distributions as td
import torch.nn as nn
from einops import rearrange

####################################################################################################################
############## EVERYTHING IN THIS FILE UNLESS STATED OTHERWISE HAS BEEN REWRITTEN FROM MAGENTA'S INVERSE ###########
############## SYNTHESIS CODE.     https://github.com/magenta/ddsp/blob/main/ddsp/losses.py              ###########
####################################################################################################################


def get_p_sinusoids_given_harmonics():
#Gets distribution of sinusoids given harmonics from candidate f0s.
    n_harmonic_gaussians = 30
    harmonics_probs = torch.ones(n_harmonic_gaussians) / n_harmonic_gaussians
    harmonics_loc = torch.range(1, n_harmonic_gaussians+1)
    harmonics_scale = 0.2
    
    mix = td.categorical.Categorical(harmonics_probs)
    comp = td.normal.Normal(loc=harmonics_loc, scale=harmonics_scale)
    prob_s_given_h = td.mixture_same_family.MixtureSameFamily(mix, comp)

    return prob_s_given_h


def get_p_harmonics_given_sinusoids(freqs, amps):
#Gets distribution of harmonics from candidate f0s given sinusoids.

    sinusoids_scale = 0.5
    sinusoids_midi = h.hz_to_midi(freqs)
    eta = 1e-7
    amps = torch.where(amps == 0, eta*torch.ones_like(amps), amps)
    amps_norm = h.safe_divide(amps, torch.sum(amps, dim=-1, keepdim=True))

    mix = td.categorical.Categorical(probs=amps_norm)
    comp = td.normal.Normal(loc=sinusoids_midi, scale=sinusoids_scale)
    prob_h_given_s = td.mixture_same_family.MixtureSameFamily(mix, comp)

    return prob_h_given_s


def get_candidate_harmonics(f0_candidates):
    n_harmonic_points = 10
    n = torch.range(1, n_harmonic_points+1)
    harmonics = f0_candidates.unsqueeze(-1) * n.unsqueeze(0).unsqueeze(0).unsqueeze(0)
    harmonics = h.hz_to_midi(harmonics)
    return harmonics


def get_loss_tensors(f0_candidates, freqs, amps):
    n_harmonic_points = 10

    #Sinusoids loss tensor
    prob_s_given_h = get_p_sinusoids_given_harmonics()

    freq_ratios = h.safe_divide(freqs.unsqueeze(2), f0_candidates.unsqueeze(-1))
    nll_sinusoids = prob_s_given_h.log_prob(freq_ratios)

    a = amps.unsqueeze(2)

    sinusoids_loss = h.safe_divide(torch.sum(nll_sinusoids * a, dim=-1), torch.sum(a, dim=-1))

    #Harmonics loss tensor
    prob_h_given_s = get_p_harmonics_given_sinusoids(freqs, amps)
    harmonics = get_candidate_harmonics(f0_candidates)
    harmonics_transpose = rearrange(harmonics, 'a b c d -> c d a b')
    nll_harmonics_transpose = -prob_h_given_s.log_prob(harmonics_transpose)
    nll_harmonics = rearrange(nll_harmonics_transpose, 'a b c d -> c d a b')

    amps_prior = torch.linspace(1.0, 1.0/n_harmonic_points, n_harmonic_points)
    harmonics_loss = nll_harmonics * amps_prior.unsqueeze(0).unsqueeze(0).unsqueeze(0)

    nyquist_midi = h.hz_to_midi(16000/2.0)
    nyquist_mask = torch.where(harmonics < nyquist_midi, torch.ones_like(harmonics_loss), torch.zeros_like(harmonics_loss))
    harmonics_loss *= h.safe_divide(nyquist_mask, torch.mean(nyquist_mask, dim=-1, keepdim=True))

    harmonics_loss = torch.mean(harmonics, dim=-1)

    return sinusoids_loss, harmonics_loss

def TWM_loss(f0_candidates, freqs, amps):
    softmax = nn.Softmax(dim=-1)
    sinusoids_loss, harmonics_loss = get_loss_tensors(f0_candidates, freqs, amps)
    combined_loss = sinusoids_loss + harmonics_loss #the weigting is 1 for both, why is it there?
    softmin_loss = combined_loss * softmax(-combined_loss / 1.0)
    softmin_loss = torch.mean(softmin_loss)
    return softmin_loss



def kernel_density_estimate(amps, freqs, scale):
    freqs_midi = h.hz_to_midi(freqs)

    eta = 1e-7
    amps = torch.where(amps == 0, eta*torch.ones_like(amps), amps)
    amps_norm = h.safe_divide(amps, torch.sum(amps, dim=-1, keepdim=True))

    mix = td.categorical.Categorical(probs=amps_norm)
    comp = td.normal.Normal(loc=freqs_midi, scale=scale)
    kde = td.mixture_same_family.MixtureSameFamily(mix, comp)

    return kde


def nll(amps, freqs, amps_target, freqs_target, scale_target):
    p_source_given_target = kernel_density_estimate(amps_target, freqs_target, scale_target)

    freqs_midi = h.hz_to_midi(freqs)

    freqs_transpose = rearrange(freqs_midi, 'a b c -> c a b')
    nll_transpose = - p_source_given_target.log_prob(freqs_transpose)
    nll = rearrange(nll_transpose, 'a b c -> b c a')

    amps_norm = h.safe_divide(amps, torch.sum(amps, dim=-1, keepdim=True))
    nll_final = torch.mean(nll*amps_norm, dim=-1)

    return nll_final


def KDEConsistencyLoss(amps_a, freqs_a, amps_b, freqs_b):
    loss = 0
    weight_a = 1.0
    weight_b = 1.0
    weight_mean_amp = 1.0
    scale_a = 0.1
    scale_b = 0.1


    print('KDE SHAPES')
    print(amps_a.shape, freqs_a.shape, amps_b.shape, freqs_b.shape)

    if weight_a > 0: #why do we need the if statements?
        loss_a = nll(amps_a, freqs_a, amps_b, freqs_b, scale_b)
        loss += torch.mean(weight_a * loss_a)
    if weight_b > 0:
        loss_b = nll(amps_b, freqs_b, amps_a, freqs_a, scale_a)
        loss += torch.mean(weight_b * loss_b)
    if weight_mean_amp > 0:
        mean_amp_a = torch.mean(amps_a, dim=-1)
        mean_amp_b = torch.mean(amps_b, dim=-1)
        loss_mean_amp = torch.mean(torch.abs(mean_amp_a - mean_amp_b))
        loss += weight_mean_amp * loss_mean_amp
    
    return loss


#################################################################################################################
######################################### Self - supervised loss ################################################
#################################################################################################################

def mean_difference(target, value, weights=None):

    # Using L1 loss

    difference = target - value
    weights = 1.0 if weights is None else weights
    loss = torch.mean(torch.abs(difference*weights))

    return loss


def amp_loss(amp, amp_target, weights=None):

    loss = mean_difference(amp, amp_target, weights)

    return loss


def freq_loss(f_hz, f_hz_target, weights=None):

    f_midi = h.hz_to_midi(f_hz)
    f_midi_target = h.hz_to_midi(f_hz_target)

    loss = mean_difference(f_midi, f_midi_target, weights)

    return loss

### Self_supervision loss Lss
def HarmonicConsistencyLoss(glob_amp,
                            glob_amp_target,
                            harm_dist,
                            harm_dist_target,
                            f0_hz,
                            f0_hz_target,
                            sin_amps,
                            sin_amps_target,
                            sin_freqs,
                            sin_freqs_target,
                            amp_weight=10.0,
                            dist_weight=100.0,
                            f0_weight=1.0,
                            sin_con_weight=0.1,
                            amp_threshold=1e-4):
    
    # Mask loss where target audio is below threshold amplitude.
    weights = (glob_amp_target >= amp_threshold).float()

    # Harmonic amplitude
    harm_amp_loss = amp_loss(glob_amp, glob_amp_target)
    harm_amp_loss = amp_weight * harm_amp_loss

    # Harmonic distribution
    harm_dist_loss = amp_loss(harm_dist, harm_dist_target, weights=weights)
    harm_dist_loss = dist_weight * harm_dist_loss

    # Fundamental frequency
    f0_hz_loss = freq_loss(f0_hz, f0_hz_target, weights=weights)
    f0_hz_loss = f0_weight * f0_hz_loss

    # Sinusoidal consistency
    sin_con_loss = KDEConsistencyLoss(sin_amps, sin_freqs, sin_amps_target, sin_freqs_target)
    sin_con_loss = sin_con_weight * sin_con_loss

    SS_loss = harm_amp_loss + harm_dist_loss + f0_hz_loss + sin_con_loss

    return SS_loss