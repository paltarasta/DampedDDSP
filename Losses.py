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
