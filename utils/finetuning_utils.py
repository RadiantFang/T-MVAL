import collections

import numpy as np
import torch
from scipy.stats import pearsonr

from Diffusion.diffusion import Diffusion
from Diffusion.unet import UNet
from utils.utils import extract

DNA_MAPPING = np.array(["A", "C", "G", "T"])


def freeze_module(module_to_freeze):
    module_to_freeze.eval()
    for parameter in module_to_freeze.parameters():
        parameter.requires_grad = False


def load_diffusion_model(
    checkpoint_path,
    device,
    *,
    dim=200,
    channels=1,
    dim_mults=(1, 2, 4),
    resnet_block_groups=4,
    timesteps=50,
    beta_start=0.0001,
    beta_end=0.2,
):
    checkpoint_state = torch.load(checkpoint_path, map_location=device)
    unet_backbone = UNet(
        dim=dim,
        channels=channels,
        dim_mults=list(dim_mults),
        resnet_block_groups=resnet_block_groups,
    )
    diffusion_model = Diffusion(
        unet_backbone,
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
    )
    diffusion_model.load_state_dict(checkpoint_state["model"])
    diffusion_model.to(device)
    return diffusion_model


def tensor_to_dna_numpy(token_tensor, base_mapping=DNA_MAPPING):
    token_array = token_tensor.cpu().numpy()
    base_array = base_mapping[token_array]
    return ["".join(base_row) for base_row in base_array]


def count_kmers(sequence_list, kmer_size):
    kmer_counts = collections.Counter()
    for sequence in sequence_list:
        if len(sequence) >= kmer_size:
            for start_index in range(len(sequence) - kmer_size + 1):
                kmer = sequence[start_index:start_index + kmer_size]
                kmer_counts[kmer] += 1
    return kmer_counts


def calculate_kmer_similarity(reference_counts, generated_counts):
    if not reference_counts or not generated_counts:
        return np.nan

    merged_kmers = sorted(list(set(reference_counts.keys()) | set(generated_counts.keys())))
    reference_total = sum(reference_counts.values())
    generated_total = sum(generated_counts.values())

    if reference_total == 0 or generated_total == 0:
        return np.nan

    reference_frequencies = [reference_counts.get(kmer, 0) / reference_total for kmer in merged_kmers]
    generated_frequencies = [generated_counts.get(kmer, 0) / generated_total for kmer in merged_kmers]

    if len(reference_frequencies) < 2:
        return np.nan

    correlation, _ = pearsonr(reference_frequencies, generated_frequencies)
    return correlation


def get_enformer(model, sequence_logits):
    token_indices = torch.argmax(sequence_logits, dim=1)
    one_hot_sequences = torch.nn.functional.one_hot(token_indices, num_classes=4).float()
    return model(one_hot_sequences)


def p_sample_guided(diffusion_model, noisy_states, timestep_tensor, timestep_index, device):
    betas_at_t = extract(diffusion_model.betas, timestep_tensor, noisy_states.shape, device)
    sqrt_one_minus_alphas_cumprod_at_t = extract(
        diffusion_model.sqrt_one_minus_alphas_cumprod,
        timestep_tensor,
        noisy_states.shape,
        device,
    )
    sqrt_recip_alphas_at_t = extract(
        diffusion_model.sqrt_recip_alphas,
        timestep_tensor,
        noisy_states.shape,
        device,
    )
    predicted_noise = diffusion_model.model(noisy_states, time=timestep_tensor)
    model_mean = sqrt_recip_alphas_at_t * (
        noisy_states - betas_at_t * predicted_noise / sqrt_one_minus_alphas_cumprod_at_t
    )

    if timestep_index == 0:
        return model_mean, predicted_noise

    posterior_variance_at_t = extract(diffusion_model.posterior_variance, timestep_tensor, noisy_states.shape)
    gaussian_noise = torch.randn_like(noisy_states)
    return model_mean + torch.sqrt(posterior_variance_at_t) * gaussian_noise, predicted_noise
