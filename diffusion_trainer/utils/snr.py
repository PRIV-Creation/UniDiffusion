import torch


def compute_snr(timesteps, noise_scheduler):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod ** 0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(device=timesteps.device)[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def snr_loss(snr_gamma, timesteps, noise_scheduler, model_pred, target):
    # Compute loss-weights as per Section 3.4 of https://arxiv.org/abs/2303.09556.
    # Since we predict the noise instead of x_0, the original formulation is slightly changed.
    # This is discussed in Section 4.2 of the same paper.
    snr = compute_snr(timesteps)
    mse_loss_weights = (
            torch.stack([snr, snr_gamma * torch.ones_like(timesteps)], dim=1).min(dim=1)[0] / snr
    )
    # We first calculate the original loss. Then we mean over the non-batch dimensions and
    # rebalance the sample-wise losses with their respective loss weights.
    # Finally, we take the mean of the rebalanced loss.
    loss = F.mse_loss(model_pred.float(), target.float(), reduction="none")
    loss = loss.mean(dim=list(range(1, len(loss.shape)))) * mse_loss_weights
    loss = loss.mean()
    return loss