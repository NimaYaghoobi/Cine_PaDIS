import os
import sys

import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import random

@torch.no_grad()
def dps_uncond_edm(
    net,
    batch_size=1,
    resolution=384,
    num_steps=96,
    sigma_min=0.003,
    sigma_max=10,
    rho=7,
    device='cuda',
    randn_like=torch.randn_like,
):
    """
    Unconditional generation using a plain EDM model trained on full images (no patching).
    """
    was_training = net.training
    net.eval()

    shape = (batch_size, net.img_channels, resolution, resolution)
    x = sigma_max * randn_like(torch.empty(shape, device=device))

    # EDM noise schedule
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + (step_indices / (num_steps - 1)) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # final step to 0

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        t_cur = t_cur.float()
        t_next = t_next.float()
        sigma = t_cur

        # Score estimation
        denoised = net(x, sigma)  # EDM expects net(x, sigma)

        d = (x - denoised) / sigma  # score = (x - denoised) / sigma

        # Euler step
        dt = t_next - t_cur
        x = x + d * dt

    net.train(was_training)
    return x
