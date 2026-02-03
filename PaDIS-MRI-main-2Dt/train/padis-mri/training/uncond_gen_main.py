import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'eval'))
from eval.denoise_padding import denoisedFromPatches, getIndices

import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import random


@torch.no_grad()
def dps_uncond(
    net,
    batch_size=1,
    resolution=384,
    psize=96,  # was 96
    pad=96,     # was 96             
    num_steps=50,
    sigma_min=0.003,
    sigma_max=10,
    rho=7,
    device='cuda',
    randn_like=torch.randn_like,
):
    """
    Unconditional generation with a patch-based denoising approach (similar to your original dps),
    but WITHOUT measurement consistency or extra MRI logic.

    - net: your trained diffusion model (expects real+imag or however your channels are arranged).
    - batch_size, channels, resolution: the shape of the images you want to sample.
    - psize, pad: if you still want to do patch-based denoising, you can keep these.
    - num_steps, sigma_min, sigma_max, rho: define the noise schedule and # of sampling steps.
    - device: 'cuda' or 'cpu'.
    """

    # Switch to eval mode (so no dropout, etc.).
    was_training = net.training
    net.eval()

    shape = (batch_size, 1, resolution, resolution)

    x_init = torch.zeros(shape, dtype=torch.complex64, device=device)  # or you can skip x_init entirely
    
    if callable(randn_like):
        x = sigma_max * randn_like(x_init)
    else:
        x = sigma_max * torch.randn_like(x_init)
    
    if pad > 0:
        x = F.pad(x, (pad, pad, pad, pad), 'constant', 0)

    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + (step_indices / (num_steps - 1)) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho

    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    patches = (resolution // psize) + 1  # if still doing patch-based
    spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)
    
    x_start = 0
    y_start = 0
    resolution = resolution + 2*pad
    x_pos = torch.arange(x_start, x_start+resolution).view(1, -1).repeat(resolution, 1)
    y_pos = torch.arange(y_start, y_start+resolution).view(-1, 1).repeat(1, resolution)
    x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
    y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
    latents_pos = torch.stack([x_pos, y_pos], dim=0).to(device)
    latents_pos = latents_pos.unsqueeze(0).repeat(1, 1, 1, 1)
    
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        alpha = 0.5 * t_cur ** 2
        
        for j in range(2):
            indices = getIndices(spaced, patches, pad, psize)  # presumably your own function

            x_ri = torch.view_as_real(x.squeeze(1)).permute(0, 3, 1, 2)  # Original logic
            D_ri = denoisedFromPatches(net, x_ri, t_cur, latents_pos, None,  
                                       indices, t_goal=0, wrong=False)

            # Merge back to complex if needed:
            D = torch.complex(D_ri[:, 0], D_ri[:, 1])  # shape [B,H,W]
            D = D.unsqueeze(1)  # shape [B,1,H,W]
            score = (D - x) / (t_cur ** 2)
            z = randn_like(x)

            if i < num_steps - 1:
                x = x + alpha/2 * score + torch.sqrt(alpha) * z
            else:
                x = x + alpha/2 * score
    if pad > 0:
        x = x[:, :, pad:-pad, pad:-pad]

    net.train(was_training)
    return x.detach()
