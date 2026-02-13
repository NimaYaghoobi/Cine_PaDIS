import os
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'eval'))
from eval.denoise_padding import denoisedFromPatches, getIndices

import torch
import torch.nn.functional as F
import numpy as np
import tqdm
import random

# @torch.no_grad()
# def dps_uncond(
#     net,
#     batch_size=1,
#     resolution=384,
#     psize=96,  # was 96
#     pad=96,     # was 96             
#     num_steps=50,
#     sigma_min=0.003,
#     sigma_max=10,
#     rho=7,
#     device='cuda',
#     randn_like=torch.randn_like,
# ):
#     """
#     Unconditional generation with a patch-based denoising approach (similar to your original dps),
#     but WITHOUT measurement consistency or extra MRI logic.

#     - net: your trained diffusion model (expects real+imag or however your channels are arranged).
#     - batch_size, channels, resolution: the shape of the images you want to sample.
#     - psize, pad: if you still want to do patch-based denoising, you can keep these.
#     - num_steps, sigma_min, sigma_max, rho: define the noise schedule and # of sampling steps.
#     - device: 'cuda' or 'cpu'.
#     """

#     # Switch to eval mode (so no dropout, etc.).
#     was_training = net.training
#     net.eval()

#     shape = (batch_size, 1, resolution, resolution)

#     x_init = torch.zeros(shape, dtype=torch.complex64, device=device)  # or you can skip x_init entirely
    
#     if callable(randn_like):
#         x = sigma_max * randn_like(x_init)
#     else:
#         x = sigma_max * torch.randn_like(x_init)
    
#     if pad > 0:
#         x = F.pad(x, (pad, pad, pad, pad), 'constant', 0)

#     step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
#     t_steps = (
#         sigma_max ** (1 / rho)
#         + (step_indices / (num_steps - 1)) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
#     ) ** rho

#     t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

#     patches = (resolution // psize) + 1  # if still doing patch-based
#     spaced = np.linspace(0, (patches - 1) * psize, patches, dtype=int)
    
#     x_start = 0
#     y_start = 0
#     resolution = resolution + 2*pad
#     x_pos = torch.arange(x_start, x_start+resolution).view(1, -1).repeat(resolution, 1)
#     y_pos = torch.arange(y_start, y_start+resolution).view(-1, 1).repeat(1, resolution)
#     x_pos = (x_pos / (resolution - 1) - 0.5) * 2.
#     y_pos = (y_pos / (resolution - 1) - 0.5) * 2.
#     latents_pos = torch.stack([x_pos, y_pos], dim=0).to(device)
#     latents_pos = latents_pos.unsqueeze(0).repeat(1, 1, 1, 1)
    
#     for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
#         alpha = 0.5 * t_cur ** 2
        
#         for j in range(2):
#             # indices = getIndices(spaced, patches, pad, psize)  # presumably your own function
#             indices = getIndices((spaced_h, spaced_w), (patches_h, patches_w), pad, psize)

#             x_ri = torch.view_as_real(x.squeeze(1)).permute(0, 3, 1, 2)  # Original logic
#             D_ri = denoisedFromPatches(net, x_ri, t_cur, latents_pos, None,  
#                                        indices, t_goal=0, wrong=False)

#             # Merge back to complex if needed:
#             D = torch.complex(D_ri[:, 0], D_ri[:, 1])  # shape [B,H,W]
#             D = D.unsqueeze(1)  # shape [B,1,H,W]
#             score = (D - x) / (t_cur ** 2)
#             z = randn_like(x)

#             if i < num_steps - 1:
#                 x = x + alpha/2 * score + torch.sqrt(alpha) * z
#             else:
#                 x = x + alpha/2 * score
#     if pad > 0:
#         x = x[:, :, pad:-pad, pad:-pad]

#     net.train(was_training)
#     return x.detach()

@torch.no_grad()
def dps_uncond(
    net,
    batch_size=1,
    resolution=(246, 384),   # NOW supports (H, W)
    psize=64,
    # pad=96,
    pad=(0, 0), # (pad_y, pad_x)
    num_steps=50,
    sigma_min=0.003,
    sigma_max=10,
    rho=7,
    device='cuda',
    randn_like=torch.randn_like,
):
    was_training = net.training
    net.eval()

    # --- NEW: rectangular H,W ---
    try:
        H, W = resolution
    except TypeError:
        H = W = int(resolution)

    pad_y, pad_x = pad
    pad_y = int(pad_y)
    pad_x = int(pad_x)
    # init
    shape = (batch_size, 1, H, W)
    x_init = torch.zeros(shape, dtype=torch.complex64, device=device)
    x = sigma_max * (randn_like(x_init) if callable(randn_like) else torch.randn_like(x_init))

    # if pad > 0:
    #     x = F.pad(x, (pad, pad, pad, pad), 'constant', 0)

    if pad_y > 0 or pad_x > 0:
        x = F.pad(x, (pad_x, pad_x, pad_y, pad_y), 'constant', 0)

    # schedule
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=device)
    t_steps = (
        sigma_max ** (1 / rho)
        + (step_indices / (num_steps - 1)) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))
    ) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])

    # --- NEW: rectangular patch grid computed from (H,W) ---
    patches_hw = (H // psize + 1, W // psize + 1)
    spaced_hw = (
        np.linspace(0, (patches_hw[0] - 1) * psize, patches_hw[0], dtype=int),  # y starts
        np.linspace(0, (patches_hw[1] - 1) * psize, patches_hw[1], dtype=int),  # x starts
    )

    # --- NEW: rectangular positional grid on padded size ---
    # Hp, Wp = H + 2 * pad, W + 2 * pad
    Hp, Wp = H + 2 * pad_y, W + 2 * pad_x
    
    x_pos = torch.arange(Wp, device=device).view(1, -1).repeat(Hp, 1)       # [Hp, Wp]
    y_pos = torch.arange(Hp, device=device).view(-1, 1).repeat(1, Wp)       # [Hp, Wp]
    x_pos = (x_pos / (Wp - 1) - 0.5) * 2.0
    y_pos = (y_pos / (Hp - 1) - 0.5) * 2.0
    latents_pos = torch.stack([x_pos, y_pos], dim=0).unsqueeze(0)           # [1, 2, Hp, Wp]

    # sampling loop
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        alpha = 0.5 * t_cur ** 2

        for j in range(2):
            # indices = getIndices(spaced_hw, patches_hw, pad, psize)
            indices = getIndices(spaced_hw, patches_hw, (pad_y, pad_x), psize)

            x_ri = torch.view_as_real(x.squeeze(1)).permute(0, 3, 1, 2)  # [B,2,Hp,Wp]
            # D_ri = denoisedFromPatches(
            #     net, x_ri, t_cur, latents_pos, None,
            #     indices,
            #     pad=pad,          # IMPORTANT: match the padding you used
            #     t_goal=0,
            #     wrong=False
            # )

            D_ri = denoisedFromPatches(
                net, x_ri, t_cur, latents_pos, None,
                indices,
                pad=(pad_y, pad_x),
                t_goal=0,
                wrong=False
            )

            D = torch.complex(D_ri[:, 0], D_ri[:, 1]).unsqueeze(1)        # [B,1,Hp,Wp]
            score = (D - x) / (t_cur ** 2)
            z = randn_like(x)

            if i < num_steps - 1:
                x = x + alpha/2 * score + torch.sqrt(alpha) * z
            else:
                x = x + alpha/2 * score

    # crop back to (H,W)
    # if pad > 0:
    #     x = x[:, :, pad:pad+H, pad:pad+W]

    if pad_y > 0 or pad_x > 0:
        x = x[:, :, pad_y:pad_y+H, pad_x:pad_x+W]

    net.train(was_training)
    return x.detach()