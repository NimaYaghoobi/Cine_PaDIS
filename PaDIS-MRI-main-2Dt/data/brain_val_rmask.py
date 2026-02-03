import sys
import os
import argparse
import numpy as np
import h5py
import sigpy as sp
import glob
import random
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm_base

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dnnlib.util import configure_bart
configure_bart()

from bart import bart
import torch
from data_utils import forward_fs, normalization_const, tqdm

# CLI args
parser = argparse.ArgumentParser(description="Generate noisy MRI samples with multiple masks per R-level")
parser.add_argument("--num_seeds",  type=int, default=1,
                    help="Number of different random masks per R-level")
parser.add_argument("--start_seed", type=int, default=0,
                    help="Base seed for reproducible mask generation")
parser.add_argument('--noise_level', type=str, default="32dB", choices=["32dB", "22dB", "12dB"], help='Noise level to add')
parser.add_argument('--acs_size', type=int, default=20, help='Number of ACS lines to use')
parser.add_argument('--h5_folder', type=str, default="/data/datasets/fastmri/multicoil_val", help='Path to input folder containing .h5 files')
parser.add_argument('--output_root', type=str, default="/data/datasets/fastmri/", help='Path to output folder to save results')
parser.add_argument('--contrast', type=str, default="t1-flair", choices=["t1-flair", "t2"], help='Contrast to filter for')

args = parser.parse_args()

# Constants
center_slice = 2
ACS_size     = args.acs_size
H            = 384  # image dimension

# Set top-level random seed and derive mask seeds
random.seed(args.start_seed)
mask_seeds = [random.randint(0, 2**31 - 1) for _ in range(args.num_seeds)]
# R-levels to sample
R_levels = list(range(2, 11))  # 2..10

# Determine noise amplitude based on SNR setting
snr = args.noise_level
if   snr == "32dB": noise_amp = np.sqrt(0)
elif snr == "22dB": noise_amp = np.sqrt(10)
elif snr == "12dB": noise_amp = np.sqrt(100)

# Gather k-space files
ksp_files_train = sorted(glob.glob(args.h5_folder + "/*.h5"))
ksp_files = []

for files in ksp_files_train:
    if args.contrast == "t1-flair":
        if 'AXT1' in files or 'FLAIR' in files:
            ksp_files.append(files)
    elif args.contrast == "t2":
        if 'AXT2' in files:
            ksp_files.append(files)

print(f"Found {len(ksp_files)} {args.contrast} .h5 files in {args.h5_folder}")

total_iterations = len(ksp_files)
indexes = [i for i in range(total_iterations)]

# Main loop over all samples
for i in tqdm(range(len(ksp_files))):
    idx = indexes[i]
    fname = os.path.basename(ksp_files[idx])
    if args.contrast == "t1-flair":
        if 'AXT1POST' in fname: tag = 't1post'
        elif 'AXT1PRE'  in fname: tag = 't1pre'
        elif 'AXT1'     in fname: tag = 't1'
        elif 'FLAIR'    in fname: tag = 'flair'
        else:                   tag = 'other'
    else:
        tag = ""
    
    # 1) Load k-space
    with h5py.File(ksp_files[i], 'r') as contents:
        ksp = np.asarray(contents['kspace'][center_slice])
        ksp = ksp.transpose(1,2,0)

    # 2) Coil image via inverse FFT
    cimg = bart(1, 'fft -iu 3', ksp)
    # extract noise region and flatten
    noise   = sp.resize(cimg, [396, cimg.shape[1], cimg.shape[2]])[0:30,0:30]
    noise_flat = noise.reshape(-1, cimg.shape[2])
    # crop image to HxH
    cimg = sp.resize(cimg, [H, H, cimg.shape[2]])

    # 3) Whiten + add white noise
    cimg_white = bart(1, 'whiten', cimg[:,:,None,:], noise_flat[:,None,None,:]).squeeze()
    cimg_white = cimg_white + (noise_amp/np.sqrt(2)) * (
        np.random.normal(size=cimg_white.shape) + 1j*np.random.normal(size=cimg_white.shape)
    )

    # 4) Back to k-space + sensitivity maps
    ksp_white    = bart(1, 'fft -u 3', cimg_white)
    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white[:,:,None,:]).squeeze()

    # 5) Ground-truth recon
    gt_img_white_cropped = sp.resize(
        bart(1,'pics -S -i 30', ksp_white[:,:,None,:], s_maps_white[:,:,None,:]),
        [H, H]
    )

    # Reformat arrays to [coil, H, H]
    ksp_white    = ksp_white.transpose(2,0,1)
    s_maps_white = s_maps_white.transpose(2,0,1)
    cimg_white   = cimg_white.transpose(2,0,1)

    # 6) Compute normalization and renormalize
    norm_const_99_white = normalization_const(s_maps_white, gt_img_white_cropped)
    ksp_white    = ksp_white / norm_const_99_white
    s_maps_white = bart(1,'ecalib -m 1 -c0',
                         ksp_white.transpose(1,2,0)[:,:,None,:]
                       ).squeeze().transpose(2,0,1)

    gt_img_white_cropped = sp.resize(
        bart(1,'pics -S -i 30',
             ksp_white.transpose(1,2,0)[:,:,None,:],
             s_maps_white.transpose(1,2,0)[:,:,None,:]
        ), [H, H]
    )
    cimg_white = bart(1,'fft -iu 3', ksp_white.transpose(1,2,0)).transpose(2,0,1)
    var = np.var(cimg_white[:,0:30,0:30])

    # ——— Build 2D masks [len(R_levels), num_seeds, H, H]
    masks = []
    for R in R_levels:
        lines_total = H
        num_sampled = np.floor(lines_total / R)
        center_idx  = np.arange((lines_total-ACS_size)//2, (lines_total+ACS_size)//2)
        outer_idx   = np.setdiff1d(np.arange(lines_total), center_idx)

        this_R = []
        for seed in mask_seeds:
            np.random.seed(seed)
            choice = np.random.choice(
                outer_idx,
                size=int(num_sampled - ACS_size),
                replace=False
            )
            m = np.zeros((H, H), dtype=float)
            m[:, center_idx] = 1.
            m[:, choice]     = 1.
            this_R.append(m)
        masks.append(np.stack(this_R, axis=0))
    masks_arr = np.stack(masks, axis=0)

    # ——— Print summary
    print(f"\nStep {i} Done")
    print("white SNR:", 10*np.log10(1/var))
    print("gt norm:", np.linalg.norm(gt_img_white_cropped))
    for idx, R in enumerate(R_levels):
        density = (H*H)/masks_arr[idx,0].sum()
        print(f"Mask R={R}: {density:.2f}")

    # ——— Save sample_i.pt and noise_var_i.pt
    out_dir = args.output_root + f"/val_rmask_{args.contrast}/{snr}/"
    os.makedirs(out_dir, exist_ok=True)
    torch.save({
        'gt':             torch.tensor(gt_img_white_cropped, dtype=torch.complex64),
        'ksp':            torch.tensor(ksp_white,           dtype=torch.complex64),
        's_map':          torch.tensor(s_maps_white,        dtype=torch.complex64),
        'masks':          torch.tensor(masks_arr,           dtype=torch.float32),
        'mask_seeds':     mask_seeds,
        'norm_consts_99': norm_const_99_white,
    }, os.path.join(out_dir, f'sample_{tag}_{i}.pt'))

    torch.save({'noise_var_noisy': var},
               os.path.join(out_dir, f'noise_var_{tag}_{i}.pt'))
