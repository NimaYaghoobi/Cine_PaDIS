import sys
import os
import numpy as np
import h5py
# import sigpy as sp  # not needed (we use crop+pad instead of resize)
import glob
import random
import argparse
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dnnlib.util import configure_bart
configure_bart()

from bart import bart
import torch

from data_utils import forward_fs, normalization_const, tqdm

# def crop_pad_center_to_square(x, target):
#     # x: (H, W, C)
#     H, W, C = x.shape

#     # center crop to at most target
#     crop_h = min(H, target)
#     crop_w = min(W, target)
#     y0 = (H - crop_h) // 2
#     x0 = (W - crop_w) // 2
#     x_crop = x[y0:y0+crop_h, x0:x0+crop_w, :]

#     # pad to target
#     pad_top = (target - crop_h) // 2
#     pad_bot = target - crop_h - pad_top
#     pad_left = (target - crop_w) // 2
#     pad_right = target - crop_w - pad_left

#     x_out = np.pad(
#         x_crop,
#         ((pad_top, pad_bot), (pad_left, pad_right), (0, 0)),
#         mode="constant"
#     )
#     return x_out, pad_top, pad_left


def crop_pad_center_to_square(x, target):

    H, W, C = x.shape
    target_h, target_w = target  # (H, W)
    crop_h = min(H, target_h)
    crop_w = min(W, target_w)
    y0 = (H - crop_h) // 2
    x0 = (W - crop_w) // 2
    x_crop = x[y0:y0+crop_h, x0:x0+crop_w, :]
    return x_crop, 0, 0


parser = argparse.ArgumentParser(description="Generate noisy MRI samples with multiple masks per R-level")
parser.add_argument('--noise_level', type=str, default="32dB", choices=["32dB", "22dB", "12dB"], help='Noise level to add')
parser.add_argument('--acs_size', type=int, default=20, help='Number of ACS lines to use')
parser.add_argument('--h5_folder', type=str, default="/data/datasets/fastmri/multicoil_val", help='Path to input folder containing .h5 files')
parser.add_argument('--output_root', type=str, default="/data/datasets/fastmri/", help='Path to output folder to save results')
parser.add_argument('--contrast', type=str, default="t1-flair", choices=["t1-flair", "t2"], help='Contrast to filter for')

args = parser.parse_args()

center_slice     = 2
ACS_size         = args.acs_size
imsize           = (246, 384)  # (H, W) center-crop only (no padding)

snr = args.noise_level
if snr == "32dB":
    noise_amp = np.sqrt(0)
elif snr == "22dB":
    noise_amp = np.sqrt(10)
elif snr == "12dB":
    noise_amp = np.sqrt(100)

# ksp_files_train = sorted(glob.glob(os.path.join(args.h5_folder, "P*/cine_sax.mat")))
# ksp_files = ksp_files_train

TARGET_KY, TARGET_KX = 246, 512

def _has_target_shape(mat_path):
    try:
        with h5py.File(mat_path, "r") as f:
            if "kspace_full" not in f:
                return False
            d = f["kspace_full"]  # (time, slice, coil, ky, kx)
            ky, kx = int(d.shape[-2]), int(d.shape[-1])
            return (ky == TARGET_KY) and (kx == TARGET_KX)
    except Exception:
        return False

ksp_files_all = sorted(glob.glob(os.path.join(args.h5_folder, "P*/cine_sax.mat")))
ksp_files = [p for p in ksp_files_all if _has_target_shape(p)]

print(f"Found {len(ksp_files_all)} cine_sax.mat total under {args.h5_folder}")
print(f"Keeping {len(ksp_files)} files with ky,kx={(TARGET_KY, TARGET_KX)}")

if not ksp_files:
    raise FileNotFoundError(
        f"No cine_sax.mat with ky,kx={(TARGET_KY, TARGET_KX)} found under {args.h5_folder}/P*/cine_sax.mat"
    )

# Note: some P*/ folders may not have cine_sax.mat; glob() naturally skips them.
print(f"Processing {len(ksp_files)} cardiac cine files (cine_sax.mat)  [label={args.contrast}]")

ksp_files = sorted(ksp_files)
total_iterations = len(ksp_files)
indexes = [i for i in range(total_iterations)]

for i in tqdm(range(total_iterations)):
    idx = indexes[i]
    slice_idx  = center_slice
    
    fname = os.path.basename(ksp_files[idx])
    
    if args.contrast == "t1-flair":
        if 'AXT1POST' in fname: tag = 't1post'
        elif 'AXT1PRE'  in fname: tag = 't1pre'
        elif 'AXT1'     in fname: tag = 't1'
        elif 'FLAIR'    in fname: tag = 'flair'
        else:                   tag = 'other'
    else:
        tag = ""
    
    # Load MRI samples and maps
    with h5py.File(ksp_files[idx], 'r') as contents:
        # Cardiac cine FullSample: (time, slice, coil, ky, kx) stored under 'kspace_full'
        dset = contents['kspace_full']
        T, S, C, Ky, Kx = dset.shape

        time_idx = T // 2
        slice_idx = S // 2

        ksp_struct = np.asarray(dset[time_idx, slice_idx, :, :, :])  # (coil, ky, kx)
        if (ksp_struct.dtype.fields is not None) and ('real' in ksp_struct.dtype.fields) and ('imag' in ksp_struct.dtype.fields):
            ksp = (ksp_struct['real'] + 1j * ksp_struct['imag']).astype(np.complex64).transpose(1, 2, 0)
        else:
            ksp = ksp_struct.astype(np.complex64).transpose(1, 2, 0)

    cimg = bart(1, 'fft -iu 3', ksp) # compare to `bart fft -iu 3 ksp cimg`

    # -------- crop+pad to 384x384 (NO RESIZE/RESAMPLE) --------
    cimg, pad_top, pad_left = crop_pad_center_to_square(cimg, imsize)

    imH, imW = cimg.shape[0], cimg.shape[1]  # rectangular size after crop

    # Noise patch from NON-PADDED region when possible (avoid all-zeros from padding)
    yN = min(30, imH - pad_top)
    xN = min(30, imW - pad_left)
    noise = cimg[pad_top:pad_top+yN, pad_left:pad_left+xN]
    noise_flat = np.reshape(noise, (-1, cimg.shape[2]))

    # Skip whitening if patch variance is ~0 (prevents BART whiten issues)
    percoil_var = np.var(noise_flat, axis=0) if noise_flat.size else np.array([0.0])
    can_whiten = np.all(np.isfinite(percoil_var)) and np.any(percoil_var > 0)

    if can_whiten:
        tmp = bart(1, 'whiten', cimg[:,:,None,:], noise_flat[:,None,None,:])
        cimg_white = tmp.squeeze() if tmp is not None else cimg
    else:
        cimg_white = cimg

    cimg_white = cimg_white + (noise_amp / np.sqrt(2))*(np.random.normal(size=cimg_white.shape) + 1j * np.random.normal(size=cimg_white.shape))

    # IMPORTANT (t2-style): we do NOT undersample here; eval code will apply masks to ksp
    ksp_white = bart(1, 'fft -u 3', cimg_white)
    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white[:,:,None,:]).squeeze()
    gt_img_white_cropped = bart(1, 'pics -S -i 30', ksp_white[:,:,None,:], s_maps_white[:,:,None,:]).squeeze()

    ksp_white = ksp_white.transpose(2, 0, 1)
    s_maps_white = s_maps_white.transpose(2, 0, 1)  
    cimg_white = cimg_white.transpose(2, 0, 1)  

    norm_const_99_white = normalization_const(s_maps_white, gt_img_white_cropped, ACS_size=ACS_size)
    ksp_white = ksp_white / norm_const_99_white
    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white.transpose(1, 2, 0)[:,:,None,:]).squeeze().transpose(2, 0, 1)

    gt_img_white_cropped = bart(1, 'pics -S -i 30', ksp_white.transpose(1, 2, 0)[:,:,None,:], s_maps_white.transpose(1, 2, 0)[:,:,None,:]).squeeze()
    cimg_white = bart(1, 'fft -iu 3', ksp_white.transpose(1, 2, 0)).transpose(2, 0, 1) # compare to `bart fft -iu 3 ksp cimg`
    yV = min(30, imH - pad_top)
    xV = min(30, imW - pad_left)
    var = np.var(cimg_white[:, pad_top:pad_top+yV, pad_left:pad_left+xV])

    total_lines = imH
    R = 2
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    # random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)

    n_rand = int(num_sampled_lines - acs_lines)
    n_rand = max(n_rand, 0)
    random_line_idx = np.random.choice(outer_line_idx, size=n_rand, replace=False) if n_rand > 0 else np.array([], dtype=int)
    if n_rand == 0:
        print(f"WARNING: Only ACS used (total_lines={total_lines}, R={R}, ACS={acs_lines})")

    mask = np.zeros((total_lines, imW))
    mask[center_line_idx, :] = 1.
    mask[random_line_idx, :] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_2 = mask[None]

    total_lines = imH
    R = 3
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    # random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    
    n_rand = int(num_sampled_lines - acs_lines)
    n_rand = max(n_rand, 0)
    random_line_idx = np.random.choice(outer_line_idx, size=n_rand, replace=False) if n_rand > 0 else np.array([], dtype=int)
    if n_rand == 0:
        print(f"WARNING: Only ACS used (total_lines={total_lines}, R={R}, ACS={acs_lines})")

    mask = np.zeros((total_lines, imW))
    mask[center_line_idx, :] = 1.
    mask[random_line_idx, :] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_3 = mask[None]

    total_lines = imH
    R = 4
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    # random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)

    n_rand = int(num_sampled_lines - acs_lines)
    n_rand = max(n_rand, 0)
    random_line_idx = np.random.choice(outer_line_idx, size=n_rand, replace=False) if n_rand > 0 else np.array([], dtype=int)
    if n_rand == 0:
        print(f"WARNING: Only ACS used (total_lines={total_lines}, R={R}, ACS={acs_lines})")

    mask = np.zeros((total_lines, imW))
    mask[center_line_idx, :] = 1.
    mask[random_line_idx, :] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_4 = mask[None]

    total_lines = imH
    R = 5
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    # random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    
    n_rand = int(num_sampled_lines - acs_lines)
    n_rand = max(n_rand, 0)
    random_line_idx = np.random.choice(outer_line_idx, size=n_rand, replace=False) if n_rand > 0 else np.array([], dtype=int)
    if n_rand == 0:
        print(f"WARNING: Only ACS used (total_lines={total_lines}, R={R}, ACS={acs_lines})")

    mask = np.zeros((total_lines, imW))
    mask[center_line_idx, :] = 1.
    mask[random_line_idx, :] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_5 = mask[None]

    total_lines = imH
    R = 6
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    # random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    
    n_rand = int(num_sampled_lines - acs_lines)
    n_rand = max(n_rand, 0)
    random_line_idx = np.random.choice(outer_line_idx, size=n_rand, replace=False) if n_rand > 0 else np.array([], dtype=int)
    if n_rand == 0:
        print(f"WARNING: Only ACS used (total_lines={total_lines}, R={R}, ACS={acs_lines})")

    mask = np.zeros((total_lines, imW))
    mask[center_line_idx, :] = 1.
    mask[random_line_idx, :] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_6 = mask[None]

    total_lines = imH
    R = 7
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    # random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    
    n_rand = int(num_sampled_lines - acs_lines)
    n_rand = max(n_rand, 0)
    random_line_idx = np.random.choice(outer_line_idx, size=n_rand, replace=False) if n_rand > 0 else np.array([], dtype=int)
    if n_rand == 0:
        print(f"WARNING: Only ACS used (total_lines={total_lines}, R={R}, ACS={acs_lines})")

    mask = np.zeros((total_lines, imW))
    mask[center_line_idx, :] = 1.
    mask[random_line_idx, :] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_7 = mask[None]

    total_lines = imH
    R = 8
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    # random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    
    n_rand = int(num_sampled_lines - acs_lines)
    n_rand = max(n_rand, 0)
    random_line_idx = np.random.choice(outer_line_idx, size=n_rand, replace=False) if n_rand > 0 else np.array([], dtype=int)
    if n_rand == 0:
        print(f"WARNING: Only ACS used (total_lines={total_lines}, R={R}, ACS={acs_lines})")

    mask = np.zeros((total_lines, imW))
    mask[center_line_idx, :] = 1.
    mask[random_line_idx, :] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_8 = mask[None]

    total_lines = imH
    R = 9
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    # random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    
    n_rand = int(num_sampled_lines - acs_lines)
    n_rand = max(n_rand, 0)
    random_line_idx = np.random.choice(outer_line_idx, size=n_rand, replace=False) if n_rand > 0 else np.array([], dtype=int)
    if n_rand == 0:
        print(f"WARNING: Only ACS used (total_lines={total_lines}, R={R}, ACS={acs_lines})")

    mask = np.zeros((total_lines, imW))
    mask[center_line_idx, :] = 1.
    mask[random_line_idx, :] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_9 = mask[None]

    total_lines = imH
    R = 10
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    # random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    
    n_rand = int(num_sampled_lines - acs_lines)
    n_rand = max(n_rand, 0)
    random_line_idx = np.random.choice(outer_line_idx, size=n_rand, replace=False) if n_rand > 0 else np.array([], dtype=int)
    if n_rand == 0:
        print(f"WARNING: Only ACS used (total_lines={total_lines}, R={R}, ACS={acs_lines})")

    mask = np.zeros((total_lines, imW))
    mask[center_line_idx, :] = 1.
    mask[random_line_idx, :] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_10 = mask[None]

    print('\n')
    print('white SNR: ' + str(10*np.log10(1/var)))
    print('gt norm: ' + str(np.linalg.norm(gt_img_white_cropped)))
    print("Mask R=2: " + str((imH*imW)/np.sum(mask_2)))
    print("Mask R=3: " + str((imH*imW)/np.sum(mask_3)))
    print("Mask R=4: " + str((imH*imW)/np.sum(mask_4)))
    print("Mask R=5: " + str((imH*imW)/np.sum(mask_5)))
    print("Mask R=6: " + str((imH*imW)/np.sum(mask_6)))
    print("Mask R=7: " + str((imH*imW)/np.sum(mask_7)))
    print("Mask R=8: " + str((imH*imW)/np.sum(mask_8)))
    print("Mask R=9: " + str((imH*imW)/np.sum(mask_9)))
    print("Mask R=10: " + str((imH*imW)/np.sum(mask_10)))
    print('\nStep ' + str(i) + ' Done')

    path = args.output_root + f"/val_{args.contrast}/" + str(snr) + "/"
    if not os.path.exists(path):
        os.makedirs(path)

    torch.save({'gt': torch.tensor(gt_img_white_cropped, dtype=torch.complex64),
                'ksp': torch.tensor(ksp_white, dtype=torch.complex64),
                's_map': torch.tensor(s_maps_white, dtype=torch.complex64),
                'mask_2': torch.tensor(mask_2),
                'mask_3': torch.tensor(mask_3),
                'mask_4': torch.tensor(mask_4),
                'mask_5': torch.tensor(mask_5),
                'mask_6': torch.tensor(mask_6),
                'mask_7': torch.tensor(mask_7),
                'mask_8': torch.tensor(mask_8),
                'mask_9': torch.tensor(mask_9),
                'mask_10': torch.tensor(mask_10),
                'norm_consts_99': norm_const_99_white,},
                os.path.join(path, f'sample_{tag}_{i}.pt') if args.contrast == "t1-flair" else os.path.join(path, f'sample_{i}.pt'))
    
    torch.save({"noise_var_noisy": var},
               os.path.join(path, f'noise_var_{tag}_{i}.pt') if args.contrast == "t1-flair" else os.path.join(path, f'noise_var_{i}.pt'))
