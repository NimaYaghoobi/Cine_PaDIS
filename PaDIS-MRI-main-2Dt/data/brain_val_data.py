import sys
import os
import numpy as np
import h5py
import sigpy as sp
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

parser = argparse.ArgumentParser(description="Generate noisy MRI samples with multiple masks per R-level")
parser.add_argument('--noise_level', type=str, default="32dB", choices=["32dB", "22dB", "12dB"], help='Noise level to add')
parser.add_argument('--acs_size', type=int, default=20, help='Number of ACS lines to use')
parser.add_argument('--h5_folder', type=str, default="/data/datasets/fastmri/multicoil_val", help='Path to input folder containing .h5 files')
parser.add_argument('--output_root', type=str, default="/data/datasets/fastmri/", help='Path to output folder to save results')
parser.add_argument('--contrast', type=str, default="t1-flair", choices=["t1-flair", "t2"], help='Contrast to filter for')

args = parser.parse_args()

center_slice     = 2
ACS_size         = args.acs_size
imsize           = 384

snr = args.noise_level
if snr == "32dB":
    noise_amp = np.sqrt(0)
elif snr == "22dB":
    noise_amp = np.sqrt(10)
elif snr == "12dB":
    noise_amp = np.sqrt(100)

ksp_files_train = sorted(glob.glob(args.h5_folder + "/**.h5"))
ksp_files = []

for files in ksp_files_train:
    if args.contrast == "t1-flair":
        if 'AXT1' in files or 'FLAIR' in files:
            ksp_files.append(files)
    elif args.contrast == "t2":
        if 'AXT2' in files:
            ksp_files.append(files)

print(f"Processing {len(ksp_files)} files for contrast {args.contrast}")

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
        # Get k-space for specific slice
        ksp = np.asarray(contents['kspace'][slice_idx]).transpose(1, 2, 0)

    cimg = bart(1, 'fft -iu 3', ksp) # compare to `bart fft -iu 3 ksp cimg`
    noise = sp.resize(cimg, [396, cimg.shape[1], cimg.shape[2]])[0:30,0:30]
    noise_flat = np.reshape(noise, (-1, cimg.shape[2]))
    cimg = sp.resize(cimg, [imsize, imsize, cimg.shape[2]])

    cimg_white = bart(1, 'whiten', cimg[:,:,None,:], noise_flat[:,None,None,:]).squeeze()
    cimg_white = cimg_white + (noise_amp / np.sqrt(2))*(np.random.normal(size=cimg_white.shape) + 1j * np.random.normal(size=cimg_white.shape))
    ksp_white = bart(1, 'fft -u 3', cimg_white)
    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white[:,:,None,:]).squeeze()
    
    gt_img_white_cropped = sp.resize(bart(1, 'pics -S -i 30', ksp_white[:,:,None,:], s_maps_white[:,:,None,:]), [imsize, imsize])

    ksp_white = ksp_white.transpose(2, 0, 1)
    s_maps_white = s_maps_white.transpose(2, 0, 1)  
    cimg_white = cimg_white.transpose(2, 0, 1)  

    norm_const_99_white = normalization_const(s_maps_white, gt_img_white_cropped, ACS_size=ACS_size)
    ksp_white = ksp_white / norm_const_99_white
    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white.transpose(1, 2, 0)[:,:,None,:]).squeeze().transpose(2, 0, 1)

    gt_img_white_cropped = sp.resize(bart(1, 'pics -S -i 30', ksp_white.transpose(1, 2, 0)[:,:,None,:], s_maps_white.transpose(1, 2, 0)[:,:,None,:]), [imsize, imsize])
    cimg_white = bart(1, 'fft -iu 3', ksp_white.transpose(1, 2, 0)).transpose(2, 0, 1) # compare to `bart fft -iu 3 ksp cimg`
    var = np.var(cimg_white[:, 0:30, 0:30])

    total_lines = imsize
    R = 2
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_2 = mask[None]

    total_lines = imsize
    R = 3
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_3 = mask[None]

    total_lines = imsize
    R = 4
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_4 = mask[None]

    total_lines = imsize
    R = 5
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_5 = mask[None]

    total_lines = imsize
    R = 6
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_6 = mask[None]

    total_lines = imsize
    R = 7
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_7 = mask[None]

    total_lines = imsize
    R = 8
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_8 = mask[None]

    total_lines = imsize
    R = 9
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_9 = mask[None]

    total_lines = imsize
    R = 10
    acs_lines = ACS_size
    num_sampled_lines = np.floor(total_lines / R)
    center_line_idx = np.arange((total_lines - acs_lines) // 2,(total_lines + acs_lines) // 2)
    outer_line_idx = np.setdiff1d(np.arange(total_lines), center_line_idx)
    random_line_idx = np.random.choice(outer_line_idx,size=int(num_sampled_lines - acs_lines), replace=False)
    mask = np.zeros((total_lines, total_lines))
    mask[:,center_line_idx] = 1.
    mask[:,random_line_idx] = 1.
    # mask = sp.resize(mask, [384, 384])
    # mask[0:32] = mask[32:64]
    # mask[352:384] = mask[32:64]
    mask_10 = mask[None]

    print('\n')
    print('white SNR: ' + str(10*np.log10(1/var)))
    print('gt norm: ' + str(np.linalg.norm(gt_img_white_cropped)))
    print("Mask R=2: " + str((imsize*imsize)/np.sum(mask_2)))
    print("Mask R=3: " + str((imsize*imsize)/np.sum(mask_3)))
    print("Mask R=4: " + str((imsize*imsize)/np.sum(mask_4)))
    print("Mask R=5: " + str((imsize*imsize)/np.sum(mask_5)))
    print("Mask R=6: " + str((imsize*imsize)/np.sum(mask_6)))
    print("Mask R=7: " + str((imsize*imsize)/np.sum(mask_7)))
    print("Mask R=8: " + str((imsize*imsize)/np.sum(mask_8)))
    print("Mask R=9: " + str((imsize*imsize)/np.sum(mask_9)))
    print("Mask R=10: " + str((imsize*imsize)/np.sum(mask_10)))
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