import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import h5py
import sigpy as sp
import glob
import random
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm as tqdm_base

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from dnnlib.util import configure_bart
configure_bart()

from bart import bart
import torch
from multiprocessing import Pool

from data_utils import forward_fs, normalization_const, tqdm

parser = argparse.ArgumentParser(description="Process MRI volumes for training set.")

parser.add_argument('--max_volumes', type=int, default=200, help='Maximum number of volumes to process')
parser.add_argument('--num_slices', type=int, default=1, help='Number of slices per volume')
parser.add_argument('--h5_folder', type=str, required=True, help='Path to input folder containing .h5 files')
parser.add_argument('--output_root', type=str, default="/data/datasets/fastmri/", help='Path to output folder to save results')
parser.add_argument('--random_seed', type=int, default=42, help='Seed for consistent sampling')
parser.add_argument('--noise_level', type=str, default="32dB", choices=["32dB", "22dB", "12dB"], help='Noise level to add')
parser.add_argument('--nproc', type=int, default=30, help='Number of CPU cores to use')
parser.add_argument('--acs_size', type=int, default=24, help='Number of ACS lines to use')


args = parser.parse_args()

device           = sp.cpu_device
n_proc           = args.nproc # number of cpu cores to use, when possible
num_slices       = args.num_slices 
center_slice     = 2
ACS_size         = args.acs_size
imsize           = 384

db = args.noise_level

if db == "32dB":
    noise_amp = np.sqrt(0)
elif db == "22dB":
    noise_amp = np.sqrt(10)
elif db == "12dB":
    noise_amp = np.sqrt(100)
else:
    raise ValueError(f"Unsupported db level: {db}")
    
h5_folder = args.h5_folder
ksp_files = [os.path.join(h5_folder, f) for f in os.listdir(h5_folder) if f.endswith(".h5")]
if not ksp_files:
    raise FileNotFoundError(f"No .h5 files found in {h5_folder}")
print(f"Found {len(ksp_files)} .h5 files in {h5_folder}")

max_volumes = args.max_volumes if args.max_volumes < len(ksp_files) else len(ksp_files)
total_iterations = max_volumes * num_slices

all_possible = list(range(len(ksp_files) * num_slices))
rng = np.random.default_rng(seed=args.random_seed)
indexes = rng.choice(all_possible, size=total_iterations, replace=False).tolist()


x_est_gt = torch.zeros(total_iterations, imsize, imsize, dtype=torch.complex64)
x_est = torch.zeros(total_iterations, imsize, imsize, dtype=torch.complex64)
u_images = torch.zeros(total_iterations, imsize, imsize, dtype=torch.complex64)
norm_consts_99 = torch.zeros(total_iterations, dtype=torch.float32)
noise_var_noisy = torch.zeros(total_iterations, dtype=torch.float32)

# path = "/data/datasets/fastmri/brain_train_d384_s200/"
path = args.output_root + f"brain_train_d{imsize}_s{max_volumes*num_slices}" + f"/{db}"
if not os.path.exists(path + "/ksp/"):
    os.makedirs(path + "/ksp/")
    
def task(i):
    idx = indexes[i]
    sample_idx = idx // num_slices
    slice_idx  = center_slice + np.mod(idx, num_slices) - num_slices // 2

    # Load MRI samples and maps
    with h5py.File(ksp_files[sample_idx], 'r') as contents:
        # Get k-space for specific slice
        ksp = np.asarray(contents['kspace'][slice_idx]).transpose(1, 2, 0)
        cimg = bart(1, 'fft -iu 3', ksp) # compare to `bart fft -iu 3 ksp cimg`
        cimg = sp.resize(cimg, [396, cimg.shape[1], cimg.shape[2]])

    noise = cimg[0:30,0:30]
    noise_flat = np.reshape(noise, (-1, cimg.shape[2]))
    
    ##
    print(f"cimg shape: {cimg.shape}")
    print(f"noise_flat shape: {noise_flat.shape}")
    ##
    cimg_white = sp.resize(bart(1, 'whiten', cimg[:,:,None,:], noise_flat[:,None,None,:]).squeeze(), [imsize, imsize, cimg.shape[2]])
    cimg_white_noisy = cimg_white + (noise_amp / np.sqrt(2))*(np.random.normal(size=cimg_white.shape) + 1j * np.random.normal(size=cimg_white.shape))
    
    ksp_white = bart(1, 'fft -u 3', cimg_white)
    ksp_white_noisy = bart(1, 'fft -u 3', cimg_white_noisy)
    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white[:,:,None,:]).squeeze()
    s_maps_white_noisy = bart(1, 'ecalib -m 1 -c0', ksp_white_noisy[:,:,None,:]).squeeze()
    
    gt_img_white_cropped = bart(1, 'pics -S -i 30', ksp_white[:,:,None,:], s_maps_white[:,:,None,:])
    gt_img_white_cropped_noisy = bart(1, 'pics -S -i 30', ksp_white_noisy[:,:,None,:], s_maps_white_noisy[:,:,None,:])

    ksp_white = ksp_white.transpose(2, 0, 1)
    ksp_white_noisy = ksp_white_noisy.transpose(2, 0, 1)
    s_maps_white = s_maps_white.transpose(2, 0, 1)  
    s_maps_white_noisy = s_maps_white_noisy.transpose(2, 0, 1)  
    cimg_white = cimg_white.transpose(2, 0, 1)  
    cimg_white_noisy = cimg_white_noisy.transpose(2, 0, 1)  

    norm_const_99_white = normalization_const(s_maps_white, gt_img_white_cropped, ACS_size=ACS_size)
    norm_const_99_white_noisy = normalization_const(s_maps_white_noisy, gt_img_white_cropped_noisy, ACS_size=ACS_size)
    ksp_white = ksp_white / norm_const_99_white
    ksp_white_noisy = ksp_white_noisy / norm_const_99_white_noisy
    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white.transpose(1, 2, 0)[:,:,None,:]).squeeze().transpose(2, 0, 1)
    s_maps_white_noisy = bart(1, 'ecalib -m 1 -c0', ksp_white_noisy.transpose(1, 2, 0)[:,:,None,:]).squeeze().transpose(2, 0, 1)

    gt_img_white_cropped = bart(1, 'pics -S -i 30', ksp_white.transpose(1, 2, 0)[:,:,None,:], s_maps_white.transpose(1, 2, 0)[:,:,None,:])
    gt_img_white_cropped_noisy=bart(1, 'pics -S -i 30',ksp_white_noisy.transpose(1,2,0)[:,:,None,:],s_maps_white_noisy.transpose(1,2,0)[:,:,None,:])

    cimg_white = bart(1, 'fft -iu 3', ksp_white.transpose(1, 2, 0)).transpose(2, 0, 1) # compare to `bart fft -iu 3 ksp cimg`
    cimg_white_noisy = bart(1, 'fft -iu 3', ksp_white_noisy.transpose(1, 2, 0)).transpose(2, 0, 1) # compare to `bart fft -iu 3 ksp cimg`
    
    var = np.var(cimg_white[:, 0:30, 0:30])
    var_noisy = np.var(cimg_white_noisy[:, 0:30, 0:30])
    
    coil_imgs_with_maps_white_noisy = cimg_white_noisy * np.conj(s_maps_white_noisy)
    u_white_noisy = np.sum(coil_imgs_with_maps_white_noisy, axis = -3)
    u_cropped_white_noisy = u_white_noisy

    print('\n')
    print('white SNR: ' + str(10*np.log10(1/var)))
    print('white_noisy SNR: ' + str(10*np.log10(1/var_noisy)))
    print('gt norm: ' + str(np.linalg.norm(gt_img_white_cropped_noisy)))
    print('u norm: ' + str(np.linalg.norm(u_cropped_white_noisy)))
    
    return i, gt_img_white_cropped, gt_img_white_cropped_noisy, u_cropped_white_noisy, norm_const_99_white_noisy, var_noisy, ksp_white_noisy, s_maps_white_noisy

with Pool(n_proc) as p:
    for i, gt_img_white_cropped, gt_img_white_cropped_noisy, u_cropped_white_noisy, norm_const_99, var_noisy, ksp_white_noisy, s_maps_white_noisy in tqdm(p.imap(task, range(total_iterations))):
        x_est_gt[i] = torch.tensor(gt_img_white_cropped, dtype=torch.complex64)
        x_est[i] = torch.tensor(gt_img_white_cropped_noisy, dtype=torch.complex64)
        u_images[i] = torch.tensor(u_cropped_white_noisy, dtype=torch.complex64)
        norm_consts_99[i] = torch.tensor(norm_const_99, dtype=torch.float32)
        noise_var_noisy[i] = torch.tensor(var_noisy, dtype=torch.float32)
        ksp_white_noisy = torch.tensor(ksp_white_noisy, dtype=torch.complex64)
        s_maps_white_noisy = torch.tensor(s_maps_white_noisy, dtype=torch.complex64)
        
        torch.save({
            "ksp_white_noisy": ksp_white_noisy,
            "s_maps_white_noisy": s_maps_white_noisy},
            path + "/ksp/" + str(i) + ".pt")
        print('Step ' + str(i) + ' Done')

torch.save({'x_est_gt': x_est_gt,
            'x_est': x_est,
            'u_images': u_images,
            'norm_consts_99': norm_consts_99,
            'noise_var_noisy': noise_var_noisy},
            path + "/noisy.pt")