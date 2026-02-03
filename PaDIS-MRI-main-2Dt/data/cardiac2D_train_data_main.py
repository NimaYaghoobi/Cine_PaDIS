import sys
import os
os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np
import h5py
import sigpy as sp
import glob
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
parser.add_argument('--h5_folder', type=str, required=True,
                    help='(Cardiac) Path to FullSample root containing P*/cine_sax.mat')
parser.add_argument('--output_root', type=str, default="/data/datasets/fastmri/", help='Path to output folder to save results')
parser.add_argument('--random_seed', type=int, default=42, help='Seed for consistent sampling')
parser.add_argument('--noise_level', type=str, default="32dB", choices=["32dB", "22dB", "12dB"], help='Noise level to add')
parser.add_argument('--nproc', type=int, default=30, help='Number of CPU cores to use')
parser.add_argument('--acs_size', type=int, default=24, help='Number of ACS lines to use')

# time is FIRST dim: (time, slice, coil, ky, kx)
parser.add_argument('--time_index', type=int, default=-1,
                    help='Which time frame to use. -1 means middle frame.')

args = parser.parse_args()

device           = sp.cpu_device
n_proc           = args.nproc
num_slices       = args.num_slices
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

# ---------- NEW: filter only files whose kspace_full has ky,kx = (246,512) ----------
TARGET_KY, TARGET_KX = 246, 512   # change these if you truly meant (346,512)

def _has_target_shape(mat_path):
    try:
        with h5py.File(mat_path, "r") as f:
            if "kspace_full" not in f:
                return False
            d = f["kspace_full"]
            ky, kx = int(d.shape[-2]), int(d.shape[-1])
            return (ky == TARGET_KY) and (kx == TARGET_KX)
    except Exception:
        return False

ksp_files_all = sorted(glob.glob(os.path.join(h5_folder, "P*/cine_sax.mat")))
ksp_files = [p for p in ksp_files_all if _has_target_shape(p)]

if not ksp_files:
    raise FileNotFoundError(f"No cine_sax.mat with ky,kx={(TARGET_KY, TARGET_KX)} found under {h5_folder}/P*/cine_sax.mat")

print(f"Found {len(ksp_files_all)} cine_sax.mat total under {h5_folder}")
print(f"Keeping {len(ksp_files)} files with ky,kx={(TARGET_KY, TARGET_KX)}")

# ---------- helper: center-crop + zero-pad to (384,384), return pad offsets ----------
def crop_pad_center_to_square(x, target):
    # x: (H, W, C)
    H, W, C = x.shape

    # center crop to at most target
    crop_h = min(H, target)
    crop_w = min(W, target)
    y0 = (H - crop_h) // 2
    x0 = (W - crop_w) // 2
    x_crop = x[y0:y0+crop_h, x0:x0+crop_w, :]

    # pad to target
    pad_top = (target - crop_h) // 2
    pad_bot = target - crop_h - pad_top
    pad_left = (target - crop_w) // 2
    pad_right = target - crop_w - pad_left

    x_out = np.pad(
        x_crop,
        ((pad_top, pad_bot), (pad_left, pad_right), (0, 0)),
        mode="constant"
    )
    return x_out, pad_top, pad_left

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

path = args.output_root + f"cardiac_train_d{imsize}_s{max_volumes*num_slices}" + f"/{db}"
if not os.path.exists(path + "/ksp/"):
    os.makedirs(path + "/ksp/")

def task(i):
    idx = indexes[i]
    sample_idx = idx // num_slices
    slice_offset = np.mod(idx, num_slices) - num_slices // 2
    mat_path = ksp_files[sample_idx]

    with h5py.File(mat_path, 'r') as contents:
        dset = contents['kspace_full']  # (time, slice, coil, ky, kx)
        T, S, C, Ky, Kx = dset.shape

        # pick time
        if args.time_index < 0:
            time_idx = T // 2
        else:
            time_idx = int(np.clip(args.time_index, 0, T - 1))

        # pick slice (center slice + offset)
        center_slice_local = S // 2
        slice_idx = int(np.clip(center_slice_local + slice_offset, 0, S - 1))

        ksp_struct = np.asarray(dset[time_idx, slice_idx, :, :, :])  # (coil, ky, kx)

        # structured complex -> complex
        if (ksp_struct.dtype.fields is not None) and ("real" in ksp_struct.dtype.fields) and ("imag" in ksp_struct.dtype.fields):
            ksp_cplx = ksp_struct["real"] + 1j * ksp_struct["imag"]
        else:
            ksp_cplx = ksp_struct.astype(np.complex64)

        # BART expects (ky, kx, coils)
        ksp = ksp_cplx.transpose(1, 2, 0)

        # coil images (ky, kx, coils)
        cimg0 = bart(1, 'fft -iu 3', ksp)

    # -------- crop+pad to 384x384 (NO RESIZE/RESAMPLE) --------
    cimg, pad_top, pad_left = crop_pad_center_to_square(cimg0, imsize)

    # noise patch picked from the NON-PADDED region if possible
    yN = min(30, imsize - pad_top)
    xN = min(30, imsize - pad_left)
    noise = cimg[pad_top:pad_top+yN, pad_left:pad_left+xN, :]
    noise_flat = np.reshape(noise, (-1, cimg.shape[2]))

    # decide if whitening is possible BEFORE calling bart whiten
    percoil_var = np.var(noise_flat, axis=0)
    can_whiten = np.all(np.isfinite(percoil_var)) and np.any(percoil_var > 0)

    if not can_whiten:
        # skip whitening safely (no LAPACK spam)
        cimg_white = cimg
        whiten_status = "SKIP(noise var ~0)"
    else:
        tmp = bart(1, 'whiten', cimg[:, :, None, :], noise_flat[:, None, None, :])
        if tmp is None:
            cimg_white = cimg
            whiten_status = "SKIP(whiten failed)"
        else:
            cimg_white = tmp.squeeze()
            whiten_status = "OK"

    print(f"[{i}] {os.path.basename(os.path.dirname(mat_path))} t={time_idx} s={slice_idx} "
          f"cimg={cimg.shape} noise={noise_flat.shape} whiten={whiten_status}")

    # add synthetic noise (same as original logic)
    cimg_white_noisy = cimg_white + (noise_amp / np.sqrt(2)) * (
        np.random.normal(size=cimg_white.shape) + 1j * np.random.normal(size=cimg_white.shape)
    )

    ksp_white = bart(1, 'fft -u 3', cimg_white)
    ksp_white_noisy = bart(1, 'fft -u 3', cimg_white_noisy)
    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white[:, :, None, :]).squeeze()
    s_maps_white_noisy = bart(1, 'ecalib -m 1 -c0', ksp_white_noisy[:, :, None, :]).squeeze()

    gt_img_white_cropped = bart(1, 'pics -S -i 30', ksp_white[:, :, None, :], s_maps_white[:, :, None, :])
    gt_img_white_cropped_noisy = bart(1, 'pics -S -i 30', ksp_white_noisy[:, :, None, :], s_maps_white_noisy[:, :, None, :])

    ksp_white = ksp_white.transpose(2, 0, 1)
    ksp_white_noisy = ksp_white_noisy.transpose(2, 0, 1)
    s_maps_white = s_maps_white.transpose(2, 0, 1)
    s_maps_white_noisy = s_maps_white_noisy.transpose(2, 0, 1)
    cimg_white = cimg_white.transpose(2, 0, 1)              # (coil, H, W)
    cimg_white_noisy = cimg_white_noisy.transpose(2, 0, 1)  # (coil, H, W)

    norm_const_99_white = normalization_const(s_maps_white, gt_img_white_cropped, ACS_size=ACS_size)
    norm_const_99_white_noisy = normalization_const(s_maps_white_noisy, gt_img_white_cropped_noisy, ACS_size=ACS_size)
    ksp_white = ksp_white / norm_const_99_white
    ksp_white_noisy = ksp_white_noisy / norm_const_99_white_noisy

    s_maps_white = bart(1, 'ecalib -m 1 -c0', ksp_white.transpose(1, 2, 0)[:, :, None, :]).squeeze().transpose(2, 0, 1)
    s_maps_white_noisy = bart(1, 'ecalib -m 1 -c0', ksp_white_noisy.transpose(1, 2, 0)[:, :, None, :]).squeeze().transpose(2, 0, 1)

    gt_img_white_cropped = bart(1, 'pics -S -i 30', ksp_white.transpose(1, 2, 0)[:, :, None, :], s_maps_white.transpose(1, 2, 0)[:, :, None, :])
    gt_img_white_cropped_noisy = bart(1, 'pics -S -i 30', ksp_white_noisy.transpose(1, 2, 0)[:, :, None, :], s_maps_white_noisy.transpose(1, 2, 0)[:, :, None, :])

    cimg_white = bart(1, 'fft -iu 3', ksp_white.transpose(1, 2, 0)).transpose(2, 0, 1)
    cimg_white_noisy = bart(1, 'fft -iu 3', ksp_white_noisy.transpose(1, 2, 0)).transpose(2, 0, 1)

    # compute var from the same non-padded corner (avoid padded zeros)
    var = np.var(cimg_white[:, pad_top:pad_top+30, pad_left:pad_left+30])
    var_noisy = np.var(cimg_white_noisy[:, pad_top:pad_top+30, pad_left:pad_left+30])

    coil_imgs_with_maps_white_noisy = cimg_white_noisy * np.conj(s_maps_white_noisy)
    u_white_noisy = np.sum(coil_imgs_with_maps_white_noisy, axis=-3)
    u_cropped_white_noisy = u_white_noisy

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
            "s_maps_white_noisy": s_maps_white_noisy
        }, path + "/ksp/" + str(i) + ".pt")

        print('Step ' + str(i) + ' Done')

torch.save({
    'x_est_gt': x_est_gt,
    'x_est': x_est,
    'u_images': u_images,
    'norm_consts_99': norm_consts_99,
    'noise_var_noisy': noise_var_noisy
}, path + "/noisy.pt")
