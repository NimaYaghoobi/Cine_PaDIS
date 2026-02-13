import os
import re
import json
import numpy as np
import torch
import random
import sigpy as sp
from pathlib import Path
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict, Union

from inverse_operators import *

torch.manual_seed(123)
np.random.seed(123)

def fftmod(x: torch.Tensor) -> torch.Tensor:
    x[...,::2,:] *= -1
    x[...,:,::2] *= -1
    return x

def _to_mag_np(x):
    if isinstance(x, np.ndarray):
        arr = x
    else:
        x = x.detach().cpu()
        arr = x.numpy()
    arr = np.squeeze(arr)
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    else:
        arr = np.abs(arr)
    if arr.ndim > 2:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        arr = arr[..., -2:, :].squeeze()
    return arr.astype(np.float64, copy=False)

def _nrmse_np(gt: np.ndarray, x: np.ndarray) -> float:
    return np.linalg.norm(gt - x) / np.linalg.norm(gt)

def _psnr_np(gt: np.ndarray, x: np.ndarray, data_range: float) -> float:
    mse = np.mean((gt - x) ** 2)
    return 20 * np.log10(data_range / np.sqrt(mse)) if mse > 0 else float('inf')

def makeFigures(noisy2, 
                denoised2, 
                orig2, 
                i, 
                out_dir='.', 
                tag="recon", 
                imsize=256, 
                plot=True,
                vmax_percentile=99.5):
    
    gt_np = _to_mag_np(orig2)
    recon_np = _to_mag_np(noisy2) 
    denoise_np = _to_mag_np(denoised2)
    
    epsilon = 1e-9
    norm_val = 1.0 
    gt_np = gt_np / (norm_val + epsilon)
    recon_np = recon_np / (norm_val + epsilon)
    denoise_np = denoise_np / (norm_val + epsilon)
    
    data_range = gt_np.max() - gt_np.min()
    if data_range <= 0: data_range = float(gt_np.max() + epsilon)

    noisypsnr = _psnr_np(gt_np, recon_np, data_range=data_range)
    denoisedpsnr = _psnr_np(gt_np, denoise_np, data_range=data_range)

    noisyssim = ssim(gt_np, recon_np, data_range=data_range)
    denoisedssim = ssim(gt_np, denoise_np, data_range=data_range)
    
    noisynrmse = _nrmse_np(gt_np, recon_np)
    denoisednrmse = _nrmse_np(gt_np, denoise_np)

    if plot:
        out_dir_path = Path(out_dir)
        out_dir_path.mkdir(parents=True, exist_ok=True)

        vmax = np.percentile(gt_np, vmax_percentile)
        vmin = 0.0 

        plt.figure(figsize=(12, 6))
        plt.subplot(1, 3, 1)
        plt.imshow(recon_np, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title(f"Recon\nPSNR: {noisypsnr:.2f} dB")

        plt.subplot(1, 3, 2)
        plt.imshow(denoise_np, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title(f"Diffusion Recon\nPSNR: {denoisedpsnr:.2f} dB")

        plt.subplot(1, 3, 3)
        plt.imshow(gt_np, cmap='gray', vmin=vmin, vmax=vmax)
        plt.axis('off')
        plt.title("Ground Truth")

        plt.tight_layout()
        save_path = out_dir_path / f"{i}_{tag}.png"
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        plt.close('all')

    return noisypsnr, denoisedpsnr, noisyssim, denoisedssim, noisynrmse, denoisednrmse



def post_eval_normalize(
    recon_dir: str,
    val_dir: str,
    plot_dir: str,
    json_basename: str = "results",
    mask_select: Optional[int] = None,
) -> Dict:
    """
    Post-process reconstructions: normalize by FS k-space ACS-derived constant,
    compute PSNR/SSIM/NRMSE vs GT, and save side-by-side plots + a results JSON.

    Args:
        recon_dir: path to npy recon files (e.g., save_dir/evaluate/recons/)
        val_dir:   path to .pt validation samples (same as --val_dir)
        plot_dir:  where to save PNGs (e.g., save_dir/new_plots/)
        json_basename: basename for the output json file ('.json' will be appended if missing)
        mask_select: restrict files by mask id when pattern contains 'mask{mask}_idx.npy'

    Returns:
        results dict with per_image metrics and summary stats.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    recon_dir = Path(recon_dir)
    val_dir = Path(val_dir)
    plot_dir = Path(plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    json_path = plot_dir / (json_basename if json_basename.endswith(".json") else f"{json_basename}.json")

    patterns = [
        "recon_admm_*.npy",
        "recon_patch_*.npy",
        "recon_whole_*.npy",
        "recon_*.npy",
    ]
    recon_files: List[Path] = []
    pattern_used = None
    for pat in patterns:
        files = sorted(recon_dir.glob(pat))
        if files:
            recon_files = files
            pattern_used = pat
            print(f"[post_eval_normalize] Found {len(recon_files)} files via '{pat}'")
            break
    if not recon_files:
        print(f"[post_eval_normalize] No reconstruction files in {recon_dir}. Tried: {patterns}")
        return {"per_image": {}, "summary": {}}

    if "admm_mask" in (pattern_used or ""):
        regex = r'^recon_admm_mask(?P<mask>\d+)_(?P<idx>\d+)\.npy$'
    elif "admm" in (pattern_used or ""):
        regex = r'^recon_admm_(?P<idx>\d+)\.npy$'
    elif "patch" in (pattern_used or ""):
        regex = r'^recon_patch_(?P<idx>\d+)\.npy$'
    elif "whole" in (pattern_used or ""):
        regex = r'^recon_whole_(?P<idx>\d+)\.npy$'
    else:
        regex = r'^recon_(?P<idx>\d+)\.npy$'

    def normalization_const_from_ksp(fs_ksp: torch.Tensor) -> float:
        """Calculate normalization constant from fully-sampled k-space via ACS."""
        ksp_numpy = fs_ksp.squeeze(0).cpu().numpy()  # [Nc,H,W]
        ACS_size = 24
        num_coils, H, W = ksp_numpy.shape
        ksp_acs_only = sp.resize(sp.resize(ksp_numpy, (num_coils, ACS_size, ACS_size)), ksp_numpy.shape)
        ACS_img = sp.rss(sp.ifft(ksp_acs_only, axes=(-2, -1)), axes=(0,))
        return float(np.percentile(np.abs(ACS_img), 99))

    results: Dict = {"per_image": {}}
    epsilon = 1e-9

    for rf in recon_files:
        m = re.fullmatch(regex, rf.name)
        if not m:
            print(f"[post_eval_normalize] Skip unexpected filename: {rf.name}")
            continue

        if "mask" in m.groupdict():
            file_mask = int(m.group("mask"))
            if mask_select is not None and file_mask != mask_select:
                continue

        sample_idx = int(m.group("idx"))

        try:
            recon = torch.from_numpy(np.load(rf)[None, ...]).to(device)  # [1,H,W]
            sample_pt = val_dir / f"sample_{sample_idx}.pt"
            data = torch.load(sample_pt, map_location=device)

            gt = data['gt'][None, None, ...].to(device)              # [1,1,H,W] complex
            s_maps = data['s_map'][None, ...].to(device)
            fs_ksp = fftmod(data['ksp'])[None, ...].to(device)
            key_mask = f"mask_{mask_select}" if mask_select is not None else "mask_7"
            mask = data[key_mask][None, ...].to(device)

        except FileNotFoundError:
            print(f"[post_eval_normalize] Missing GT sample for idx {sample_idx}; skipping.")
            continue
        except KeyError as e:
            print(f"[post_eval_normalize] Missing key {e} for idx {sample_idx}; skipping.")
            continue

        norm_val = normalization_const_from_ksp(fs_ksp)
        gt_norm = torch.abs(gt) / (norm_val + epsilon)
        recon_norm = torch.abs(recon) / (norm_val + epsilon)

        gt2 = gt_norm.squeeze().detach().cpu().numpy()
        rc2 = recon_norm.squeeze().detach().cpu().numpy()
        data_range = float(gt2.max() - gt2.min()) if gt2.size > 0 else 1.0

        # Metrics
        m_nrmse = _nrmse_np(gt2, rc2)
        m_psnr = _psnr_np(gt2, rc2, data_range=data_range)
        from skimage.metrics import structural_similarity as _ssim
        m_ssim = float(_ssim(gt2, rc2, data_range=data_range))

        results["per_image"][str(sample_idx)] = {"psnr": m_psnr, "ssim": m_ssim, "nrmse": m_nrmse}

        us_ksp = fs_ksp * mask
        mri_op = MRI_utils(mask=mask, maps=s_maps)
        adjoint = mri_op.adjoint(us_ksp)
        adj2 = (torch.abs(adjoint) / (norm_val + epsilon)).squeeze().detach().cpu().numpy()

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        vmax = np.percentile(gt2, 99.5) if np.isfinite(gt2).all() else gt2.max()

        axes[0].imshow(adj2, cmap='gray', vmax=vmax);  axes[0].set_title('Adjoint (Zero-Filled)'); axes[0].axis('off')
        axes[1].imshow(rc2,  cmap='gray', vmax=vmax);  axes[1].set_title('Reconstruction');        axes[1].axis('off')
        axes[2].imshow(gt2,  cmap='gray', vmax=vmax);  axes[2].set_title('Ground Truth');          axes[2].axis('off')

        fig.suptitle(
            f"Sample {sample_idx} | NRMSE: {m_nrmse:.4f}, PSNR: {m_psnr:.2f} dB, SSIM: {m_ssim:.4f}",
            fontsize=16
        )
        plt.tight_layout()
        fig.savefig(plot_dir / f"comparison_sample_{sample_idx}.png")
        plt.close(fig)

    all_psnr = [v['psnr'] for v in results['per_image'].values()]
    all_ssim = [v['ssim'] for v in results['per_image'].values()]
    all_nrmse = [v['nrmse'] for v in results['per_image'].values()]
    if all_psnr:
        results["summary"] = {
            "psnr_mean": float(np.mean(all_psnr)), "psnr_std": float(np.std(all_psnr)),
            "ssim_mean": float(np.mean(all_ssim)), "ssim_std": float(np.std(all_ssim)),
            "nrmse_mean": float(np.mean(all_nrmse)), "nrmse_std": float(np.std(all_nrmse)),
        }

    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[post_eval_normalize] Metrics → {json_path}")
    print(f"[post_eval_normalize] Plots   → {plot_dir}")

    if "summary" in results:
        s = results["summary"]
        print(f"[post_eval_normalize] PSNR  {s['psnr_mean']:.2f} ± {s['psnr_std']:.2f} dB")
        print(f"[post_eval_normalize] SSIM  {s['ssim_mean']:.4f} ± {s['ssim_std']:.4f}")
        print(f"[post_eval_normalize] NRMSE {s['nrmse_mean']:.4f} ± {s['nrmse_std']:.4f}")

    return results
