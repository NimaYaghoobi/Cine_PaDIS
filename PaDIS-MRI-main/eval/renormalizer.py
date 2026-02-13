import numpy as np
import torch
import os
import re
import json
import argparse
from pathlib import Path
from tqdm import tqdm

import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import sigpy as sp

from inverse_operators import *

def fftmod(x):
    """K-space centering modulation, an alternative to fftshift."""
    x[..., ::2, :] *= -1
    x[..., :, ::2] *= -1
    return x

def normalization_const_from_ksp(fs_ksp):
    """Calculates the normalization constant directly from fully-sampled k-space."""
    # print(f"Shape fs ksp: {fs_ksp.shape}")
    ksp_numpy = fs_ksp.squeeze(0).cpu().numpy()
    # print(f"after 1 sqeeze: {ksp_numpy.shape}")
    ACS_size = 20  
    num_coils, H, W = ksp_numpy.shape
    
    ksp_acs_only = sp.resize(sp.resize(ksp_numpy, (num_coils, ACS_size, ACS_size)), ksp_numpy.shape)
    ACS_img = sp.rss(sp.ifft(ksp_acs_only, axes=(-2, -1)), axes=(0,))
    norm_const_99 = np.percentile(np.abs(ACS_img), 99)
    # print(f"norm const: {norm_const_99}")
    return norm_const_99

def nrmse(gt_img, recon_img):
    """Normalized Root Mean Squared Error."""
    return np.linalg.norm(gt_img - recon_img) / np.linalg.norm(gt_img)

def psnr(gt_img, recon_img, data_range):
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((gt_img - recon_img) ** 2)
    return 20 * np.log10(data_range / np.sqrt(mse)) if mse > 0 else float('inf')

def save_individual_image(image_data, save_path, title, vmax):
    """Save a single image."""
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    ax.imshow(image_data, cmap='gray', vmax=vmax)
    ax.set_title(title)
    ax.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=150)
    plt.close(fig)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    recon_dir = Path(args.recon_dir)
    val_dir = Path(args.val_dir)
    
    if args.plot_dir is None:
        plot_dir = recon_dir.parent / "new_plots"
    else:
        plot_dir = Path(args.plot_dir)
    
    plot_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving plots to: {plot_dir}")
    
    if args.adjoint:
        adjoint_dir = plot_dir / "adjoint"
        adjoint_dir.mkdir(parents=True, exist_ok=True)
    if args.recon:
        recon_save_dir = plot_dir / "recon"
        recon_save_dir.mkdir(parents=True, exist_ok=True)
    if args.ground_truth:
        gt_dir = plot_dir / "ground_truth"
        gt_dir.mkdir(parents=True, exist_ok=True)
    
    json_output_path = plot_dir / args.json_filename
    
    patterns = [
        f"recon_admm_mask{args.mask_select}_*.npy",  
        "recon_admm_mask*_*.npy",                    
        "recon_admm_*.npy",
        "recon_patch_*.npy",
        "recon_whole_*.npy",
        "recon_*.npy",
    ]
    recon_files = []
    pattern_used = None
    for pattern in patterns:
        files = sorted(recon_dir.glob(pattern))
        if files:
            recon_files = files
            pattern_used = pattern
            print(f"Found {len(recon_files)} files matching pattern '{pattern}'")
            break

    if not recon_files:
        print(f"Error: No reconstruction files found in {recon_dir}")
        print(f"Tried patterns: {patterns}")
        return

    results = {"per_image": {}}
    if "admm_mask" in pattern_used:
        regex_pattern = r'^recon_admm_mask(?P<mask>\d+)_(?P<idx>\d+)\.npy$'
    elif "admm" in pattern_used:
        regex_pattern = r'^recon_admm_(?P<idx>\d+)\.npy$'
    elif "patch" in pattern_used:
        regex_pattern = r'^recon_patch_(?P<idx>\d+)\.npy$'
    elif "whole" in pattern_used:
        regex_pattern = r'^recon_whole_(?P<idx>\d+)\.npy$'
    else:
        regex_pattern = r'^recon_(?P<idx>\d+)\.npy$'
    
    for recon_file in tqdm(recon_files, desc="Evaluating Reconstructions"):
        m = re.fullmatch(regex_pattern, recon_file.name)
        if not m:
            print(f"Warning: Skipping file with unexpected name format: {recon_file.name}")
            continue

        if "mask" in m.groupdict():
            mask_in_name = int(m.group("mask"))
            if mask_in_name != args.mask_select:
                continue

        sample_idx = int(m.group("idx"))

        
        try:
            recon = torch.from_numpy(np.load(recon_file)[None, ...]).to(device)
            
            gt_data_path = val_dir / f"sample_{sample_idx}.pt"
            data = torch.load(gt_data_path)
            
            gt = data['gt'][None, None, ...].to(device)
            s_maps = data['s_map'][None, ...].to(device)
            mask_str = f"mask_{args.mask_select}" 
            mask = data[mask_str][None, ...].to(device)
            fs_ksp = fftmod(data['ksp'])[None, ...].to(device)
            
        except FileNotFoundError:
            print(f"Warning: Ground truth file not found for sample {sample_idx}. Skipping.")
            continue
        except KeyError:
            print(f"Warning: Key '{mask_str}' not found in sample {sample_idx}. Skipping.")
            continue

        norm_val = normalization_const_from_ksp(fs_ksp)
        epsilon = 1e-9
        
        gt_norm = torch.abs(gt) / (norm_val + epsilon)
        recon_norm = torch.abs(recon) / (norm_val + epsilon)
        
        gt_norm_2d = gt_norm.squeeze().cpu().numpy()
        recon_norm_2d = recon_norm.squeeze().cpu().numpy()
        
        data_range_norm = gt_norm_2d.max() - gt_norm_2d.min()
        val_nrmse = nrmse(gt_norm_2d, recon_norm_2d)
        val_psnr = psnr(gt_norm_2d, recon_norm_2d, data_range=data_range_norm)
        val_ssim = ssim(gt_norm_2d, recon_norm_2d, data_range=data_range_norm)
        
        results["per_image"][str(sample_idx)] = {
            "psnr": float(val_psnr),
            "ssim": float(val_ssim),
            "nrmse": float(val_nrmse)
        }

        us_ksp = fs_ksp * mask
        mri_op = MRI_utils(mask=mask, maps=s_maps)
        adjoint = mri_op.adjoint(us_ksp)
        adjoint_norm_2d = (torch.abs(adjoint) / (norm_val + epsilon)).squeeze().cpu().numpy()

        vmax = np.percentile(gt_norm_2d, 99.5)

        if args.side_by_side:
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))

            axes[0].imshow(adjoint_norm_2d, cmap='gray', vmax=vmax)
            axes[0].set_title('Adjoint (Zero-Filled)')
            axes[0].axis('off')

            axes[1].imshow(recon_norm_2d, cmap='gray', vmax=vmax)
            axes[1].set_title('Reconstruction')
            axes[1].axis('off')

            axes[2].imshow(gt_norm_2d, cmap='gray', vmax=vmax)
            axes[2].set_title('Ground Truth')
            axes[2].axis('off')
            
            fig.suptitle(f'Sample {sample_idx} | NRMSE: {val_nrmse:.4f}, PSNR: {val_psnr:.2f} dB, SSIM: {val_ssim:.4f}', fontsize=16)
            
            plt.tight_layout()
            plt.savefig(plot_dir / f"comparison_sample_{sample_idx}.png")
            plt.close(fig)
        
        # Save individual images if requested
        if args.adjoint:
            save_path = adjoint_dir / f"adjoint_sample_{sample_idx}.png"
            save_individual_image(adjoint_norm_2d, save_path, f'Sample {sample_idx}', vmax)
        
        if args.recon:
            save_path = recon_save_dir / f"recon_sample_{sample_idx}.png"
            save_individual_image(recon_norm_2d, save_path, f'Sample {sample_idx}', vmax)
        
        if args.ground_truth:
            save_path = gt_dir / f"gt_sample_{sample_idx}.png"
            save_individual_image(gt_norm_2d, save_path, f'Sample {sample_idx}', vmax)

    all_psnr = [v['psnr'] for v in results['per_image'].values()]
    all_ssim = [v['ssim'] for v in results['per_image'].values()]
    all_nrmse = [v['nrmse'] for v in results['per_image'].values()]

    if all_psnr:
        results["summary"] = {
            "psnr_mean": float(np.mean(all_psnr)), "psnr_std": float(np.std(all_psnr)),
            "ssim_mean": float(np.mean(all_ssim)), "ssim_std": float(np.std(all_ssim)),
            "nrmse_mean": float(np.mean(all_nrmse)), "nrmse_std": float(np.std(all_nrmse))
        }
    
    with open(json_output_path, 'w') as f: 
        json.dump(results, f, indent=2)

    print(f"\nProcessing complete.")
    print(f"Metrics saved to: {json_output_path}")
    print(f"Plots saved to: {args.plot_dir}")
    if "summary" in results:
        print("\n--- Summary ---")
        print(f"PSNR: {results['summary']['psnr_mean']:.2f} ± {results['summary']['psnr_std']:.2f} dB")
        print(f"SSIM: {results['summary']['ssim_mean']:.4f} ± {results['summary']['ssim_std']:.4f}")
        print(f"NRMSE: {results['summary']['nrmse_mean']:.4f} ± {results['summary']['nrmse_std']:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate MRI reconstructions, calculate metrics, and generate plots.")
    parser.add_argument("--recon-dir", type=str, required=True, help="Path to the directory containing reconstruction .npy files.")
    parser.add_argument("--val-dir", type=str, required=True, help="Path to the ground truth validation data directory.")
    parser.add_argument("--plot-dir", type=str, default=None, help="Directory to save the output comparison plots. If not specified, uses 'new_plots' folder in the parent directory of recon-dir.")
    parser.add_argument("--json-filename", type=str, default="results.json", help="Filename for the output metrics JSON file, saved in the plot directory.")
    
    # New flags for saving options
    parser.add_argument("--side-by-side", action="store_true", help="Save side-by-side comparison plots (adjoint, recon, ground truth in one image)")
    parser.add_argument("--adjoint", action="store_true", help="Save individual adjoint (zero-filled) images in 'adjoint' subfolder")
    parser.add_argument("--recon", action="store_true", help="Save individual reconstruction images in 'recon' subfolder")
    parser.add_argument("--ground-truth", action="store_true", help="Save individual ground truth images in 'ground_truth' subfolder")
    parser.add_argument("--mask-select", type=int, default=7, help="Mask index to use; builds key name 'mask_{int}'. Default: 7")

    args = parser.parse_args()
    
    # Check that at least one saving option is specified
    if not any([args.side_by_side, args.adjoint, args.recon, args.ground_truth]):
        print("Warning: No saving flags specified. Please use at least one of: --side-by-side, --adjoint, --recon, --ground-truth")
        parser.print_help()
        exit(1)
    
    main(args)