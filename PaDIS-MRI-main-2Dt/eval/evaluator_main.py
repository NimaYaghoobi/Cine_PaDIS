import os
import re
import sys
import pickle
import random
import glob
import threading
import numpy as np
import torch
import tqdm
import time
import argparse
import json
from pathlib import Path
from PIL import Image
import csv
from skimage.metrics import peak_signal_noise_ratio as psnr, structural_similarity as ssim
from numpy.fft import fftshift
import matplotlib.pyplot as plt
from typing import List, Tuple

padis_path = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, padis_path)
import dnnlib

from dnnlib.util import configure_bart
configure_bart()
from bart import bart

from inverse_operators import *
from utils import fftmod, makeFigures
from recon import dps2, dps_uncond, dps_edm, dps_uncond_edm

random.seed(123)
torch.manual_seed(123)
np.random.seed(123)
torch.set_printoptions(profile="full")


class DPSHyperEvaluator:
    def __init__(self,
                model,
                mask_select: int,
                val_dir: str,
                image_size: int,
                pad: int,
                psize: int,
                val_count: int = 100,
                seed: int = 123)-> None:
        """
        model: trained diffusion model (ema)
        val_dir: directory containing validation .pt files
        image_size: size of GT images (e.g. 384)
        pad, psize: PaDIS-MRI padding & patch sizes
        val_count: number of validation volumes to use
        """
        self.image_size = image_size
        self.pad = pad
        self.psize = psize
        self.mask_select = mask_select
        self.inverseop = InverseOperator(self.image_size)
        
        self.model = model.eval() if model is not None else None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 

        random.seed(seed)
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        if not os.path.isdir(val_dir): raise FileNotFoundError(f'val_dir not found: {val_dir}')
        sample_files = glob.glob(os.path.join(val_dir, "sample_*.pt"))
        max_files = len(sample_files)
        print(f"Found {max_files} validation samples in {val_dir}")

        val_count = max_files if val_count is None else min(val_count, max_files)

        print(f"Using {val_count} validation samples")
        
        all_indices = list(range(max_files))
        self.val_indices = sorted(random.sample(all_indices, val_count))
        print(f"Sample indices: {self.val_indices}")
        self.val_dir = val_dir
        self.R_levels = list(range(2,11))
        
        self._build_latents_and_pos()

    def _load_sample(self, 
                     idx: int, 
                     seed_ind: int=None,
                     mask_ind: int=None)-> Tuple[torch.Tensor, torch.Tensor, MRI_utils]:
        """
        Data loader for data-driven priors (FastMRI-EDM and PaDIS-MRI).
        
        Args:
            idx: sample index
            seed_ind: for uncertainty quantification - which mask seed to use at a given R
            mask_ind: for R mask sweep - which R to use (overrides self.mask_select)
        Returns:
            ksp_und: torch.Tensor [1, Nc, H, W] complex (on CUDA)
            gt: torch.Tensor [1, 1, H, W] complex (on CUDA)
            mri_inf_utils: MRI_utils (for computing adjoint)
        """
        data = torch.load(os.path.join(self.val_dir, f"sample_{idx}.pt"))
        gt = data['gt'][None,None,...].cuda()
        mask_id = self.mask_select if not mask_ind else mask_ind
        s_maps = fftmod(data['s_map'])[None,...].cuda()
        fs_ksp = fftmod(data['ksp'])[None,...].cuda()

        mask_str = f"mask_{mask_id}"
        print(f"mask_select: {self.mask_select}")
        
        if seed_ind is not None:
            masks_tensor = data['masks']
            r_idx = self.R_levels.index(self.mask_select)
            mask2d = masks_tensor[r_idx, seed_ind]
            mask = mask2d.unsqueeze(0).cuda()
        else:
            mask = data[mask_str][None,...].cuda()
            
        ksp = mask * fs_ksp
        mri_inf_utils = MRI_utils(maps=s_maps, mask=mask)
        
        return ksp, gt, mri_inf_utils
    
    def _load_sample_for_bart(self, 
                              idx: int, 
                              seed_ind: int = None, 
                              mask_ind: int = None)-> Tuple[np.ndarray, torch.Tensor, MRI_utils, np.ndarray, np.ndarray, np.ndarray]:
        """
        Data loader for BART. 

        Args:
            idx: sample index
            seed_ind: for uncertainty quantification - which mask seed to use at a given R
            mask_ind: for R mask sweep - which R to use (overrides self.mask_select)

        Returns:
            ksp_und: np.ndarray [H, W, 1, Nc] complex64 (BART layout)
            gt: torch.Tensor [1, 1, H, W] complex (on CUDA)
            mri_inf_utils: MRI_utils (for computing adjoint)
            coil_sens: np.ndarray [H, W, 1, Nc] complex64 (recomputed using BART ecalib to match input requirements)
            ksp_ref_bart: np.ndarray [H, W, 1, Nc] complex64 
            mask_np: np.ndarray [H, W] 
        """
        data = torch.load(os.path.join(self.val_dir, f"sample_{idx}.pt"))
        device = torch.device('cuda')

        gt = data['gt'][None, None, ...].to(device)  # [1,1,H,W] complex (or real→complex upstream)
        fs_ksp = fftmod(data['ksp'])[None, ...].to(device)  # [1, Nc, H, W] complex

        if seed_ind is not None:
            r_idx  = self.R_levels.index(self.mask_select)
            mask2d = data['masks'][r_idx, seed_ind].to(device)     # [H, W]
            mask_t = mask2d[None, None, ...]                       # [1,1,H,W]
        else:
            chosen = self.mask_select if mask_ind is None else mask_ind
            mask_t = data[f"mask_{chosen}"][None, ...].to(device)  # [1,1,H,W]
            mask2d = mask_t.squeeze(0).squeeze(0)                  # [H, W]
        ksp_us_torch = mask_t * fs_ksp                             # [1, Nc, H, W]
        ksp_und = (
            ksp_us_torch.squeeze(0).permute(1, 2, 0)               # [H, W, Nc]
            .detach().cpu().numpy().astype(np.complex64)[..., np.newaxis, :]
        )
        ksp_ref_bart = (
            fs_ksp.squeeze(0).permute(1, 2, 0)                     
            .detach().cpu().numpy().astype(np.complex64)[..., np.newaxis, :]
        )
        ksp_ref_bart = bart(1, f'resize -c 0 {ksp_und.shape[0]}', ksp_ref_bart)

        coil_sens, _ = bart(2, 'ecalib -a -m 1', ksp_ref_bart)

        s_maps = data['s_map'][None, ...].to(device)
        mri_inf_utils = MRI_utils(maps=s_maps, mask=mask_t)

        mask_np = mask2d.detach().cpu().numpy()

        return ksp_und, gt, mri_inf_utils, coil_sens, ksp_ref_bart, mask_np

    
    def _build_latents_and_pos(self, pad_override: int=None)-> None:
        """(Re)build the `self.latents` and `self.latents_pos` tensors
           whenever `psize` or `pad` (or image_size) change."""
           
        pad = pad_override if pad_override is not None else self.pad
        self.latents = torch.randn(
            [1, 1, self.image_size, self.image_size],
            device=self.device
        )

        resolution = self.image_size + 2*pad
        x = torch.linspace(-1, 1, resolution, device=self.device)
        y = torch.linspace(-1, 1, resolution, device=self.device)
        x_pos = x.view(1, -1).repeat(resolution, 1)
        y_pos = y.view(-1, 1).repeat(1, resolution)
        pos = torch.stack([x_pos, y_pos], dim=0)  # [2, R, R]
        self.latents_pos = pos.unsqueeze(0)       # [1, 2, R, R]

    
    def dps_edm_wrapper(self, inverse_op: InverseOperator, 
                        measurement: torch.Tensor, 
                        clean: torch.Tensor, 
                        zeta: float, 
                        num_steps: int, 
                        save_dir: str=None, 
                        tag: str=None)-> Tuple[torch.Tensor, float, float, float, float, float, float]:
        """
        Args:
            inverse_op: InverseOperator instance
            measurement: torch.Tensor [1, Nc, H, W] complex (on CUDA) or None for unconditional
            clean: torch.Tensor [1, 1, H, W] complex (on CUDA) or None for unconditional
            zeta: float, data consistency strength
            num_steps: int, number of diffusion steps
            save_dir: str or None, directory to save figures (if provided)
            tag: str or None, tag to append to saved figure filenames (if provided)
        Returns:
            recon: torch.Tensor [1, 1, H, W] complex (on CPU)
            noisypsnr, denoisedpsnr, noisyssim, denoisedssim, noisynrmse, denoisednrmse: floats
        """
        if measurement is None:
            recon = dps_uncond_edm(
                net=self.model,
                batch_size=1,
                resolution=self.image_size,
                num_steps=num_steps,
                sigma_min=0.003,
                sigma_max=10.0,
                rho=7,
                device=self.device,
                randn_like=torch.randn_like,
            )
            
            recon_cpu = recon.cpu()
            mag = torch.abs(recon_cpu.squeeze(0).squeeze(0)).numpy()
            mag_clip = np.clip(mag, 0, 1)
            return mag_clip, 0, 0, 0, 0, 0, 0
        else:
            recon, a, b, c, d, e, f = dps_edm(
                                        net=self.model,
                                        measurement=measurement,
                                        clean=clean,
                                        inverseop=inverse_op,
                                        num_steps=num_steps,
                                        zeta=zeta,
                                        save_dir=save_dir,
                                        tag=tag
                                        )

            return recon, a, b, c, d, e, f
    

    def dps2_wrapper(self, 
                     inverse_op: InverseOperator, 
                     measurement: torch.Tensor, 
                     clean: torch.Tensor, 
                     zeta: float, 
                     pad: int, 
                     psize: int, 
                     num_steps: int, 
                     save_dir: str=None, 
                     tag: str=None)-> Tuple[torch.Tensor, float, float, float, float, float, float]:
        """
        Args:
            inverse_op: InverseOperator instance
            measurement: torch.Tensor [1, Nc, H, W] complex (on CUDA) or None for unconditional
            clean: torch.Tensor [1, 1, H, W] complex (on CUDA) or None for unconditional
            zeta: float, data consistency strength
            pad: int, padding size for PaDIS-MRI
            psize: int, patch size for PaDIS-MRI
            num_steps: int, number of diffusion steps
            save_dir: str or None, directory to save figures (if provided)
            tag: str or None, tag to append to saved figure filenames (if provided)
        Returns:
            recon: torch.Tensor [1, 1, H, W] complex (on CPU)
            noisypsnr, denoisedpsnr, noisyssim, denoisedssim, noisynrmse, denoisednrmse: floats
        """        
        if measurement is None:
            recon = dps_uncond(
                net=self.model,
                batch_size=1,
                resolution=self.image_size,
                psize=psize,
                pad=pad,
                num_steps=num_steps,
                sigma_min=0.003,
                sigma_max=10.0,
                rho=7,
                device=self.device,
                randn_like=torch.randn_like,
            )
            
            recon_cpu = recon.cpu()
            mag = torch.abs(recon_cpu.squeeze(0).squeeze(0)).numpy()
            mag_clip = np.clip(mag, 0, 1)
            return mag_clip, 0, 0, 0, 0, 0, 0
        
        else:    
            recon, a, b, c, d, e, f = dps2(
                                        net=self.model,
                                        latents=self.latents,
                                        latents_pos=self.latents_pos,
                                        inverseop=inverse_op,
                                        measurement=measurement,
                                        clean=clean,
                                        pad=pad,
                                        psize=psize,
                                        zeta=zeta,
                                        num_steps=num_steps,
                                        save_dir=save_dir,
                                        tag=tag
                                    )
            return recon, a, b, c, d, e, f

    def admm_tv_wrapper(self, 
                        inverse_op: InverseOperator, 
                        measurement: torch.Tensor, 
                        clean: torch.Tensor, 
                        lam: float, 
                        max_iter: int=100, 
                        coil_sens: torch.Tensor=None, 
                        mask: torch.Tensor=None,  
                        save_dir: str=None, 
                        tag: str=None)-> Tuple[torch.Tensor, float, float, float, float, float, float]:
        """
        Wrapper for BART based reconstructions. 
        Modify the recon_l1 command to try other BART reconstruction algos.
        Matches interface of PadIS and EDM wrappers.
        """
        
        recon_l1 = bart(1, f'pics -S -l1 -r {lam}', measurement, coil_sens)
        adjoint_np = bart(1, 'pics -S -H', measurement, coil_sens)
        
        recon_l1_shifted = fftshift(np.squeeze(recon_l1))
        denoised_tensor = torch.from_numpy(recon_l1_shifted)
        
        adjoint_shifted = fftshift(np.squeeze(adjoint_np))
        noisy_tensor = torch.from_numpy(adjoint_shifted)

        orig_tensor = clean

        noisypsnr, denoisedpsnr, noisyssim, denoisedssim, noisynrmse, denoisednrmse = makeFigures(
            noisy2=noisy_tensor,
            denoised2=denoised_tensor,
            orig2=orig_tensor, 
            i=0, # dummy index, we only save the final output recon from BART; no intermediate steps
            out_dir=save_dir,
            tag=tag,
            plot=True 
        )
         
        recon_torch = denoised_tensor
         
        return recon_torch, noisypsnr, denoisedpsnr, noisyssim, denoisedssim, noisynrmse, denoisednrmse
        
    def sweep_patch_sizes(
        self,
        num_trials: int,
        patch_sizes: list[int],
        zeta: float,
        num_steps: int,
        save_dir: str,
        algo: str="padis",
        gpus: list[int] = None,
        tag: str = "",
        report_every: int = 10,
    ):
        """
        Try different patch sizes for PaDIS-MRI to see which works best at inference time.
        Note that we found that p=32 or 64 works the best as that was what the most frequent patch sizes were during training.

        Args:
            num_trials: int, number of random val samples to run per patch size
            patch_sizes: list of int, patch sizes to sweep over
            zeta: float, data consistency strength
            num_steps: int, number of diffusion steps
            save_dir: str, directory to save results
            algo: str, "padis" (always)
            gpus: list of int or None, if multiple GPUs are provided, will parallelize over them
            tag: str, tag to append to saved figure filenames (if provided)
            report_every: int, frequency of progress reports
        """
        os.makedirs(save_dir, exist_ok=True)
        if num_trials <= 0 or not self.val_indices:
            print("No validation samples available for sweep_patch_sizes; skipping.")
            return
        subset = random.sample(self.val_indices, num_trials)

        csv_path = os.path.join(save_dir, "patch_sweep_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "psize", "idx", "psnr", "ssim", "nrmse"
            ])
            writer.writeheader()

            summary = {}
            for psize in patch_sizes:
                summary[psize] = {"psnr":[], "ssim":[], "nrmse":[]}
                run_dir = os.path.join(save_dir, f"psize_{psize}/")
                os.makedirs(run_dir, exist_ok=True)

                # update self.psize & pad for this run
                self.psize = psize
                self._build_latents_and_pos(pad_override=psize)

                for i, idx in enumerate(subset, start=1):
                    meas, gt, invop = self._load_sample(idx)
                    args = dict(
                        inverse_op=invop,
                        measurement=meas.cuda(),
                        clean=gt.cuda(),
                        zeta=zeta,
                        pad=psize,
                        psize=psize,
                        num_steps=num_steps,
                        save_dir=run_dir,
                        tag=f"{tag}_psize{psize}_{idx}"
                    )
                    if algo.lower() == "padis":
                        recon, _, psnr_val, _, ssim_val, _, nrmse_val = self.dps2_wrapper(**args)
                    else:
                        print(f"WARNING: Why are you sweeping patch sizes with algo={algo}?")
                        args.pop("pad"); args.pop("psize")
                        recon, _, psnr_val, _, ssim_val, _, nrmse_val = self.dps_edm_wrapper(**args)

                    # record
                    writer.writerow({
                        "psize": psize,
                        "idx": idx,
                        "psnr": psnr_val,
                        "ssim": ssim_val,
                        "nrmse": nrmse_val,
                    })
                    summary[psize]["psnr"].append(psnr_val)
                    summary[psize]["ssim"].append(ssim_val)
                    summary[psize]["nrmse"].append(nrmse_val)

                    if i % report_every == 0:
                        m = sum(summary[psize]["psnr"])/len(summary[psize]["psnr"])
                        print(f"[psize={psize}] done {i}/{num_trials} — PSNR mean so far: {m:.2f}")

            # append summary rows
            writer.writerow({})
            writer.writerow({"psize": "MEAN±STD"})
            for psize, vals in summary.items():
                writer.writerow({
                    "psize": psize,
                    "psnr":  f"{np.mean(vals['psnr']):.2f}±{np.std(vals['psnr']):.2f}",
                    "ssim":  f"{np.mean(vals['ssim']):.4f}±{np.std(vals['ssim']):.4f}",
                    "nrmse": f"{np.mean(vals['nrmse']):.4f}±{np.std(vals['nrmse']):.4f}",
                })

        print(f"Patch‐sweep complete; metrics in {csv_path}")


    def sweep_masks(
        self,
        num_trials: int,
        mask_list: list[int],
        zeta: float,
        num_steps: int,
        save_dir: str,
        algo: str,
        gpus: list[int] = None,
        tag: str = "",
        report_every: int = 10,
        lam: float = 1e-4,
    ):
        """
        For each mask in mask_list, run the selected algorithm on num_trials random samples.
        Saves figures and per-sample recon arrays, and writes a CSV of PSNR/SSIM/NRMSE.
        
        Args:
            num_trials: int, number of random val samples to run per mask
            mask_list: list of int, mask IDs (R=2..10) to sweep over
            gpus: list of int or None, if multiple GPUs are provided, (only tested on one GPU)
            tag: str, tag to append to saved figure filenames (if provided)
            report_every: int, frequency of progress reports
            lam: float, ADMM TV strength (only used if algo="admm")
        """
        os.makedirs(save_dir, exist_ok=True)
        if num_trials <= 0 or not self.val_indices:
            print("No validation samples available for sweep_masks; skipping.")
            return

        subset = random.sample(self.val_indices, num_trials)

        csv_path = os.path.join(save_dir, "mask_sweep_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["mask", "idx", "psnr", "ssim", "nrmse"])
            writer.writeheader()

            summary = {}
            for mask_id in mask_list:
                summary[mask_id] = {"psnr": [], "ssim": [], "nrmse": []}
                run_dir = os.path.join(save_dir, f"mask_{mask_id}")
                os.makedirs(run_dir, exist_ok=True)
                recons_dir = os.path.join(run_dir, "recons")
                os.makedirs(recons_dir, exist_ok=True)

                for i, idx in enumerate(subset, start=1):
                    tag_label = f"{tag}_mask{mask_id}_{idx}"

                    if algo.lower() == "admm":
                        # BART path: explicit mask id
                        meas_bart, gt, _, coil_sens, _, mask_np = self._load_sample_for_bart(
                            idx, mask_ind=mask_id
                        )
                        recon, _, psnr_val, _, ssim_val, _, nrmse_val = self.admm_tv_wrapper(
                            inverse_op=None,
                            measurement=meas_bart,
                            clean=gt,
                            lam=lam,
                            max_iter=num_steps,
                            coil_sens=coil_sens,
                            mask=mask_np,
                            save_dir=run_dir,
                            tag=tag_label,  
                        )

                    else:
                        # Diffusion 
                        meas, gt, invop = self._load_sample(idx, mask_ind=mask_id)
                        args_common = dict(
                            inverse_op=invop,
                            measurement=meas.cuda(),
                            clean=gt.cuda(),
                            zeta=zeta,
                            num_steps=num_steps,
                            save_dir=run_dir,
                            tag=tag_label,
                        )
                        if algo.lower() == "padis":
                            recon, _, psnr_val, _, ssim_val, _, nrmse_val = self.dps2_wrapper(
                                **dict(args_common, pad=self.pad, psize=self.psize)
                            )
                        elif algo.lower() == "edm":
                            recon, _, psnr_val, _, ssim_val, _, nrmse_val = self.dps_edm_wrapper(
                                **args_common
                            )
                        else:
                            raise ValueError(f"Unknown algorithm: {algo}")

                    # CSV row
                    writer.writerow({
                        "mask": mask_id,
                        "idx": idx,
                        "psnr": psnr_val,
                        "ssim": ssim_val,
                        "nrmse": nrmse_val,
                    })

                    # save recon 
                    np.save(os.path.join(recons_dir, f"recon_{tag_label}.npy"),
                            recon.cpu().numpy())

                    # accumulate metrics
                    summary[mask_id]["psnr"].append(psnr_val)
                    summary[mask_id]["ssim"].append(ssim_val)
                    summary[mask_id]["nrmse"].append(nrmse_val)

                    if i % report_every == 0:
                        m = float(np.mean(summary[mask_id]["psnr"]))
                        print(f"[mask={mask_id}] done {i}/{num_trials} — PSNR mean so far: {m:.2f}")

            # summary rows
            writer.writerow({})
            writer.writerow({"mask": "MEAN±STD"})
            for mask_id, vals in summary.items():
                writer.writerow({
                    "mask":  mask_id,
                    "psnr":  f"{np.mean(vals['psnr']):.2f}±{np.std(vals['psnr']):.2f}",
                    "ssim":  f"{np.mean(vals['ssim']):.4f}±{np.std(vals['ssim']):.4f}",
                    "nrmse": f"{np.mean(vals['nrmse']):.4f}±{np.std(vals['nrmse']):.4f}",
                })

        print(f"Mask‐sweep complete; metrics in {csv_path}")


    def _evaluate_zeta(self, zeta: float, num_steps: int, subset: list[int]) -> float:
        """Evaluate average PSNR over a subset of val samples for a given zeta. Helper for hyperparam_search()."""
        if not subset:
            print("WARNING: _evaluate_zeta called with empty subset; returning -inf.")
            return -np.inf
        
        scores = []
        for idx in subset:
            meas, gt, invop = self._load_sample(idx)
            _, _, recon_psnr, _, _, _, _ = self.dps2_wrapper(invop, meas, gt, zeta, self.pad, self.psize, num_steps)
            scores.append(recon_psnr)
        return float(np.mean(scores)) if scores else -np.inf
    
    def hyperparam_search(
        self,
        zeta_min: float = 1.0,
        zeta_max: float = 10.0,
        grid_points: int = 5,
        random_samples: int = 5,
        default_steps: int = 100,
        subset_size: int = 3,
    ):
        """
        1) coarse grid of zeta in [zeta_min,zeta_max]
        2) random refine ±50% around best grid zeta
        Returns best zeta.
        """
        subset = self.val_indices[:subset_size]
        if not subset:
            print("WARNING: hyperparam_search has empty subset; returning default zeta_min.")
            return zeta_min

        grid_zetas = np.linspace(zeta_min, zeta_max, grid_points)
        best_z, best_score = None, -np.inf
        for z in grid_zetas:
            avg_psnr = self._evaluate_zeta(z, default_steps, subset)
            if avg_psnr > best_score:
                best_z, best_score = z, avg_psnr

        # refinement (skip for faster results)
        low, high = max(zeta_min, best_z*0.5), min(zeta_max, best_z*1.5)
        rand_zetas = np.random.uniform(low, high, random_samples)
        for z in rand_zetas:
            avg_psnr = self._evaluate_zeta(z, default_steps, subset)
            if avg_psnr > best_score:
                best_z, best_score = z, avg_psnr

        return float(best_z)
    
    
    def evaluate_uncertainty(
        self,
        seed_list: list[int],
        zeta: float,
        num_steps: int,
        pad: int,
        psize: int,
        algo: str,
        save_dir: str = None,
        gpus: list[int] = None,
        report_every: int = 1,
        tag: str = None,
        lam: float = 1e-4,   # for ADMM only
    ):
        """
        Uncertainty quantification via multiple stochastic reconstructions with different masks at the same R.
        For each val sample, run the chosen algorithm over all masks at selected R generated with different seeds, save per-seed recons,
        and then write pixelwise mean/std maps. Supports padis, edm, and admm. 
        
        Args:
            seed_list: list of int, seeds used to generate different masks at the chosen R
            gpus: list of int or None, if multiple GPUs are provided, will parallelize over them (works for this function only)
        """
        os.makedirs(save_dir, exist_ok=True)
        uncer_dir = os.path.join(save_dir, "uncertainty")
        os.makedirs(uncer_dir, exist_ok=True)

        # pre-create per-sample folders
        for idx in self.val_indices:
            os.makedirs(os.path.join(uncer_dir, str(idx)), exist_ok=True)

        if gpus is None:
            gpus = [torch.cuda.current_device()] if torch.cuda.is_available() else [None]
        else:
            max_dev = torch.cuda.device_count()
            gpus = [g for g in gpus if (g is None) or (0 <= g < max_dev)]
            if not gpus:
                gpus = [torch.cuda.current_device()] if torch.cuda.is_available() else [None]

        splits = [self.val_indices[i::len(gpus)] for i in range(len(gpus))]

        def worker(subset, gpu_id):
            if gpu_id is not None and torch.cuda.is_available():
                torch.cuda.set_device(gpu_id)

            if algo.lower() != "admm" and self.model is not None:
                self.model = self.model.cuda(gpu_id)
                self.latents = self.latents.to(gpu_id)
                self.latents_pos = self.latents_pos.to(gpu_id)

            for i, idx in enumerate(subset, start=1):
                recon_list = []
                sample_dir = os.path.join(uncer_dir, str(idx))

                for mask_seed in seed_list:
                    tag_label = f"{idx}_mask{mask_seed}"

                    if algo.lower() == "padis":
                        meas, gt, invop = self._load_sample(idx, seed_ind=mask_seed)
                        meas = meas.cuda(gpu_id)
                        gt   = gt.cuda(gpu_id)
                        invop.maps = invop.maps.cuda(gpu_id)
                        invop.mask = invop.mask.cuda(gpu_id)

                        recon, *_ = self.dps2_wrapper(
                            inverse_op=invop, measurement=meas, clean=gt,
                            zeta=zeta, pad=pad, psize=psize, num_steps=num_steps,
                            save_dir=sample_dir, tag=tag_label
                        )

                    elif algo.lower() == "edm":
                        meas, gt, invop = self._load_sample(idx, seed_ind=mask_seed)
                        meas = meas.cuda(gpu_id)
                        gt   = gt.cuda(gpu_id)
                        invop.maps = invop.maps.cuda(gpu_id)
                        invop.mask = invop.mask.cuda(gpu_id)

                        recon, *_ = self.dps_edm_wrapper(
                            inverse_op=invop, measurement=meas, clean=gt,
                            zeta=zeta, num_steps=num_steps,
                            save_dir=sample_dir, tag=tag_label
                        )

                    elif algo.lower() == "admm":
                        meas_bart, gt, _, coil_sens, _, mask_np = self._load_sample_for_bart(
                            idx, seed_ind=mask_seed
                        )
                        recon, *_ = self.admm_tv_wrapper(
                            inverse_op=None,
                            measurement=meas_bart,
                            clean=gt,
                            lam=lam,
                            max_iter=num_steps,
                            coil_sens=coil_sens,
                            mask=mask_np,
                            save_dir=sample_dir,
                            tag=tag_label,
                        )

                    else:
                        raise ValueError(f"Unknown algorithm: {algo}")

                    # save per-seed recon
                    recon_np = recon.cpu().numpy()
                    np.save(os.path.join(sample_dir, f"recon_mask{mask_seed}.npy"), recon_np)
                    recon_list.append((mask_seed, recon_np))

                # compute mean/std maps over all seeds
                stack   = np.stack([r for _, r in recon_list], axis=0)  
                mean_map = np.mean(stack, axis=0)
                std_map  = np.std(stack, axis=0)

                np.save(os.path.join(sample_dir, "mean_map.npy"), mean_map)
                np.save(os.path.join(sample_dir, "std_map.npy"),  std_map)

                if i % report_every == 0 or i == len(subset):
                    print(f"[GPU{gpu_id}] done {i}/{len(subset)} samples")

        threads = []
        for gpu, subset in zip(gpus, splits):
            t = threading.Thread(target=worker, args=(subset, gpu))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()

        print("Done with the uncertainty map generation.")

    def evaluate(
        self,
        zeta: float,
        num_steps: int,
        pad: int,
        psize: int,
        algo: str,
        save_dir: str,
        tag: str,
        gpus: list[int] = None,
        report_every: int = 10,
        lam: float = 1e-4, 
    ):
        """
        Runs PaDIS-MRI on all 100 validation volumes, saves side-by-side figures,
        and returns metrics dict with avg+std of PSNR, SSIM, NRMSE.
        """
        os.makedirs(save_dir, exist_ok=True)
        recons_dir = os.path.join(save_dir, "recons")
        os.makedirs(recons_dir, exist_ok=True)
        
        if gpus is None or len(gpus) == 0:
            gpus = [torch.cuda.current_device()] if torch.cuda.is_available() else [None]
        else:
            max_dev = torch.cuda.device_count()
            gpus = [g for g in gpus if (g is None) or (0 <= g < max_dev)]
            if not gpus:
                gpus = [torch.cuda.current_device()] if torch.cuda.is_available() else [None]

        
        splits = [self.val_indices[i::len(gpus)] for i in range(len(gpus))]
        results = {gpu: {'idx':[], 'psnr':[], 'ssim':[], 'nrmse':[]} for gpu in gpus}
                
        def worker(subset, gpu_id):
            torch.cuda.set_device(gpu_id)
            
            if algo.lower() != "admm" and self.model is not None:   
                self.model = self.model.cuda(gpu_id)
                self.latents = self.latents.to(gpu_id)
                self.latents_pos= self.latents_pos.to(gpu_id)
                
            local = results[gpu_id]
            
            for i, idx in enumerate(subset, start=1):
                tag_label = f"{idx}_{tag}"
                
                if algo.lower() == "admm": 
                    meas, gt, invop, coil_sens, ksp_ref, mask = self._load_sample_for_bart(idx)
                else:
                    meas, gt, invop = self._load_sample(idx)
                    
                    meas = meas.cuda(gpu_id)
                    gt = gt.cuda(gpu_id)
                    invop.maps = invop.maps.cuda(gpu_id)
                    invop.mask = invop.mask.cuda(gpu_id)
                
                if algo.lower() == "padis":
                    recon, _, recon_psnr, _, recon_ssim, _, recon_nrmse = self.dps2_wrapper(
                        invop, meas, gt, zeta, pad, psize, num_steps, save_dir, tag_label
                    )
                elif algo.lower() == "edm":
                    recon, _, recon_psnr, _, recon_ssim, _, recon_nrmse = self.dps_edm_wrapper(
                        invop, meas, gt, zeta, num_steps, save_dir, tag_label
                    )
                elif algo.lower() == "admm":
                    recon, _, recon_psnr, _, recon_ssim, _, recon_nrmse = self.admm_tv_wrapper(
                        invop, meas, gt, lam, num_steps, coil_sens, mask, save_dir, tag_label
                    )
                else:
                    raise ValueError(f"Unknown algorithm: {algo}")
                    
                # save
                np.save(os.path.join(recons_dir, f"recon_{tag}_{idx}.npy"), recon.cpu().numpy())

                local['idx'].append(int(idx))
                local['psnr'].append(float(recon_psnr))
                local['ssim'].append(float(recon_ssim))
                local['nrmse'].append(float(recon_nrmse))

                # running report
                if i % report_every == 0 or i == len(subset):
                    m = np.mean(local['psnr'])
                    s = np.std(local['psnr'])
                    print(f"[GPU{gpu_id}] {i}/{len(subset)} — PSNR {m:.2f}±{s:.2f}")
        
        threads = []
        for gpu, subset in zip(gpus, splits):
            t = threading.Thread(target=worker, args=(subset, gpu))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        
        all_idx  = sum((results[g]['idx'] for g in gpus), [])
        all_psnr = sum((results[g]['psnr'] for g in gpus), [])
        all_ssim = sum((results[g]['ssim'] for g in gpus), [])
        all_nrm  = sum((results[g]['nrmse'] for g in gpus), [])
        
        per_image = {
            idx: {'psnr': psnr, 'ssim': ssim, 'nrmse': nrm}
            for idx, psnr, ssim, nrm in zip(all_idx, all_psnr, all_ssim, all_nrm)
        }

        summary = {
            'psnr_mean': np.mean(all_psnr), 'psnr_std': np.std(all_psnr),
            'ssim_mean': np.mean(all_ssim), 'ssim_std': np.std(all_ssim),
            'nrmse_mean': np.mean(all_nrm), 'nrmse_std': np.std(all_nrm),
        }  
        
        metrics = {'per_image': per_image, 'summary': summary}
        
        with open(os.path.join(save_dir, 'metrics.json'), 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return metrics

    def load_model(self, model_path: str, device: str = 'cuda'):
        """Load the EMA network from a .pkl snapshot."""
        with dnnlib.util.open_url(model_path, verbose=False) as f:
            model = pickle.load(f)['ema']
        return model.to(device).eval()
    
    def generate_unconditional_samples(
        self,
        model_paths: list[str],
        output_root: str,
        num_samples_per_model: int = 5,
        algo: str = "padis",
        device: str = 'cuda',
    ):
        """
        For each path in model_paths:
        - load the EMA model
        - create a subfolder under output_root using the model's filename
        - generate num_samples_per_model unconditional images (with distinct seeds)
        - save each as sample_XX_seedYYYY.png
        """
        os.makedirs(output_root, exist_ok=True)
        base_seed = int(time.time())  # changing each run

        for m_idx, model_path in enumerate(model_paths):
            model_name = Path(model_path).parent.name
            out_dir = os.path.join(output_root, model_name)
            os.makedirs(out_dir, exist_ok=True)

            print(f"[{m_idx+1}/{len(model_paths)}] Loading model {model_name}...")
            model = self.load_model(model_path, device)
            self.model = model
            self.latents = self._build_latents_and_pos()

            for i in range(num_samples_per_model):
                seed = base_seed + m_idx * num_samples_per_model + i
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)

                # -- sample and save --
                if algo == "padis":
                    recon, *_ = self.dps2_wrapper(
                        inverse_op=None,
                        measurement=None,
                        clean=None,
                        zeta=0.0,  # not used
                        pad=self.pad,
                        psize=self.psize,
                        num_steps=65,
                        save_dir=out_dir,
                        tag=f"uncond_{i:02d}"
                    )
                else:
                    recon, *_ = self.dps_edm_wrapper(
                        inverse_op=None,
                        measurement=None,
                        clean=None,
                        zeta=0.0,  # not used
                        num_steps=130,
                        save_dir=out_dir,
                        tag=f"uncond_{i:02d}"
                    )
                    
                np.save(os.path.join(out_dir, f"sample_{i:02d}_seed{seed}.npy"), recon)
                
                plt.figure(figsize=(4,4), dpi=100)
                plt.imshow(recon, cmap='gray')
                plt.axis('off')
                plt.tight_layout(pad=0)

                save_path = os.path.join(out_dir, f"sample_{i:02d}_seed{seed}.png")
                plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
                plt.close()
            
            print(f"Saved {num_samples_per_model} samples {out_dir}")
    
    