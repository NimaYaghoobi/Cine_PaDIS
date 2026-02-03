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


def tqdm(*args, **kwargs):
    if hasattr(tqdm_base, '_instances'):
        for instance in list(tqdm_base._instances):
            tqdm_base._decr_instances(instance)
    return tqdm_base(*args, **kwargs)


def forward_fs(img, m):
    coil_imgs = img*m
    return bart(1, 'fft -u 3', coil_imgs.transpose(1, 2, 0)).transpose(2, 0, 1)

# def normalization_const(s, gt, ACS_size=20, imsize=384):                   
#     # Get normalization constant from undersampled RSS
#     gt_maps_cropped = sp.resize(s, [s.shape[0], imsize, imsize])
#     gt_ksp_cropped = forward_fs(gt[None,...], gt_maps_cropped)
#     # zero out everything but ACS
#     gt_ksp_acs_only = sp.resize(sp.resize(gt_ksp_cropped, (s.shape[0], ACS_size, ACS_size)), gt_ksp_cropped.shape)
#     # make RCS img
#     ACS_img = sp.rss(sp.ifft(gt_ksp_acs_only, axes =(-2,-1)), axes=(0,))
#     norm_const_99 = np.percentile(np.abs(ACS_img), 99)
    
#     return norm_const_99


def normalization_const(s, gt, ACS_size=20, imsize=None):
    # Get normalization constant from undersampled RSS
    H, W = gt.shape[-2], gt.shape[-1]
    if imsize is None:
        imH, imW = H, W
    else:
        imH, imW = imsize if isinstance(imsize, (tuple, list)) else (imsize, imsize)

    gt_maps_cropped = sp.resize(s, [s.shape[0], imH, imW])
    gt_ksp_cropped = forward_fs(gt[None, ...], gt_maps_cropped)

    # zero out everything but ACS
    gt_ksp_acs_only = sp.resize(
        sp.resize(gt_ksp_cropped, (s.shape[0], ACS_size, ACS_size)),
        gt_ksp_cropped.shape
    )

    ACS_img = sp.rss(sp.ifft(gt_ksp_acs_only, axes=(-2, -1)), axes=(0,))
    norm_const_99 = np.percentile(np.abs(ACS_img), 99)
    return norm_const_99
