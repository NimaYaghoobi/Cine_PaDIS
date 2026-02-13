import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import scipy.io
from diffusers import AutoencoderKL
import random
from skimage.metrics import peak_signal_noise_ratio as psnr
import matplotlib.pyplot as plt
import sys

torch.manual_seed(2)

def getIndices(spaced, patches, pad, psize, freezeindex = False):
    spaced_h, spaced_w = spaced          # spaced = (spaced_h, spaced_w)
    patches_h, patches_w = patches       # patches = (patches_h, patches_w)
    
    # a, b = 0, 0  # Default values when pad = 0
    # if pad > 0:
    #     a = random.randint(0, pad-1)
    #     b = random.randint(0, pad-1)


    pad_y, pad_x = pad  # pad is (pad_y, pad_x)
    
    a, b = 0, 0
    if pad_y > 0:
        a = random.randint(0, pad_y - 1)
    if pad_x > 0:
        b = random.randint(0, pad_x - 1)
    
    if freezeindex:
        a = 0
        b = 0
    indices = []
    for p in range(patches_h):
        for q in range(patches_w):
            indices.append([spaced_h[p]+a, spaced_h[p]+a+psize, spaced_w[q]+b, spaced_w[q]+b+psize])
    return indices

def denoisedFromPatches(net, x, t_hat, latents_pos, class_labels, indices, pad=(0, 0), t_goal = -1, avg=1, spaced=[], wrong=False):
    if len(spaced) > 1:
        indices = getIndices(spaced, (5, 5), 24, 56)
    if wrong:
        x_hat = x
    else:
        x_hat = torch.clone(x)
        
        
    channels = len(x_hat[0,:,0,0])
    # N = len(x_hat[0,0,0,:])
    Hp = x_hat.shape[-2]
    Wp = x_hat.shape[-1]
    psize = indices[0][1] - indices[0][0]
    patches = len(indices)
    # pad = int((N - np.sqrt(patches)*psize))

    output = torch.zeros_like(x_hat)
    x_input = torch.zeros(patches, channels, psize, psize).to(torch.device('cuda'))
    pos_input = torch.zeros(patches, 2, psize, psize).to(torch.device('cuda'))

    for i in range(patches):
        z = indices[i]
        x_input[i,:,:,:] = torch.squeeze(x_hat[0,:,z[0]:z[1], z[2]:z[3]])
        pos_input[i,:,:,:] = torch.squeeze(latents_pos[:,:,z[0]:z[1], z[2]:z[3]])
    
    bigout = net(x_input, t_hat, pos_input, class_labels).to(torch.float64)
    
    for i in range(patches):
        z = indices[i]
        x_patch = x_hat[0,:,z[0]:z[1], z[2]:z[3]]
        output[0,:,z[0]:z[1], z[2]:z[3]] += bigout[i,:,:,:]
        output[0,:,z[0]:z[1], z[2]:z[3]] -= x_patch
    x_hat = x_hat + output

    # temp = t_goal + torch.randn_like(x_hat) * t_goal
    # # temp[:,:,pad:N-pad, pad:N-pad] = x_hat[:,:,pad:N-pad, pad:N-pad]
    # temp[:, :, pad:Hp-pad, pad:Wp-pad] = x_hat[:, :, pad:Hp-pad, pad:Wp-pad]

    pad_y, pad_x = pad  # pad is (pad_y, pad_x)
    
    temp = t_goal + torch.randn_like(x_hat) * t_goal
    temp[:, :, pad_y:Hp-pad_y, pad_x:Wp-pad_x] = x_hat[:, :, pad_y:Hp-pad_y, pad_x:Wp-pad_x]

    return temp
