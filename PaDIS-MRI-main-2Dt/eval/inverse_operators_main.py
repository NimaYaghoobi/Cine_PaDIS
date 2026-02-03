import torch
import numpy as np
import sys
import torch.nn.functional as F
from matplotlib import pyplot as plt
from typing import Optional

class MRI_utils:
    """
    Defines the forward and adjoint operations for multi-coil MRI reconstruction.
    """
    def __init__(self, mask: torch.Tensor, maps: torch.Tensor):
        self.mask = mask
        self.maps = maps

    def forward(self,x: torch.Tensor) -> torch.Tensor:
        x_cplx = torch.view_as_complex(x.permute(0,-2,-1,1).contiguous())[:,None,...]
            
        coil_imgs = self.maps*x_cplx
        coil_ksp = fft(coil_imgs)
        sampled_ksp = self.mask*coil_ksp
        return sampled_ksp

    def adjoint(self,y: torch.Tensor) -> torch.Tensor:
        """
        compute the adjoint.
        y: [B, Nc, H, W, 2] (masked k-space)
        """
        sampled_ksp = self.mask*y
        coil_imgs = ifft(sampled_ksp)
        img_out = torch.sum(torch.conj(self.maps)*coil_imgs,dim=1) #sum over coil dimension

        return img_out[:,None,...]

def fft(x: torch.Tensor) -> torch.Tensor:
    x = torch.fft.fft2(x, dim=(-2, -1), norm='ortho')
    return x

def ifft(x: torch.Tensor) -> torch.Tensor:
    x = torch.fft.ifft2(x, dim=(-2, -1), norm='ortho')
    return x

def fftmod(x: torch.Tensor) -> torch.Tensor:
    x[...,::2,:] *= -1
    x[...,:,::2] *= -1
    return x


class InverseOperator(object):
    def __init__(self, imsize: int):
        self.imsize = imsize

    def A(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        Fx = fft(x.squeeze(0))
        y = mask * Fx[None]               # PFx
        return y

    def AT(self, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = mask * y
        x_transpose = ifft(y.squeeze(0))
        return torch.abs(x_transpose[None])

    def Adagger(self, y: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = mask * y
        x_transpose = ifft(y.squeeze(0))
        return torch.abs(x_transpose[None])