# ---------------------------------------------------------------
# Modified from:
# https://github.com/jasonhu4/PaDIS/blob/main/training/dataset.py
#
# The license for the original version of this file can be
# found here: https://github.com/jasonhu4/PaDIS/blob/main/LICENSE.
# ---------------------------------------------------------------


"""Streaming images and labels from datasets created with dataset_tool.py."""

import os
import sys
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import scipy.io
from scipy.io import loadmat
import h5py
import matplotlib.pyplot as plt

sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
import dnnlib


try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------
# Abstract base class for datasets.

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        random_seed = 0,        # Random seed to use when applying max_size.
        cache       = False,    # Cache images in CPU memory?
    ):
        print(cache)
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._cache = cache
        self._cached_images = dict() # {raw_idx: np.ndarray, ...}
        self._raw_labels = None
        self._label_shape = None

        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed % (1 << 31)).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])
        # self._raw_idx = self._raw_idx[:max_size]

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        raw_idx = self._raw_idx[idx]
        image = self._cached_images.get(raw_idx, None)
        if image is None:
            image = self._load_raw_image(raw_idx)
            if self._cache:
                self._cached_images[raw_idx] = image
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        #assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]

        return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        # assert self.image_shape[1] == self.image_shape[2]
        # return self.image_shape[1]
        return min(self.image_shape[1], self.image_shape[2])

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        assert len(self.label_shape) == 1
        return self.label_shape[0]

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64
    
"""
Added this new dataloader that directly loads the .pt file.
"""

class ImageFolderDatasetX(Dataset):
    def __init__(self,
        path,
        resolution,
        pad,
        channels = 2,
        imsize = 384,
    ):
        # self.pad = pad
        
        pad_y, pad_x = pad
        self.pad_y = int(pad_y)
        self.pad_x = int(pad_x)
        
        data_path = path
        self._path = "/".join(data_path.split("/")[0:-1]) + "/samples/"
        print("\nLoading Dataset from: " + str(data_path))
        data = torch.load(data_path)
        if "x_est_gt" not in data:
            raise KeyError("The .pt file must contain a tensor with key 'x_est_gt'.")
        
        slice_tensor = data["x_est_gt"]  # Shape: (num_examples, height, width) (complex-valued).
        real_imag_tensor = torch.view_as_real(slice_tensor).numpy().astype(np.float32).transpose(0, 3, 1, 2)
        # self.all_data = np.pad(real_imag_tensor, 
        #                        pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), 
        #                        mode="constant", 
        #                        constant_values=0)

        self.all_data = np.pad(real_imag_tensor,
                                pad_width=((0, 0), (0, 0), (self.pad_y, self.pad_y), (self.pad_x, self.pad_x)),
                                mode="constant",
                                constant_values=0)
    
        self.num_examples = self.all_data.shape[0]  # Number of examples.
        self.channels = self.all_data.shape[1]      # Should be 2 (real + imaginary).
        self.height = self.all_data.shape[2]       # Padded height.
        self.width = self.all_data.shape[3]  # Padded width.
        
        raw_shape = [self.num_examples, self.channels, self.height, self.width]
        name = os.path.splitext(os.path.basename(data_path))[0]
        print(f"Dataset Name: {name}")
        print(f"Dataset chanels: {self.channels}")
        print(f"Dataset Shape: {self.all_data.shape}\n") 
        
        super().__init__(name=name, raw_shape=raw_shape)
    
    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        return NotImplementedError

    def _open_file(self, fname):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)
    
    def _load_raw_image(self, raw_idx):
        image = self.all_data[raw_idx]  # noisy_pt[x_est_gt] size: 2, 384, 384
        return image #2 384 384

    def _load_raw_labels(self):
        return NotImplementedError
        
