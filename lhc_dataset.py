# lhc_dataset.py
import h5py
import torch
import numpy as np
from torch.utils.data import Dataset
import torch.nn.functional as F

class ChannelDataset(Dataset):
    """
    Loads channel `channel_index` from hdf5 dataset 'X_jets'.
    Outputs (1,H,W) tensors normalized per-sample to [-1,1].
    Ensures final size = image_size (square).
    """
    def __init__(self, hdf5_file, channel_index=0, max_samples=None, image_size=256):
        self.image_size = int(image_size)
        # Open once to read into memory (keeps hdf5 closed)
        with h5py.File(hdf5_file, 'r') as f:
            ds = f['X_jets']
            total = ds.shape[0]
            take = total if max_samples is None else min(total, max_samples)
            X_jets = ds[:take]

        # pick the requested channel and store as float32 numpy array
        channel_data = X_jets[..., channel_index].astype(np.float32)  # shape (N, H, W)
        # convert to torch tensor (N,1,H,W)
        self.data = torch.from_numpy(channel_data).unsqueeze(1)  # (N,1,H,W)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        x = self.data[idx].float()  # (1,H,W)

        # per-sample min-max normalize to [0,1]
        mn = x.min()
        mx = x.max()
        denom = (mx - mn) if (mx - mn) != 0 else 1.0
        x = (x - mn) / (denom + 1e-8)

        # ensure correct size using interpolation (handles arbitrary original sizes)
        # interpolate expects (N,C,H,W) floats in range [0,1]
        x = x.unsqueeze(0)  # (1,1,H,W)
        x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
        x = x.squeeze(0)  # (1,H,W)

        # map [0,1] -> [-1,1]
        x = x * 2.0 - 1.0

        return x
