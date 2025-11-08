import torch
import numpy as np
import math
import torch.distributed as dist
from torch.utils.data.sampler import Sampler
from torch.utils.data import DataLoader
import os
import multiprocessing
def get_data_scaler(config):
    """Data normalizer. Assume data are always in [0, 1]."""
    if config.data.centered:
        # Rescale to [-1, 1]
        return lambda x: x * 2.0 - 1.0
    else:
        return lambda x: x


def get_data_inverse_scaler(config):
    """Inverse data normalizer."""
    if config.data.centered:
        # Rescale [-1, 1] to [0, 1]
        return lambda x: (x + 1.0) / 2.0
    else:
        return lambda x: x


def load_data(
    data_dir,
    val_dir,
    val_bsz,
    dataset,
    spacing_limits,
    n_slices_per_sandwich,
    local_batch_size,
    image_size,
    loss,
    deterministic=False,
    include_test=False,
    seed=42,
    num_workers=2,
    num_volumes=1,
):

    # Compute batch size for this worker.
    root = data_dir
    corrupt_type = None

    from .photosynth import Photosynth
    from .photo_validation import Validation_stacker

    trainset = Photosynth(data_dir, 
                          n_slices_per_sandwich=n_slices_per_sandwich, 
                          nvols=num_volumes,
                          spacing_limits=[int(spacing_limits[0]), int(spacing_limits[2:])],
                          local_batch_size=local_batch_size, 
                          loss='L1') 
    
    valset = Validation_stacker(val_dir)

    train_loader = DataLoader(
        dataset=trainset,
        num_workers=num_workers,
        persistent_workers=False,
        pin_memory=True) 
    
    val_loader = DataLoader(
        dataset=valset,
        batch_size=val_bsz,
        num_workers=num_workers,
        shuffle=False,
        persistent_workers=False,
        pin_memory=True) 


    return train_loader, val_loader
