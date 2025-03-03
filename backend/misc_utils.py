from contextlib import contextmanager
import sys, os
from pathlib import Path
import torch
import numpy as np
import h5py


def get_wsi_extensions():
    return ('.svs', '.tif',
            '.dcm',
            '.vms', '.vmu', '.ndpi',
            '.scn',
            '.mrxs',
            '.tiff',
            '.svslide',
            '.tif',
            '.bif')

def converge_feature_options(feature):
    """
    Takes whatever the feature is and makes it a tensor

    """
    # String into path
    if isinstance(feature, str):
        feature = Path(feature)

    # Load files
    if isinstance(feature, Path) and feature.suffix == '.pt':
        feature = torch.load(feature) # pytorch
    if isinstance(feature, Path) and feature.suffix == '.npy':
        feature = np.load(feature)   # numpy
    if isinstance(feature, Path) and feature.suffix == '.h5':
        feature = h5py.File(feature)['features'][:] # h5 - given the files has a `features` key

    # Convert to tensor
    if isinstance(feature, np.ndarray):
        feature = torch.from_numpy(feature)
    
    return feature
