import sys
sys.path.append('CLAM') #Allows CLAM to maintain its own import structure

import os
from abc import ABC, abstractmethod
from feature_extractor.models import get_encoder
from feature_extractor.dataset_modules.dataset_h5 import Whole_Slide_Bag_FP
from feature_extractor.extract_features_fp import compute_w_loader
import openslide
import torch
from torch.utils.data import DataLoader
from pathlib import Path
from config import Config
import h5py




if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class Extractor():
    def __init__(self, model_path = None, model_name="resnet50.a1_in1k", target_patch_size=224) -> None:
        
        self.target_patch_size = target_patch_size
        print(f"{model_name}")
        self.model, self.img_transforms = get_encoder(model_name, target_img_size=target_patch_size)
        print(model_path)
        if model_path:
            print(model_path)
            self.model = torch.load(model_path)
        self.model.eval()
        self.model = self.model.to(device)
        self.feature_map_size = None

    def extract(self, wsi_file_path: Path, feature_save_path: Path, patch_file_path: Path, skip_existing: bool=True):
        feature_save_path.parent.mkdir(parents=True, exist_ok=True)
        if skip_existing and feature_save_path.resolve().is_file():
            # Check if length of feature_map is same as other features
            feature_map = h5py.File(feature_save_path.resolve())['features'][0] # type: ignore
            self.set_feature_map_size(len(feature_map))    
            
            print(f'{feature_save_path} already exists, skipping extractions, including pre-existing filepaths')
            return feature_save_path, True
        
        wsi = openslide.open_slide(wsi_file_path)
        dataset = Whole_Slide_Bag_FP(file_path=patch_file_path,
                                    wsi=wsi,
                                    img_transforms=self.img_transforms)
        loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
        loader = DataLoader(dataset=dataset, batch_size=256, **loader_kwargs)
        current_feature_map_size = compute_w_loader(feature_save_path, loader = loader, model = self.model, verbose = 1)
        self.set_feature_map_size(current_feature_map_size)
        
        return feature_save_path, False

    
    def set_feature_map_size(self, _feature_map_size):
        if not self.feature_map_size:
                self.feature_map_size = _feature_map_size
        else:
            if self.feature_map_size != _feature_map_size:
                raise ValueError(f"the size of feature map does not match other feature maps. This is most likely result of a corrupted or partial feature file {feature_save_path}")      
    
    def get_feature_map_size(self):
        if self.feature_map_size:
            return self.feature_map_size
        else:
            raise LookupError("Feature Map is not set!")
        
    def get_extractor(self):
        return self.model
    
if __name__ == '__main__':
    e = Extractor(model_name='resnet50_trunc')
    g = e.extract_as_generator(['datasets/training/tiny_camel_train/positive/tumor_001.tif'], ['temp_patches/patch/tumor_001.h5'])
    try: # This try except allows the loop that gives updates to run and then the except gives the final result of the process, taken from the stop itteration exception at the end of the generator
        while True:
            update = next(g)
    except StopIteration as e: # This try except allows the loop that gives updates to run and then the except gives the final result of the process, taken from the stop itteration exception at the end of the generator
        print(e.value)
    