import time
import os
import argparse

import torch
from torch.utils.data import DataLoader
import h5py
import openslide
from tqdm import tqdm

import numpy as np

from .utils.file_utils import save_hdf5
from .dataset_modules.dataset_h5 import Dataset_All_Bags, Whole_Slide_Bag_FP
from .models import get_encoder


## TODO: Add a check for device selection `cuda``, `mps` or `cpu`
## TODO: If patch_size is not given, send original image to FE
## TODO: Change all the `os.listdir` to `glob`
## TODO: remove redundancy of h5 and pt files

# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
    
def compute_w_loader(output_path, loader, model, verbose = 0):
    """
    args:
        output_path: directory to save computed features (.h5 file)
        model: pytorch model
        verbose: level of feedback
    """
    if verbose > 0:
        print(f'processing a total of {len(loader)} batches'.format(len(loader)))

    mode = 'w'
    length_of_features = 0
    for _, data in enumerate(tqdm(loader)):
        with torch.inference_mode():	
            batch = data['img']
            coords = data['coord'].numpy().astype(np.int32)
            batch = batch.to(device, non_blocking=True)
            
            features = model(batch)
            # Hibou model returns a dict with two keys 'last_hidden_state' and 'pooler_output'
            # We need values of pooler_output
            if isinstance(features, dict):
                features=features.pooler_output
            features = features.cpu().numpy().astype(np.float32)
            length_of_features = features.shape[1]
            asset_dict = {'features': features, 'coords': coords}
            save_hdf5(output_path, asset_dict, attr_dict= None, mode=mode)
            mode = 'a'
    
    return length_of_features



# def extract_features(csv_path, patches_dir, output_dir="extracted_features", 
#                      model_name="resnet50", target_patch_size=224, 
#                      auto_skip=True, 
#                      batch_size=256):
    
#     if csv_path is None:
#         raise NotImplementedError

#     bags_dataset = Dataset_All_Bags(csv_path)
    
#     os.makedirs(output_dir, exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'pt_files'), exist_ok=True)
#     os.makedirs(os.path.join(output_dir, 'h5_files'), exist_ok=True)
#     dest_files = os.listdir(os.path.join(output_dir, 'pt_files'))

#     model, img_transforms = get_encoder(model_name, target_img_size=target_patch_size)
            
#     model.eval()
#     model = model.to(device)

#     loader_kwargs = {'num_workers': 8, 'pin_memory': True} if device.type == "cuda" else {}
#     progress= tqdm(range(len(bags_dataset)))
#     for bag_candidate_idx in progress:
#         slide_path = bags_dataset[bag_candidate_idx]
#         slide_name = os.path.basename(slide_path)
#         slide_id = os.path.splitext(slide_name)[0]
#         bag_name = slide_id + '.h5'
#         h5_file_path = os.path.join(patches_dir, bag_name)
#         progress.set_description(f"working with {slide_name}")
        

#         if auto_skip and slide_id+'.pt' in dest_files:
#             print('skipped {}'.format(slide_id))
#             continue 

#         output_path = os.path.join(output_dir, 'h5_files', bag_name)
#         time_start = time.time()
#         wsi = openslide.open_slide(slide_path)
#         dataset = Whole_Slide_Bag_FP(file_path=h5_file_path, 
#                                         wsi=wsi,  # type: ignore
#                                         img_transforms=img_transforms,
#                                         debug = True)

#         loader = DataLoader(dataset=dataset, batch_size=batch_size, **loader_kwargs)
#         output_file_path = compute_w_loader(output_path, loader = loader, model = model, verbose = 0)

#         time_elapsed = time.time() - time_start
#         # print('\ncomputing features for {} took {} s'.format(output_file_path, time_elapsed))

#         with h5py.File(output_file_path, "r") as file:
#             features = file['features'][:] # type: ignore
#             # print('features size: ', features.shape)
#             # print('coordinates size: ', file['coords'].shape)

#         features = torch.from_numpy(features)
#         bag_base, _ = os.path.splitext(bag_name)
#         torch.save(features, os.path.join(output_dir, 'pt_files', bag_base+'.pt'))









# def main():
#     parser = argparse.ArgumentParser(description='Feature Extraction')
#     parser.add_argument('--patches_dir', type=str, default=None,
#                         help="Location of the stored patches (h5 files)")
#     parser.add_argument('--data_slide_dir', type=str, default=None,
#                         help="directory containing original WSIs")
#     parser.add_argument('--slide_ext', type=str, default= '.svs')
#     parser.add_argument('--csv_path', type=str, default=None,
#                         help="Path of the csv file generated during patching")
#     parser.add_argument('--output_dir', type=str, default=None,
#                         help="Output directory where features will be stored")
#     parser.add_argument('--model_name', type=str, default='resnet50_trunc', choices=['resnet50_trunc', 'uni_v1', 'conch_v1'])
#     parser.add_argument('--batch_size', type=int, default=256)
#     parser.add_argument('--no_auto_skip', default=False, action='store_true')
#     parser.add_argument('--target_patch_size', type=int, default=224)

#     args = parser.parse_args()
#     print('initializing dataset')
#     extract_features(args.csv_path, args.output_dir, args.model_name, args.target_patch_size, args.patches_dir, args.auto_skip, args.batch_size)








# if __name__ == '__main__':
#     main()
    


