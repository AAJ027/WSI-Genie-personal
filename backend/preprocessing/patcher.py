import sys
sys.path.append('CLAM') #Allows CLAM to maintain its own import structure

from pathlib import Path
from glob import glob
import os
from preprocessing.wsi_core import WholeSlideImage
from preprocessing.wsi_core.wsi_utils import StitchCoords
from config import Config
from misc_utils import get_wsi_extensions

def patch(wsi_file_path: str | Path, patch_size: int=256, patch_save_path: str="", stitch_save_path:str="", skip_existing: bool=True):
    # print(f"{patch_save_path=}, {stitch_save_path=}")
    
    # Creating the directories if they are not already created
    patch_save_path.parent.mkdir(parents=True, exist_ok=True)
    stitch_save_path.parent.mkdir(parents=True, exist_ok=True)

    if skip_existing and patch_save_path.resolve().is_file():
        print(f'{patch_save_path} already exists, skipping patching, including pre-existing filepaths')
        return patch_save_path, stitch_save_path, True

    # Create WSI Object
    print("Created WSI Object")
    wsi_obj = WholeSlideImage(str(wsi_file_path))


    # Segment
    print("Segmenting")
    seg_level = wsi_obj.get_best_level_for_downsample(64)
    # Defaulr Params as per CLAM
    filter_params={'a_t':1, 'a_h':1, 'max_n_holes':2}
    wsi_obj.segmentTissue(seg_level=seg_level, filter_params=filter_params)


    # Patch
    print("Patching")
    wsi_obj.process_contours(str(patch_save_path.parent), patch_size=patch_size, step_size=patch_size) #Parent selected to deal with internal clam stuff TODO: Make this cleaner
    
    # Stitch
    print("Stitching")
    heatmap = StitchCoords(patch_save_path, wsi_obj, downscale=64, bg_color=(0,0,0), alpha=-1, draw_grid=False)
    heatmap.save(stitch_save_path)
    
    return patch_save_path, stitch_save_path, False

if __name__ == '__main__':
    print('Test Patching')
    input_wsi = Path("/Users/parth/datasets/CAMELYON16/training/tumor/tumor_101.tif")
    patch_save_path = Path("/Users/parth/Desktop/lung_colon_cancer/data/sample2_lung_image_sets/lung_n/patch_data")
    stitch_save_path = Path("/Users/parth/Desktop/lung_colon_cancer/data/sample2_lung_image_sets/lung_n/stitch_data")

    patch(wsi_file_path=input_wsi,
         patch_save_dir=patch_save_path,
         stitch_save_dir=stitch_save_path,
         patch_size=256)