# internal imports
# from wsi_core.WholeSlideImage import WholeSlideImage
from wsi_core.wsi_object import WholeSlideImage
from wsi_core.wsi_utils import StitchCoords
from wsi_core.batch_process_utils import initialize_df

# other imports
import os
import numpy as np
import time
import argparse
import pandas as pd
import tqdm
import glob

def stitching(file_path, wsi_object, downscale = 64):
        start = time.time()
        heatmap = StitchCoords(file_path, wsi_object, downscale=downscale, bg_color=(0,0,0), alpha=-1, draw_grid=False)
        total_time = time.time() - start
        
        return heatmap, total_time

def segment(WSI_object, seg_params = None, filter_params = None, mask_file = None):
    ### Start Seg Timer
    start_time = time.time()
    # Use segmentation file
    if mask_file is not None:
        WSI_object.initSegmentation(mask_file)
    # Segment	
    else:
        WSI_object.segmentTissue(**seg_params, filter_params=filter_params)

    ### Stop Seg Timers
    seg_time_elapsed = time.time() - start_time   
    return WSI_object, seg_time_elapsed

def patching(WSI_object, **kwargs):
    ### Start Patch Timer
    start_time = time.time()

    # Patch
    file_path = WSI_object.process_contours(**kwargs)


    ### Stop Patch Timer
    patch_time_elapsed = time.time() - start_time
    return file_path, patch_time_elapsed


def seg_and_patch(source, csv_save_dir, patch_save_dir, stitch_save_dir,
                  patch_size = 256, step_size = 256, 
                  seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,},
                  filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}, 
                  patch_params = {'use_padding': True, 'contour_fn': 'four_pt'},
                  patch_level = 0,
                  auto_skip=True):
    
 
    ## TODO: add compatibility for other extensions like tif
    slides = sorted(glob.glob(os.path.join(source, "*.tif")))
    slide_names = list(map(os.path.basename, slides))
    df = initialize_df(slides, seg_params, filter_params, None, patch_params)


    ## df is a pd Dataframe with data of the csv created - `process_list_autogen.csv`
    total = len(df)


    seg_times = 0.
    patch_times = 0.
    # stitch_times = 0.
    progress = tqdm.tqdm(range(total))
    # progress = tqdm.tqdm()
    for idx in progress:
        df.to_csv(os.path.join(csv_save_dir, 'process_list_autogen.csv'), index=False)
        slide_path = os.path.join(source, df.loc[idx, 'slide_path'])
        slide_name = os.path.basename(slide_path)
        progress.set_description('processing {}'.format(slide_path))
        
        
        df.loc[idx, 'process'] = 0
        slide_id, _ = os.path.splitext(slide_name)

        if auto_skip and os.path.isfile(os.path.join(patch_save_dir, slide_id + '.h5')):
            print('{} already exists in destination location, skipped'.format(slide_id))
            df.loc[idx, 'status'] = 'already_exist'
            continue

        # Inialize WSI
        full_path = slide_path
        # print(f"full_path: {full_path}")
        WSI_object = WholeSlideImage(full_path)


        current_filter_params = filter_params   # {}
        current_seg_params = seg_params # {}
        current_patch_params = patch_params # {}


        if current_seg_params['seg_level'] < 0:
            if len(WSI_object.level_dim) == 1:
                current_seg_params['seg_level'] = 0
            else:
                current_seg_params['seg_level'] = WSI_object.get_best_level_for_downsample(64)

        w, h = WSI_object.level_dim[current_seg_params['seg_level']] 
        if w * h > 1e8:
            print('level_dim {} x {} is likely too large for successful segmentation, aborting'.format(w, h))
            df.loc[idx, 'status'] = 'failed_seg'
            continue

        df.loc[idx, 'seg_level'] = current_seg_params['seg_level']



        WSI_object, seg_time_elapsed = segment(WSI_object, current_seg_params, current_filter_params) 

        patch_time_elapsed = -1 # Default time
        current_patch_params.update({'patch_level': patch_level, 'patch_size': patch_size, 'step_size': step_size, 
                                        'save_path': patch_save_dir})
        _, patch_time_elapsed = patching(WSI_object = WSI_object,  **current_patch_params,)
        


        # print("segmentation took {} seconds".format(seg_time_elapsed))
        # print("patching took {} seconds".format(patch_time_elapsed))
        df.loc[idx, 'status'] = 'processed'

        seg_times += seg_time_elapsed
        patch_times += patch_time_elapsed

    seg_times /= total
    patch_times /= total

    df.to_csv(os.path.join(csv_save_dir, 'process_list_autogen.csv'), index=False)
    print("average segmentation time in s per slide: {}".format(seg_times))
    print("average patching time in s per slide: {}".format(patch_times))
        
    return seg_times, patch_times




def create_patches(source, patch_save_dir, csv_save_dir=None, patch_size = 256, step_size = 256, patch_level=0, auto_skip=True):
    print('source: ', source)
    print('patch_save_dir: ', patch_save_dir)

    if not csv_save_dir:
        csv_save_dir = patch_save_dir
    
    directories = {'source': source, 
                   'patch_save_dir': patch_save_dir,
                   'csv_save_dir': csv_save_dir
                   } 
    os.makedirs(patch_save_dir, exist_ok=True)
    os.makedirs(csv_save_dir, exist_ok=True)
            

    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,}
    filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    
    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                   }
    print(parameters)

    seg_times, patch_times = seg_and_patch(source, patch_save_dir=patch_save_dir, csv_save_dir=csv_save_dir, 
                                           **parameters, patch_size = patch_size,
                                           step_size=step_size, patch_level=patch_level,
                                            auto_skip=auto_skip)



def main():
    parser = argparse.ArgumentParser(description='seg and patch')
    parser.add_argument('--source', type = str,
                        help='directory to save processed data')
    parser.add_argument('--patch_save_dir', type = str,
                        help='directory to save processed data')
    parser.add_argument('--csv_save_dir', default=None, type = str,
                        help='directory to save csv')
    
    parser.add_argument('--step_size', type = int, default=256,
                        help='step_size')
    parser.add_argument('--patch_size', type = int, default=256,
                        help='patch_size')
    parser.add_argument('--patch_level', type=int, default=0, 
                        help='downsample level at which to patch')

    parser.add_argument('--auto_skip', default=True, action='store_true')

    


    args = parser.parse_args()
    source = args.source
    patch_save_dir = args.patch_save_dir
    
    if not args.csv_save_dir:
        csv_save_dir = patch_save_dir
    else:
        csv_save_dir = args.csv_save_dir

    
    directories = {'source': args.source, 
                   'save_dir': args.save_dir,
                   'patch_save_dir': patch_save_dir,
                   } 

    for key, val in directories.items():
        print("{} : {}".format(key, val))
        if key not in ['source']:
            os.makedirs(val, exist_ok=True)

    seg_params = {'seg_level': -1, 'sthresh': 8, 'mthresh': 7, 'close': 4, 'use_otsu': False,}
    filter_params = {'a_t':100, 'a_h': 16, 'max_n_holes':8}
    patch_params = {'use_padding': True, 'contour_fn': 'four_pt'}

    
    parameters = {'seg_params': seg_params,
                  'filter_params': filter_params,
                  'patch_params': patch_params,
                   }
    print(parameters)

    seg_times, patch_times = seg_and_patch(source = source, patch_save_dir=patch_save_dir, csv_save_dir=csv_save_dir, 
                                           **parameters, patch_size = args.patch_size, step_size=args.step_size, 
                                            patch_level=args.patch_level, auto_skip=args.no_auto_skip)

if __name__ == '__main__':
    main()
