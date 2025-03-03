from pathlib import Path
from glob import glob
import pandas as pd
from flask import send_file
from io import BytesIO
import time
import random
import os
from config import Config

# Training packages
from misc_utils import get_wsi_extensions
from preprocessing.extractor import Extractor
from pooler import TransMILPooler

def get_dataset_basic_info(folder_path):
    dataset_dir = folder_path
    wsi_paths = [f for f in dataset_dir.rglob('*.*') if f.suffix in get_wsi_extensions() and f.is_file()]
    # print(f"total wsis: {len(wsi_paths)}")
    # print(f"wsi_paths: {wsi_paths}")
    filepaths = [str(f)  for f in wsi_paths]
    filenames = [f.stem for f in wsi_paths]
    extensions = [f.suffix for f in wsi_paths]
    # Girder/dsa saves every file under its own named directory
    # labels = [wsi.parent.parent.stem if wsi.stem == wsi.parent.stem else wsi.parent.stem for wsi in wsi_paths]
    labels = [f.parent.stem for f in wsi_paths]
    sizes = [os.stat(f).st_size for f in wsi_paths]

    df = pd.DataFrame({'filepath':filepaths,
                       'filename':filenames,
                       'extension':extensions,
                       'label':labels,
                       'bytes':sizes})
    
    return df

def serve_pil_image(pil_img):
    img_io = BytesIO()
    pil_img.save(img_io, 'JPEG', quality=70)
    img_io.seek(0)
    return send_file(img_io, mimetype='image/jpeg')



