from torch.utils.data import Dataset
from torchvision import transforms

from typing import Callable, Dict, List, Tuple

import numpy as np
import os

from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator

import skimage
from skimage.filters import threshold_otsu
from skimage import draw, io
from skimage.util import img_as_ubyte

from PIL.ImageStat import Stat

import glob
from xml.etree import ElementTree as ET

import pandas as pd
pd.options.mode.chained_assignment = None

# Remove DecompressionBomb warning or exception from Pillow Image
from PIL import Image
Image.MAX_IMAGE_PIXELS = None


class wsiDataset(Dataset):
    def __init__(self, image_dir: str, annotations_dir: str=None, masks_dir: str=None, tile_size: int = 256, transform: Callable=None, skip: bool=True):
        '''
        Initialize a dataset with Whole Slide Images. 
        Create masks if annotation file is provided. and mark only those masked patches as positive instnaces
        '''
        # super().__init__()
        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        self.masks_dir = masks_dir
        self.tile_size = tile_size
        self.transform = transform
        self.skip = skip

        if not (self.annotations_dir or self.masks_dir):
            raise NotImplementedError("Did not get annotations directory or masks directory.")
        
        if self.annotations_dir and not self.masks_dir:
                self.masks_dir = os.path.join(image_dir, "masks")

        ## TODO: Add suuport for a class_to_idx json file.
        ## TODO: Add an option for user to change class' indices after auto-detecting the classes
        self.classes, self.class_to_idx = self.__find_classes()

        self.image_to_class = {}    # image_path -> class
        for target_class in self.classes:
            target_dir = os.path.join(image_dir, target_class)
            print(f"walking in {target_dir}, class_to_idx is {self.class_to_idx}")
            supported_extensions = ["*.tif", "*.svs"]
            files = []
            for extension in supported_extensions:
                 files.extend(sorted(glob.glob(os.path.join(target_dir, extension))))
            image_map = {file_path: target_class for file_path in files}
            self.image_to_class.update(image_map)

        if self.annotations_dir:
            print(f"Generating masks from annotations, it might take a while")
            self.__generate_masks()

        self.instances = pd.DataFrame()
        for slide_path, target_class in self.image_to_class.items():
            tiles = self.get_tiles(slide_path, target_class)

        # Balancing positive and negative examples
        # TODO: check for binary or multiclass classification before balancing
        # TODO: Add a variable to allow small imbalanes (prevelance<x%)
        positive_instances_idx = self.instances[self.instances['label'] == 1].idx
        negative_instances_idx = self.instances[self.instances['label'] == 0].idx
        # print(f"positive_instances: {positive_instances.count()}, negative_insatances: {negative_instances.count()}")
        difference = len(positive_instances_idx) - len(negative_instances_idx)
        
        if difference > 0:
            idx=positive_instances_idx.tolist()
            # print("Dropping positive classes")
            drop_indices = np.random.choice(idx, difference, replace=False)
        elif difference < 0:
            idx=negative_instances_idx.tolist()
            # print("Dropping negative classes")
            drop_indices = np.random.choice(idx, difference, replace=False)
        self.instances = self.instances.drop(drop_indices) # type: ignore
        self.instances.reset_index(drop=True, inplace=True)
        
        # print(f"positive_instances: {positive_instances.count()}, negative_insatances: {negative_instances.count()}")
        
    def get_tiles(self, slide_path: str, target_class: int) -> None:
        """
        Processes a whole slide image file specified by `slide_path` to extract and analyze tiles, and returns
        a DataFrame with various computed statistics for each tile.

        Updates instance variable `self.instances`

        Parameters:
            slide_path (str): The file path to the whole slide image.
            target_class (int): The class identifier for which tiles are processed.

        Returns:
            pd.DataFrame: A DataFrame containing the following columns:
                - slide_path (str): Path to the slide image.
                - tiles (tuple(int, int)): The coordinates of the top left corner of each tile.
                - dimensions (tuple(int, int)): The dimensions (width, height) of each tile.
                - mean (float): The mean pixel value of the tile.
                - std (float): The standard deviation of the pixel values in the tile.
                - label (int): The class label assigned to the tile, typically related to `target_class`.
        """        
        # TODO: Maybe add option to use userset/predefined threshold for otsu
        fname = os.path.basename(slide_path)
        with open_slide(slide_path) as slide:
            slide_width, slide_height = slide.dimensions
            thumbnail = slide.get_thumbnail((slide_width / self.tile_size, slide_height / self.tile_size))
        
        image_stats = Stat(thumbnail)
        mean = image_stats.mean
        std = image_stats.stddev
        
        thumbnail_grey = np.array(thumbnail.convert('L')) # convert to grayscale
        thresh = threshold_otsu(thumbnail_grey)

        # Binarize image to separate tissue and background
        binary = thumbnail_grey > thresh
        tiles = pd.DataFrame(pd.DataFrame(binary).stack())
        tiles['is_tissue'] = ~tiles[0]
        tiles.drop(0, axis=1, inplace=True)
    

        # Find mask corresponding to slide image
        fname_no_extension = os.path.splitext(fname)[0]
        masks_list = os.listdir(self.masks_dir)
        mask_name = fname_no_extension+"_mask.png"

        # Masks only available for positive instances
        if mask_name in masks_list:
            mask_path = os.path.join(self.masks_dir, mask_name)
            with open_slide(mask_path) as mask:
                ## TODO: Add a check for same size on mask and slide or get the thumbnail same size as slide
                thumbnail_mask = mask.get_thumbnail(mask.dimensions)
            tiles_roi = pd.DataFrame(pd.DataFrame(np.array(thumbnail_mask.convert("L"))).stack())
            # Unmasked region is our Region of Interest
            tiles_roi['roi'] = tiles_roi[0] > 0
            tiles_roi.drop(0, axis=1, inplace=True)
            tiles = pd.concat([tiles, tiles_roi], axis=1)
            tiles = tiles[tiles.roi == True]     # Only keep tiles which are in our RoI
            tiles.drop('roi', axis=1, inplace=True)
        
        tiles = tiles[tiles.is_tissue == True]  # Only keep tiles which have tissues/remove background tiles
        tiles.drop("is_tissue", axis=1, inplace=True)                
        
        tiles['tiles'] = list(tiles.index)   # Get location of tiles (x, y)
        tiles.reset_index(inplace=True, drop=True)
    
        tiles["slide_path"] = slide_path
        tiles["mean"] = mean
        tiles["std"] = std
        tiles["dimensions"] = (slide_width, slide_height)
        tiles['label'] = target_class

        # Reordering the columns of the dataframe
        tiles = tiles[["slide_path", "tiles", "dimensions", "mean", "std", "label"]]
        pd.concat((self.instances, tiles), axis=0, ignore_index=True)
        return tiles
        
    
    def __find_classes(self) -> Tuple[List[str], Dict[str, int]]:
        """Finds the class folders in a dataset.
        """
        # Directroy naems are classes
        classes = sorted(entry.name for entry in os.scandir(self.image_dir) if entry.is_dir())

        if self.annotations_dir or self.masks_dir:
            # If annotataions directory is in same root directory, remove it from classes
            annotations_dir = os.path.basename(os.path.normpath(self.annotations_dir))
            masks_dir = os.path.basename(os.path.normpath(self.masks_dir))
            if annotations_dir in classes:
                classes.remove(annotations_dir)
            if masks_dir in classes:
                classes.remove(masks_dir)
        
        if not classes:
            raise FileNotFoundError(f"Couldn't find any class folder in {self.image_dir}.")
        
    
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        print(f"Found classes: {classes}")
        return sorted(classes), class_to_idx

    def __generate_masks(self):
        os.makedirs(self.masks_dir, exist_ok=True)
        annotation_path_list = glob.glob(os.path.join(self.annotations_dir, '*.xml'))
        annotation_name_list = [os.path.splitext(os.path.basename(path))[0] for path in annotation_path_list]
        mask_list = glob.glob(os.path.join(self.masks_dir, '*_mask.png'))

        for slide_path, target_class in self.image_to_class.items():
                fname = os.path.basename(slide_path)
                fname_no_extension = os.path.splitext(fname)[0]

                if fname_no_extension not in annotation_name_list:
                    continue
                output_name = os.path.join(self.masks_dir, fname_no_extension+"_mask.png")
                if self.skip and output_name in mask_list:
                    print(f"Mask file {os.path.basename(output_name)} already exists, skipping", end="\r")
                    continue
                
                with open_slide(slide_path) as slide:
                    slide_width, slide_height = slide.dimensions

                # Initialize the mask
                mask_width, mask_height = slide_height//self.tile_size, slide_width//self.tile_size
                mask = np.zeros((mask_width, mask_height))
                
                tree = ET.parse(os.path.join(self.annotations_dir, fname_no_extension+".xml"))
                xml_root = tree.getroot()

                # Iterate through each annotation to draw the polygons
                for annotation in xml_root.findall('.//Annotation'):
                    vertices_row = []
                    vertices_col = []
                    
                    for coordinate in annotation.find('.//Coordinates'): # type: ignore
                        x = float(coordinate.get('X'))//self.tile_size # type: ignore
                        y = float(coordinate.get('Y'))//self.tile_size # type: ignore
                        
                        vertices_col.append(x)
                        vertices_row.append(y)

                    if vertices_row and vertices_col:
                        # Draw the polygon on the mask
                        polygon_rr, polygon_cc = draw.polygon(vertices_row, vertices_col)
                        mask[polygon_rr, polygon_cc] += 1
                mask = mask%2
                # mask = mask == 1
                io.imsave(output_name, img_as_ubyte(mask))
        print()

    def __len__(self):
        return len(self.instances.index)
    
    def num_classes(self):
        return len(self.classes)
    
    def __getitem__(self, index):
        ## TODO: Maybe cache the tiles for reuse
        image_path, tile_loc, dimensions, mean, std, y = self.instances.iloc[index]
        with open_slide(image_path) as slide:
            tiles = DeepZoomGenerator(slide, tile_size=self.tile_size, overlap=0, limit_bounds=False) 
            max_resolution = tiles.level_count-1    # Last level is highest resolution
            x = tiles.get_tile(max_resolution, tile_loc[::-1])
            norm = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
            x = norm(x)
        return (image_path, tile_loc, dimensions, x, y)


if __name__ == "__main__":
    annotations_path = "/Users/parth/datasets/camelyon16/training/lesion_annotations"

    image_dir = "/Users/parth/datasets/CAMELYON16/training/"
    annotations_path = "/Users/parth/datasets/CAMELYON16/training/lesion_annotations"
    dataset = wsiDataset(image_dir="/Users/parth/datasets/CAMELYON16/training", annotations_dir=annotations_path)

    # dataset = wsiDataset(image_dir=image_dir, max_tiles=500, transform=transforms.ToTensor(), annotations_dir=annotations_path)
    print(f"Total length of dataset is {len(dataset)}")