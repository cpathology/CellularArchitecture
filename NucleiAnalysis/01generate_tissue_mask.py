# -*- coding: utf-8 -*-

import os, sys
import shutil, argparse, pytz, pickle, json
import numpy as np
import openslide
import cv2
from scipy import ndimage
from skimage import io, color, morphology, measure, filters, img_as_ubyte


def set_args():
    parser = argparse.ArgumentParser(description = "Generate tissue mask based on nuclei segmentation")
    parser.add_argument("--data_root",         type=str,       default="/Data")
    parser.add_argument("--core_slide_dir",    type=str,       default="CoreSlides")
    parser.add_argument("--slide_seg_dir",     type=str,       default="CoreSegs") 
    parser.add_argument("--slide_tissue_dir",  type=str,       default="CoreTissues")
    parser.add_argument("--tissue_slide_level",type=int,       default=3)  
    parser.add_argument("--min_tissue_size",   type=int,       default=30000)    
    parser.add_argument("--dataset",           type=str,       default="ICON", choices=["ICON", "Immuno"])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    # directory setting
    dataset_root = os.path.join(args.data_root, args.dataset)
    slide_core_dir = os.path.join(dataset_root, args.core_slide_dir)
    slide_seg_dir = os.path.join(dataset_root, args.slide_seg_dir)
    slide_tissue_dir = os.path.join(dataset_root, args.slide_tissue_dir)
    if not os.path.exists(slide_tissue_dir):
        os.makedirs(slide_tissue_dir)

    core_lst = sorted([os.path.splitext(ele)[0] for ele in os.listdir(slide_core_dir) if ele.endswith(".tiff")])
    # Traverse slide one-by-one
    for ind, cur_core in enumerate(core_lst):
        print("Generate mask on {}/{} on {}".format(ind+1, len(core_lst), cur_core))
        cur_core_path = os.path.join(slide_core_dir, cur_core + ".tiff")
        slide_head = openslide.OpenSlide(cur_core_path)
        # load segmentation
        slide_seg_dict = None
        cur_seg_path = os.path.join(slide_seg_dir, cur_core + ".json")
        if not os.path.exists(cur_seg_path):
            continue
        with open(cur_seg_path) as f:
            slide_seg_dict = json.load(f)
        slide_nuc_dict = slide_seg_dict["nuc"]

        # initiaite the map
        lowlevel_w, lowlevel_h = slide_head.level_dimensions[args.tissue_slide_level]
        cell_map = np.zeros((lowlevel_h, lowlevel_w), dtype=np.uint32)
        divide_ratio = np.power(2, args.tissue_slide_level)
        for key in slide_nuc_dict.keys():
            cell_x, cell_y = slide_nuc_dict[key]["centroid"]
            map_x = int(np.floor(cell_x / divide_ratio))
            map_x = lowlevel_w - 1 if map_x == lowlevel_w else map_x
            map_y = int(np.floor(cell_y / divide_ratio))    
            map_y = lowlevel_h - 1 if map_y == lowlevel_h else map_y
            cell_map[map_y, map_x] += 1
        # refine the map
        cell_map[cell_map >= 255] = 255
        smooth_map = filters.gaussian(cell_map.astype(np.uint8), sigma=9) * 255.0
        density_threshold = divide_ratio * divide_ratio / 5120.0
        cell_mask = smooth_map > density_threshold
        cell_mask = ndimage.binary_fill_holes(cell_mask)
        cell_mask = morphology.remove_small_objects(cell_mask, min_size=args.min_tissue_size, connectivity=8)
        # save cell_mask
        tissue_mask_path = os.path.join(slide_tissue_dir, cur_core + "-L" + str(args.tissue_slide_level) + "-Tissue.png")
        io.imsave(tissue_mask_path, cell_mask * 255)