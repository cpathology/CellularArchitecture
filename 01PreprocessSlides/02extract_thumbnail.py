# -*- coding: utf-8 -*-

import os, sys
import shutil, argparse, pytz
import openslide
from skimage import io
import numpy as np
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000


def set_args():
    parser = argparse.ArgumentParser(description = "Extract thumbnail from slide core")
    parser.add_argument("--data_root",         type=str,       default="/Data")
    parser.add_argument("--core_slide_dir",    type=str,       default="CoreSlides")
    parser.add_argument("--core_thumbnail_dir",type=str,       default="CoreThumbnails")
    parser.add_argument("--thumbnail_min_size",type=int,       default=400)    
    parser.add_argument("--dataset",           type=str,       default="ICON", choices=["ICON", "Immuno"])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    dataset_root = os.path.join(args.data_root, args.dataset)
    core_slide_dir = os.path.join(dataset_root, args.core_slide_dir)
    core_thumbnail_dir = os.path.join(dataset_root, args.core_thumbnail_dir)
    if not os.path.exists(core_thumbnail_dir):
        os.makedirs(core_thumbnail_dir)
    
    slide_lst = sorted([os.path.splitext(ele)[0] for ele in os.listdir(core_slide_dir) if ele.endswith(".tiff")])
    # traverse slide one-by-one
    for cur_slide in slide_lst:
        raw_slide_path = os.path.join(core_slide_dir, cur_slide + ".tiff")
        slide_head = openslide.OpenSlide(raw_slide_path)
        slide_level_dimensions = slide_head.level_dimensions
        # find thumbnail levels
        thumbnail_level = -1
        for cur_levels in slide_level_dimensions:
            min_val = min(cur_levels)
            if min_val > args.thumbnail_min_size:
                thumbnail_level += 1
        # extract the thumbnail and save
        thumbnail_size = slide_head.level_dimensions[thumbnail_level]
        thumbnail_img = slide_head.read_region(location=(0, 0), level=thumbnail_level, size=thumbnail_size)
        thumbnail_img = np.asarray(thumbnail_img)[:, :, :3]
        thumbnail_path = os.path.join(core_thumbnail_dir, cur_slide + ".png")
        io.imsave(thumbnail_path, thumbnail_img)        