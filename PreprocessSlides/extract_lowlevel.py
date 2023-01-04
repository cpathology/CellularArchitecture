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
    parser = argparse.ArgumentParser(description = "Extract lowlevel images from slides")
    parser.add_argument("--data_root",         type=str,       default="/Data/CellularArchitectureEmbed")
    parser.add_argument("--slide_dir",         type=str,       default="RawSlides")
    parser.add_argument("--lowlevel_dir",      type=str,       default="LowLevels")
    parser.add_argument("--dataset",           type=str,       default="ICON")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    dataset_root = os.path.join(args.data_root, args.dataset)
    slide_dir = os.path.join(dataset_root, args.slide_dir)
    lowlevel_dir = os.path.join(dataset_root, args.lowlevel_dir)
    if not os.path.exists(lowlevel_dir):
        os.makedirs(lowlevel_dir)
    
    slide_lst = sorted([os.path.splitext(ele)[0] for ele in os.listdir(slide_dir) if ele.endswith(".svs")])
    # traverse slide one-by-one
    for cur_slide in slide_lst:
        raw_slide_path = os.path.join(slide_dir, cur_slide + ".svs")
        slide_head = openslide.OpenSlide(raw_slide_path)
        ttl_level_num = len(slide_head.level_dimensions)
        thumbnail_level = ttl_level_num - 1
        thumbnail_size = slide_head.level_dimensions[thumbnail_level]
        thumbnail_img = slide_head.read_region(location=(0, 0), level=thumbnail_level, size=thumbnail_size)
        thumbnail_img = np.asarray(thumbnail_img)[:, :, :3]
        thumbnail_path = os.path.join(lowlevel_dir, cur_slide + ".png")
        io.imsave(thumbnail_path, thumbnail_img)        