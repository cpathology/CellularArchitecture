# -*- coding: utf-8 -*-

import os, sys
import shutil, argparse, pytz
from joblib import Parallel, delayed
from datetime import datetime
import openslide
import numpy as np
from skimage import io

from block_utils import get_splitting_coors


def set_args():
    parser = argparse.ArgumentParser(description = "Splitting WSI to blocks")
    parser.add_argument("--data_root",         type=str,       default="/Data")
    parser.add_argument("--core_slide_dir",    type=str,       default="CoreSlides")
    parser.add_argument("--block_root_dir",    type=str,       default="BlockAnalysis")
    parser.add_argument("--block_img_dir",     type=str,       default="SlideBlocks")
    parser.add_argument("--block_size",        type=int,       default=6400)
    parser.add_argument("--num_workers",       type=int,       default=32)
    parser.add_argument("--dataset",           type=str,       default="ICON", choices=["ICON", ])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    # Slide directory
    dataset_root = os.path.join(args.data_root, args.dataset)
    core_slide_dir = os.path.join(dataset_root, args.core_slide_dir)
    slide_lst = sorted([os.path.splitext(ele)[0] for ele in os.listdir(core_slide_dir) if ele.endswith(".tiff")])
    # Block directory
    block_img_dir = os.path.join(dataset_root, args.block_root_dir, args.block_img_dir)
    if os.path.exists(block_img_dir):
        shutil.rmtree(block_img_dir)    
    os.makedirs(block_img_dir)

    # split slides one-by-one
    for idx, cur_slide_name in enumerate(slide_lst):
        cur_time_str = datetime.now(pytz.timezone('America/Chicago')).strftime("%m/%d/%Y, %H:%M:%S")
        print("Start @ {}".format(cur_time_str))
        # prepare block directory
        slide_block_dir = os.path.join(block_img_dir, cur_slide_name)
        if not os.path.exists(slide_block_dir):
            os.makedirs(slide_block_dir)
        # load slide
        cur_slide_path = os.path.join(core_slide_dir, cur_slide_name + ".tiff")
        slide_head = openslide.OpenSlide(cur_slide_path)
        slide_w, slide_h = slide_head.dimensions
        wsi_img = slide_head.read_region(location=(0, 0), level=0, size=(slide_w, slide_h))
        np_img = np.asarray(wsi_img)[:, :, :3]
        # save block in a parallel manner
        print("....Splitting {:2d}/{:2d} {}".format(idx+1, len(slide_lst), cur_slide_name))
        coors_list = get_splitting_coors(slide_w, slide_h, args.block_size)
        def save_block(block_info):
            x, y, w, h = block_info
            cur_block_name = cur_slide_name +"-Wstart{:05}Hstart{:05}Wlen{:05}Hlen{:05}.png".format(x, y, w, h)
            cur_block_path = os.path.join(slide_block_dir, cur_block_name)
            if not os.path.exists(cur_block_path):
                cur_block = np_img[y:y+h,x:x+w]
                io.imsave(cur_block_path, cur_block)
        Parallel(n_jobs=args.num_workers)(delayed(save_block)(block_info) for block_info in coors_list)
