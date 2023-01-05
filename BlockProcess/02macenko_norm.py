# -*- coding: utf-8 -*-

import os, sys
import shutil, argparse, pytz
from datetime import datetime
import numpy as np
from skimage import io
from histocartography.preprocessing import MacenkoStainNormalizer


def set_args():
    parser = argparse.ArgumentParser(description = "Macenko stain norm block images")
    parser.add_argument("--data_root",         type=str,       default="/Data")
    parser.add_argument("--block_root_dir",    type=str,       default="BlockAnalysis")
    parser.add_argument("--block_img_dir",     type=str,       default="SlideBlocks")
    parser.add_argument("--block_norm_dir",    type=str,       default="BlockNorms")
    parser.add_argument("--dataset",           type=str,       default="ICON", choices=["ICON", "Immuno"])
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    # directory setting
    dataset_root = os.path.join(args.data_root, args.dataset)
    block_img_dir = os.path.join(dataset_root, args.block_root_dir, args.block_img_dir)
    slide_lst = [ele for ele in os.listdir(block_img_dir) if os.path.isdir(os.path.join(block_img_dir, ele))]
    block_norm_dir = os.path.join(dataset_root, args.block_root_dir, args.block_norm_dir)
    if os.path.exists(block_norm_dir):
        shutil.rmtree(block_norm_dir)
    os.makedirs(block_norm_dir)

    # normalize image one-by-one
    normalizer = MacenkoStainNormalizer() 
    for ind, cur_slide in enumerate(slide_lst):
        slide_block_dir = os.path.join(block_img_dir, cur_slide)
        slide_block_lst = sorted([ele for ele in os.listdir(slide_block_dir) if ele.endswith(".png")])
        print("Normalize {}/{} {} with {} blocks".format(ind+1, len(slide_lst), cur_slide, len(slide_block_lst)))
        slide_norm_dir = os.path.join(block_norm_dir, cur_slide)
        os.makedirs(slide_norm_dir)
        for cur_block in slide_block_lst:
            cur_block_path = os.path.join(slide_block_dir, cur_block)
            cur_block_img = io.imread(cur_block_path)
            cur_norm_block = normalizer.process(cur_block_img)
            cur_norm_path = os.path.join(slide_norm_dir, cur_block)
            io.imsave(cur_norm_path, cur_norm_block)
