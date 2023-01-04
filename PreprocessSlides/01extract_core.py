# -*- coding: utf-8 -*-

import os, sys
import shutil, argparse, pytz
import openslide
from skimage import io
import numpy as np
import PIL
from PIL import Image
PIL.Image.MAX_IMAGE_PIXELS = 933120000

from preprocess_utils import parse_imagescope_rects, get_splitting_coors, numpy2tiff


def set_args():
    parser = argparse.ArgumentParser(description = "Crop slide & convert core to pyramidal tiff")
    parser.add_argument("--data_root",         type=str,       default="/Data")
    parser.add_argument("--raw_slide_dir",     type=str,       default="RawSlides")
    parser.add_argument("--core_slide_dir",    type=str,       default="CoreSlides")
    parser.add_argument("--slide_suffix",      type=str,       default="svs", choices=["svs", "tif"])
    parser.add_argument("--dataset",           type=str,       default="ICON", choices=["ICON", ])    
    parser.add_argument("--block_size",        type=int,       default=6400)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    raw_slide_dir = os.path.join(args.data_root, args.dataset, args.raw_slide_dir)
    core_slide_dir = os.path.join(args.data_root, args.dataset, args.core_slide_dir)
    if not os.path.exists(core_slide_dir):
        os.makedirs(core_slide_dir)

    raw_slide_list = sorted([ele for ele in os.listdir(raw_slide_dir) if ele.endswith(args.slide_suffix)])
    print("There are {} slides.".format(len(raw_slide_list)))
    core_num = 0
    for ind, slide_name in enumerate(raw_slide_list):
        cur_slide_name = os.path.splitext(slide_name)[0]
        # parse annotation
        cur_slide_anno = cur_slide_name + ".xml"
        cur_anno_path = os.path.join(raw_slide_dir, cur_slide_anno)
        if not os.path.exists(cur_anno_path):
            continue
        else:
            core_num += 1
        # check core slide exist or not
        core_slide_path = os.path.join(core_slide_dir, cur_slide_name + ".tiff")
        if os.path.exists(core_slide_path):
            continue        

        print("Extract {}/{} with slide name {}".format(ind+1, len(raw_slide_list), cur_slide_name))
        # parse core rect
        roi_boxes = parse_imagescope_rects(cur_anno_path)
        if len(roi_boxes) != 1:
            print("Number of box is: {}".format(len(roi_boxes)))
        (w_start, h_start), (w_len, h_len) = roi_boxes[0]
        core_img_arr = np.zeros((h_len, w_len, 3), dtype=np.uint8)
        
        # block-wise load
        cur_slide_path = os.path.join(raw_slide_dir, slide_name)
        cur_slide_head = openslide.OpenSlide(cur_slide_path)
        coors_list = get_splitting_coors(w_len, h_len, args.block_size)
        for block_info in coors_list:
            block_x, block_y, block_w, block_h = block_info
            cur_block = cur_slide_head.read_region(location=(w_start+block_x, h_start+block_y), level=0, size=(block_w, block_h))
            cur_block = np.asarray(cur_block)[:,:,:3]
            core_img_arr[block_y:block_y+block_h, block_x:block_x+block_w, :] = cur_block
        core_img_path = os.path.join(core_slide_dir, cur_slide_name + ".tif")
        if os.path.exists(core_img_path):
            os.remove(core_img_path)  
        numpy2tiff(core_img_arr, core_img_path)

        # convert the tif image to pyramid tiff
        conversion_cmd_str = " ".join(["vips", "im_vips2tiff", core_img_path, core_slide_path + ":jpeg:75,tile:256x256,pyramid"])
        os.system(conversion_cmd_str) 
        if os.path.exists(core_img_path):
            os.remove(core_img_path)
    print("{}/{} core cropped.".format(core_num, len(raw_slide_list)))