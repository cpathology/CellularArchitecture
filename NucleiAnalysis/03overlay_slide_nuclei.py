# -*- coding: utf-8 -*-

import os, sys
import shutil, argparse, pytz, pickle, json
import numpy as np
import openslide
import cv2
from scipy import ndimage
from skimage import io, color, morphology, measure, filters, img_as_ubyte

from nuclei_utils import numpy2tiff

def set_args():
    parser = argparse.ArgumentParser(description = "Overlay classified nuclei onto slide")
    parser.add_argument("--data_root",         type=str,       default="/Data")
    parser.add_argument("--core_slide_dir",    type=str,       default="CoreSlides")
    parser.add_argument("--core_cls_dir",      type=str,       default="CoreNucleiCls")  
    parser.add_argument("--core_overlay_dir",  type=str,       default="CoreOverlay")   
    parser.add_argument("--dataset",           type=str,       default="ICON", choices=["ICON", "Immuno"])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    type_color_dict = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}    

    # directory setting
    dataset_root = os.path.join(args.data_root, args.dataset)
    slide_core_dir = os.path.join(dataset_root, args.core_slide_dir)
    slide_nuclei_cls_dir = os.path.join(dataset_root, args.core_cls_dir)
    core_overlay_dir = os.path.join(dataset_root, args.core_overlay_dir)
    if not os.path.exists(core_overlay_dir):
        os.makedirs(core_overlay_dir)

    core_lst = sorted([os.path.splitext(ele)[0] for ele in os.listdir(slide_core_dir) if ele.endswith(".tiff")])
    # Traverse slide one-by-one
    for ind, cur_core in enumerate(core_lst):
        print("Overlay nuclei on {}/{} on {}".format(ind+1, len(core_lst), cur_core))    
        slide_nuclei_path = os.path.join(slide_nuclei_cls_dir, cur_core + ".json")
        if not os.path.exists(slide_nuclei_path):
            print("{} classification not finished yet.".format(cur_core))
            continue
        
        # load slide
        cur_core_path = os.path.join(slide_core_dir, cur_core + ".tiff")
        cur_core_head = openslide.OpenSlide(cur_core_path)     
        cur_core_img = cur_core_head.read_region(location=(0, 0), level=0, size=cur_core_head.level_dimensions[0])     
        cur_core_img = np.ascontiguousarray(np.asarray(cur_core_img)[:,:,:3], dtype=np.uint8)

        # overlay nuclei
        nuclei_inst_dict = json.load(open(slide_nuclei_path, "r"))
        line_thickness = -1
        for idx, [inst_id, inst_info] in enumerate(nuclei_inst_dict.items()):
            inst_contour = np.expand_dims(np.array(inst_info["contour"]), axis=1)
            inst_type = inst_info["label"]
            inst_color = type_color_dict[inst_type]
            cv2.drawContours(cur_core_img, [inst_contour, ], 0, inst_color, line_thickness)
        
        # save overlay as BigTIFF image
        big_tif_path = os.path.join(core_overlay_dir, cur_core + ".tif")
        numpy2tiff(cur_core_img, big_tif_path)
        # convert the tif image to pyramid tiff
        pyramid_tif_path = os.path.join(core_overlay_dir, cur_core + ".tiff")
        conversion_cmd_str = " ".join(["vips", "im_vips2tiff", big_tif_path, pyramid_tif_path + ":jpeg:75,tile:256x256,pyramid"])
        os.system(conversion_cmd_str)
        os.remove(big_tif_path)        