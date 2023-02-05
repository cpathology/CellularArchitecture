# -*- coding: utf-8 -*-

import os, sys
import shutil, argparse, pytz, pickle, json
import math
import numpy as np
import openslide
import cv2
from scipy import ndimage
from skimage import io, color, morphology, measure, filters, img_as_ubyte


def set_args():
    parser = argparse.ArgumentParser(description = "Generate WSI Embedded Maps")
    parser.add_argument("--data_root",         type=str,       default="/Data")
    parser.add_argument("--core_slide_dir",    type=str,       default="CoreSlides")
    parser.add_argument("--slide_cls_dir",     type=str,       default="CoreNucleiCls")
    parser.add_argument("--embed_map_dir",     type=str,       default="CoreEmbedMap")    
    parser.add_argument("--cell_type_num",     type=int,       default=3)
    parser.add_argument("--tissue_slide_level",type=int,       default=6)
    parser.add_argument("--dataset",           type=str,       default="ICON", choices=["ICON", "Immuno"])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    cell_id_dict = {"AEC": 0, "LYM": 1, "OC": 2}
    # directory setting
    dataset_root = os.path.join(args.data_root, args.dataset)
    slide_core_dir = os.path.join(dataset_root, args.core_slide_dir)  
    slide_nuclei_cls_dir = os.path.join(dataset_root, args.slide_cls_dir)
    slide_nuclei_map_dir = os.path.join(dataset_root, args.embed_map_dir)
    if not os.path.exists(slide_nuclei_map_dir):
        os.makedirs(slide_nuclei_map_dir)

    core_lst = sorted([os.path.splitext(ele)[0] for ele in os.listdir(slide_core_dir) if ele.endswith(".tiff")])
    # Traverse slide one-by-one
    for ind, cur_core in enumerate(core_lst):
        print("Embed nuclei on {}/{} on {}".format(ind+1, len(core_lst), cur_core))
        cur_core_path = os.path.join(slide_core_dir, cur_core + ".tiff")
        cur_core_head = openslide.OpenSlide(cur_core_path)     
        core_w, core_h = cur_core_head.level_dimensions[args.tissue_slide_level]
        # initiate embed map
        divide_ratio = np.power(2, args.tissue_slide_level)
        core_embed_map = np.zeros((core_h, core_w, args.cell_type_num), dtype=np.uint8)   
        # load slide nuclei
        slide_nuclei_path = os.path.join(slide_nuclei_cls_dir, cur_core + ".json")
        if not os.path.exists(slide_nuclei_path):
            continue
        nuclei_cls_dict = None
        with open(slide_nuclei_path) as f:
            nuclei_cls_dict = json.load(f)
        nuc_ids = [ele for ele in nuclei_cls_dict.keys()]
        for nuc_id in nuc_ids:
            nuc_label = nuclei_cls_dict[nuc_id]["label"]
            nuc_slide_x, nuc_slide_y = nuclei_cls_dict[nuc_id]["centroid"]
            nuc_embed_x = int(math.floor(nuc_slide_x * 1.0 / divide_ratio))
            nuc_embed_x = nuc_embed_x if nuc_embed_x < core_w else core_w - 1
            nuc_embed_y = int(math.floor(nuc_slide_y * 1.0 / divide_ratio))
            nuc_embed_y = nuc_embed_y if nuc_embed_y < core_h else core_h - 1
            core_embed_map[nuc_embed_y, nuc_embed_x, nuc_label] += 1
        # Save to embed map
        embed_map_path = os.path.join(slide_nuclei_map_dir, cur_core + ".png")
        io.imsave(embed_map_path, core_embed_map)
