# -*- coding: utf-8 -*-

import os, sys
import shutil, argparse
import math, pytz, pickle, json
import numpy as np
import openslide
import cv2
from skimage import io
import mahotas.features.texture as mht
import pandas as pd


def set_args():
    parser = argparse.ArgumentParser(description = "Generate WSI Embedded Maps")
    parser.add_argument("--data_root",         type=str,       default="/Data")
    parser.add_argument("--core_slide_dir",    type=str,       default="CoreSlides")
    parser.add_argument("--embed_map_dir",     type=str,       default="CoreEmbedMap")
    parser.add_argument("--core_fea_dir",      type=str,       default="CoreFeas")  
    parser.add_argument("--dataset",           type=str,       default="ICON", choices=["ICON", "Immuno"])

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    # directory setting
    dataset_root = os.path.join(args.data_root, args.dataset)
    slide_core_dir = os.path.join(dataset_root, args.core_slide_dir)  
    slide_embed_dir = os.path.join(dataset_root, args.embed_map_dir)
    core_fea_dir = os.path.join(dataset_root, args.core_fea_dir)
    if not os.path.exists(core_fea_dir):
        os.makedirs(core_fea_dir)

    core_lst = sorted([os.path.splitext(ele)[0] for ele in os.listdir(slide_core_dir) if ele.endswith(".tiff")])
    # Traverse slide one-by-one
    slide_lst = []
    aec_correlation_lst, aec_contrast_lst, aec_entropy_lst = [], [], []
    lym_correlation_lst, lym_contrast_lst, lym_entropy_lst = [], [], []
    for ind, cur_core in enumerate(core_lst):
        print("Extract features on {}/{} on {}".format(ind+1, len(core_lst), cur_core))
        slide_lst.append(cur_core)
        # load embed map
        embed_map_path = os.path.join(slide_embed_dir, cur_core + ".png")
        slide_embed_map = io.imread(embed_map_path)
        aec_map = slide_embed_map[:,:,0]
        aec_haralick_fea = mht.haralick(aec_map, ignore_zeros=True, compute_14th_feature=False, return_mean=True)
        aec_correlation_lst.append(aec_haralick_fea[2])
        aec_contrast_lst.append(aec_haralick_fea[1])
        aec_entropy_lst.append(aec_haralick_fea[8])
        lym_map = slide_embed_map[:,:,1]
        lym_haralick_fea = mht.haralick(lym_map, ignore_zeros=True, compute_14th_feature=False, return_mean=True)
        lym_correlation_lst.append(lym_haralick_fea[2])
        lym_contrast_lst.append(lym_haralick_fea[1])
        lym_entropy_lst.append(lym_haralick_fea[8]) 
    zip_lst = list(zip(slide_lst, aec_correlation_lst, aec_contrast_lst, aec_entropy_lst, 
                       lym_correlation_lst, lym_contrast_lst, lym_entropy_lst))
    pathomics_df = pd.DataFrame(zip_lst, columns = ["Slide", "AEC-Correlation", "AEC-Contrast", "AEC-Entropy", 
                                                    "LYM-Correlation", "LYM-Contrast", "LYM-Entropy"])
    core_fea_path = os.path.join(core_fea_dir, "PathomicFeatures.csv")
    pathomics_df.to_csv(core_fea_path, index=False)    
    