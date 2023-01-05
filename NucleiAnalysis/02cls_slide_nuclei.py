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
    parser.add_argument("--block_root_dir",    type=str,       default="BlockAnalysis")
    parser.add_argument("--block_img_dir",     type=str,       default="SlideBlocks")
    parser.add_argument("--block_norm_dir",    type=str,       default="BlockNorms")    
    parser.add_argument("--slide_seg_dir",     type=str,       default="SlideSegs") 
    parser.add_argument("--slide_cls_dir",     type=str,       default="SlideNucleiCls")    
    parser.add_argument("--slide_tissue_dir",  type=str,       default="SlideTissues")
    parser.add_argument("--tissue_slide_level",type=int,       default=3)  
    parser.add_argument("--checkpoint_dir",    type=str,       default="Checkpoints")    
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
    block_img_dir = os.path.join(dataset_root, args.block_root_dir, args.block_img_dir)
    block_norm_dir = os.path.join(dataset_root, args.block_root_dir, args.block_norm_dir)
    slide_nuclei_cls_dir = os.path.join(dataset_root, args.slide_cls_dir)
    if not os.path.exists(slide_nuclei_cls_dir):
        os.makedirs(slide_nuclei_cls_dir)

    # nuclei classificaiton model setting
    checkpoint_dir = os.path.join(args.data_root, args.checkpoint_dir, "NucleiCls")
    nuclei_cls_model_path = os.path.join(checkpoint_dir, "lung_cell_classifier.model")  
    nuclei_clf = pickle.load(open(nuclei_cls_model_path, "rb"))  

    core_lst = sorted([os.path.splitext(ele)[0] for ele in os.listdir(slide_core_dir) if ele.endswith(".tiff")])
    # Traverse slide one-by-one
    for ind, cur_core in enumerate(core_lst):
        print("Classify nuclei on {}/{} on {}".format(ind+1, len(core_lst), cur_core))
        core_nuclei_seg_path = os.path.join(slide_seg_dir, cur_core + ".json")
        if not os.path.exists(core_nuclei_seg_path):
            print("{} segementation not not finished yet.".format(cur_core))
            continue            
        core_block_img_dir = os.path.join(block_img_dir, cur_core)
        num_block_img = len([ele for ele in os.listdir(core_block_img_dir) if ele.endswith(".png")])
        core_block_norm_dir = os.path.join(block_norm_dir, cur_core)  
        if not os.path.exists(core_block_norm_dir):
            continue
        norm_img_lst = [ele for ele in os.listdir(core_block_norm_dir) if ele.endswith(".png")]
        num_norm_img = len(norm_img_lst)
        if num_norm_img != num_block_img:
            print("{} image normalization not finished yet.".format(cur_core))
            continue
        # load stain-normed slide
        cur_core_path = os.path.join(slide_core_dir, cur_core + ".tiff")
        cur_core_head = openslide.OpenSlide(cur_core_path)     
        core_w, core_h = cur_core_head.level_dimensions[0]
        core_img_arr = np.zeros((core_h, core_w, 3), dtype=np.uint8)   
        for cur_norm_name in norm_img_lst:
            wstart_pos = cur_norm_name.find("Wstart")
            hstart_pos = cur_norm_name.find("Hstart")
            wlen_pos = cur_norm_name.find("Wlen")
            hlen_pos = cur_norm_name.find("Hlen")
            h_start = int(cur_norm_name[hstart_pos+6:hstart_pos+6+5])
            w_start = int(cur_norm_name[wstart_pos+6:wstart_pos+6+5])
            h_len = int(cur_norm_name[hlen_pos+4:hlen_pos+4+5])
            w_len = int(cur_norm_name[wlen_pos+4:wlen_pos+4+5])
            cur_norm_path = os.path.join(core_block_norm_dir, cur_norm_name)
            cur_norm_img = io.imread(cur_norm_path)
            core_img_arr[h_start:h_start+h_len, w_start:w_start+w_len] = cur_norm_img
        # load tissue_mask
        tissue_mask_path = os.path.join(slide_tissue_dir, cur_core + "-L" + str(args.tissue_slide_level) + "-Tissue.png")
        core_tissue_mask = io.imread(tissue_mask_path)
        divide_ratio = np.power(2, args.tissue_slide_level)
        lowlevel_w, lowlevel_h = cur_core_head.level_dimensions[args.tissue_slide_level]

        # load segmentation
        core_seg_dict = None
        with open(core_nuclei_seg_path) as f:
            core_seg_dict = json.load(f)
        core_nuc_dict = core_seg_dict["nuc"]
        # collect eligible nuclei
        cls_seg_dict = {}
        nuclei_fea = []
        cls_num = 1
        for key in core_nuc_dict.keys():
            cell_x, cell_y = core_nuc_dict[key]["centroid"]
            map_x = int(np.floor(cell_x / divide_ratio))
            map_y = int(np.floor(cell_y / divide_ratio))    
            if map_x == lowlevel_w:
                continue
            if map_y == lowlevel_h:
                continue
            if core_tissue_mask[map_y, map_x] == 0:
                continue
            # collect features
            cell_cnt = core_nuc_dict[key]["contour"]
            cell_cnt = np.expand_dims(np.asarray(cell_cnt), axis=1)
            # Area            
            cell_area = cv2.contourArea(cell_cnt)
            # Intensity 
            x, y, w, h = cv2.boundingRect(cell_cnt)
            nuc_mask = np.zeros((h, w), dtype=np.uint8)
            cell_cnt[:,0,0] -= x
            cell_cnt[:,0,1] -= y
            cv2.drawContours(nuc_mask, contours=[cell_cnt, ], contourIdx=0, color=1, thickness=-1)
            nuc_img = core_img_arr[y:y+h, x:x+w]
            cell_intensity = np.mean(cv2.mean(nuc_img, mask=nuc_mask)[:3])
            # Roundness
            cnt_perimeter = cv2.arcLength(cell_cnt, True)
            cell_roundness = 4 * 3.14 * cell_area / (cnt_perimeter * cnt_perimeter)
            nuclei_fea.append([cell_area, cell_intensity, cell_roundness])
            nuclei_cls_dict = {}
            nuclei_cls_dict["centroid"] = core_nuc_dict[key]["centroid"]
            nuclei_cls_dict["contour"] = core_nuc_dict[key]["contour"]
            cls_seg_dict[str(cls_num)] = nuclei_cls_dict
            cls_num += 1
        nuclei_fea = np.asarray(nuclei_fea)
        nuclei_labels = nuclei_clf.predict(nuclei_fea)
        nuclei_labels = [int(label) for label in nuclei_labels]
        for ind, label in enumerate(nuclei_labels):
            cls_seg_dict[str(ind+1)]["label"] = label
        # save json file
        slide_nuclei_path = os.path.join(slide_nuclei_cls_dir, cur_core + ".json")
        with open(slide_nuclei_path, 'w') as fp:
            json.dump(cls_seg_dict, fp)        




