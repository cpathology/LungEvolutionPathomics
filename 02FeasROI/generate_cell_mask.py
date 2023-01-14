# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math, csv
import collections, pickle
import shutil, copy
import pandas as pd
import numpy as np
from skimage import io, color
import cv2
import tifffile

from seg_utils import bounding_box


def set_args():
    parser = argparse.ArgumentParser(description = "Extract ROI cell features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--slide_roi_dir",    type=str,       default="SlidesROIs")
    parser.add_argument("--roi_img_dir",      type=str,       default="MacenkoROIs")
    parser.add_argument("--roi_seg_dir",      type=str,       default="RegionSegs")  
    parser.add_argument("--roi_cellmask_dir", type=str,       default="CellMasks")  
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA", "China"])    
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)    

    roi_img_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.roi_img_dir)
    roi_seg_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.roi_seg_dir)
    roi_cellmask_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.roi_cellmask_dir)
    if os.path.exists(roi_cellmask_dir):
        shutil.rmtree(roi_cellmask_dir)
    os.makedirs(roi_cellmask_dir)    


    print("="*80)
    print("****Generate cell mask for each ROI****")
    print("="*80)    
    # traverse all ROIs
    roi_list = [os.path.splitext(ele)[0] for ele in os.listdir(roi_img_dir) if ele.endswith(".png")]
    for ind, cur_roi in enumerate(roi_list):
        if (ind + 1) % 10 == 0:
            print("Extract {}/{}".format(ind+1, len(roi_list)))
        # load image
        roi_img_path = os.path.join(roi_img_dir, cur_roi + ".png")
        roi_img = io.imread(roi_img_path)
        # load segmentation
        roi_seg_path = os.path.join(roi_seg_dir, cur_roi + ".json")
        cur_roi_seg = {}
        with open(roi_seg_path, 'r') as fp:
            cur_roi_seg = json.load(fp)

        # initate region mask
        roi_cell_mask = np.zeros((roi_img.shape[0], roi_img.shape[1]), dtype = np.uint32)
        for ind, key in enumerate(cur_roi_seg.keys()):
            cell_id = int(key)
            if cell_id != ind + 1:
                sys.exit("Key information not matching")
            cell_cnt = cur_roi_seg[key]["contour"]
            cell_cnt = np.expand_dims(np.asarray(cell_cnt), axis=1)
            x, y, w, h = cv2.boundingRect(cell_cnt)
            nuc_mask = np.zeros((h, w), dtype=np.uint8)
            cell_cnt[:,0,0] -= x
            cell_cnt[:,0,1] -= y
            cv2.drawContours(nuc_mask, contours=[cell_cnt, ], contourIdx=0, color=1, thickness=-1)
            roi_cell_mask[y:y+h, x:x+w] += nuc_mask * cell_id
        if len(np.unique(roi_cell_mask)) != len(cur_roi_seg.keys()) + 1:
            sys.exit("Value issues exist in generated cell mask")
        roi_cell_mask_path = os.path.join(roi_cellmask_dir, cur_roi + ".tiff")
        tifffile.imwrite(roi_cell_mask_path, roi_cell_mask)       