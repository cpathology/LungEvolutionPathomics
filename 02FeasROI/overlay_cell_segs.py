# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math, csv
import collections, pickle
import shutil, copy
import pandas as pd
import numpy as np
from skimage import io, color
import cv2

from seg_utils import bounding_box


def set_args():
    parser = argparse.ArgumentParser(description = "Extract ROI cell features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--slide_roi_dir",    type=str,       default="SlidesROIs")
    parser.add_argument("--block_dir",        type=str,       default="RegionROIs")
    parser.add_argument("--roi_seg_dir",      type=str,       default="RegionSegs")
    parser.add_argument("--cell_fea_dir",     type=str,       default="CellFeas")
    parser.add_argument("--cell_overlay_dir", type=str,       default="OverlayCellSeg") 
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA", "China"])
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)    

    roi_img_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.block_dir)
    roi_seg_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.roi_seg_dir)
    cell_fea_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.cell_fea_dir)
    cell_overlay_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.cell_overlay_dir)
    if os.path.exists(cell_overlay_dir):
        shutil.rmtree(cell_overlay_dir)
    os.makedirs(cell_overlay_dir) 

    print("="*80)
    print("****Start overlaying cells to each ROI****")
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
        # load cell   
        cell_fea_df = None
        cell_fea_path = os.path.join(cell_fea_dir, cur_roi + ".csv")
        cell_fea_df = pd.read_csv(cell_fea_path)
        cell_id_lst, cell_label_lst = cell_fea_df["ID"].tolist(), cell_fea_df["Label"].tolist()
        id_label_dict = {str(cid): label for cid, label in zip(cell_id_lst, cell_label_lst)}

        # type_color = {0: [255, 0, 0], 1: [0, 255, 0], 2: [0, 0, 255]}
        for key in cur_roi_seg.keys():
            cell_cnt = cur_roi_seg[key]["contour"]
            cell_cnt = np.expand_dims(np.asarray(cell_cnt), axis=1)
            cell_type = id_label_dict[key]
            cv2.drawContours(roi_img, contours=[cell_cnt, ], contourIdx=0, color=(23,190,207), thickness=1)
        overlay_img_path = os.path.join(cell_overlay_dir, cur_roi + ".png")
        io.imsave(overlay_img_path, roi_img)     