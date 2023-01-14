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
    parser.add_argument("--roi_img_dir",      type=str,       default="MacenkoROIs")
    parser.add_argument("--roi_seg_dir",      type=str,       default="RegionSegs")
    parser.add_argument("--cell_fea_dir",     type=str,       default="CellFeas")
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA", "China"])  
    parser.add_argument("--celltype_dir",     type=str,       default="CellType") 
    parser.add_argument("--min_cell_num",     type=int,       default=10)  
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)    

    roi_img_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.roi_img_dir)
    roi_seg_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.roi_seg_dir)
    cell_fea_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.cell_fea_dir)
    if os.path.exists(cell_fea_dir):
        shutil.rmtree(cell_fea_dir)
    os.makedirs(cell_fea_dir)    

    # load cell classification model
    celltype_model_path = os.path.join(args.data_root, args.celltype_dir, "CellModels", "fusing_cell_classifier.model")
    cell_clf = pickle.load(open(celltype_model_path, "rb"))

    print("="*80)
    print("****Start cell feature extraction for each ROI****")
    print("="*80)    
    # traverse all ROIs
    roi_list = [os.path.splitext(ele)[0] for ele in os.listdir(roi_img_dir) if ele.endswith(".png")]
    for ind, cur_roi in enumerate(roi_list):
        if (ind + 1) % 100 == 0:
            print("Extract {}/{}".format(ind+1, len(roi_list)))
        # load image
        roi_img_path = os.path.join(roi_img_dir, cur_roi + ".png")
        roi_img = io.imread(roi_img_path)
        # load segmentation
        roi_seg_path = os.path.join(roi_seg_dir, cur_roi + ".json")
        cur_roi_seg = {}
        with open(roi_seg_path, 'r') as fp:
            cur_roi_seg = json.load(fp)

        # extract features
        cell_list, area_list, intensity_list, roundness_list = [], [], [], []
        for key in cur_roi_seg.keys():
            cell_cnt = cur_roi_seg[key]["contour"]
            cell_cnt = np.expand_dims(np.asarray(cell_cnt), axis=1)
            # Area            
            cell_area = cv2.contourArea(cell_cnt)
            # Intensity 
            x, y, w, h = cv2.boundingRect(cell_cnt)
            nuc_mask = np.zeros((h, w), dtype=np.uint8)
            cell_cnt[:,0,0] -= x
            cell_cnt[:,0,1] -= y
            cv2.drawContours(nuc_mask, contours=[cell_cnt, ], contourIdx=0, color=1, thickness=-1)
            nuc_img = roi_img[y:y+h, x:x+w]
            cell_intensity = np.mean(cv2.mean(nuc_img, mask=nuc_mask)[:3])
            # Roundness
            cnt_perimeter = cv2.arcLength(cell_cnt, True)
            cell_roundness = 4 * 3.14 * cell_area / (cnt_perimeter * cnt_perimeter)
            
            # add cell information
            cell_list.append(key)
            area_list.append(cell_area)
            intensity_list.append(cell_intensity)
            roundness_list.append(cell_roundness)
        
        cell_fea_list = list(zip(cell_list, area_list, intensity_list, roundness_list))
        fea_names = ["ID", "Area", "Intensity", "Roundness"]
        cell_fea_df = pd.DataFrame(cell_fea_list, columns = fea_names)
        # predict cell labels
        cell_feas = cell_fea_df[["Area", "Intensity", "Roundness"]].to_numpy().astype(np.float64)
        if len(cell_fea_df) < args.min_cell_num:
            print("{} has too less cells detected".format(cur_roi))
            continue
        cell_labels = cell_clf.predict(cell_feas)
        cell_fea_df["Label"] = cell_labels.tolist()

        # save feature
        cell_fea_path = os.path.join(cell_fea_dir, cur_roi + ".csv")
        cell_fea_df.to_csv(cell_fea_path, index=False)        