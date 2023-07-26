# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math, csv, pickle
import shutil, copy
import numpy as np
import pandas as pd
from skimage import io
import tifffile
import cv2
from spatialentropy import leibovici_entropy, altieri_entropy

def set_args():
    parser = argparse.ArgumentParser(description = "Extract ROI stage-wise features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--slide_roi_dir",    type=str,       default="SlidesROIs")
    parser.add_argument("--cell_fea_dir",     type=str,       default="CellFeas") 
    parser.add_argument("--roi_seg_dir",      type=str,       default="RegionSegs")         
    parser.add_argument("--roi_cellmask_dir", type=str,       default="CellMasks")  
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA", "China"])    
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)

    # roi & stage
    ROIs, Stages = [], []
    # cell ratio features
    aec_rs, lym_rs, oc_rs = [], [], []
    # cell density features
    aec_ds, lym_ds, oc_ds = [], [], []
    # lesion entropies
    altieri2_entropies, altieri3_entropies = [], []    

    roi_data_root = os.path.join(args.data_root, args.slide_roi_dir, args.dataset)
    # obtain lesion stage information
    lesion_stage_dict = {}
    lesion_stage_path = os.path.join(roi_data_root, "{}LesionStages.json".format(args.dataset))
    with open(lesion_stage_path) as fp:
        lesion_stage_dict = json.load(fp)    
    # traverse all ROIs
    cell_fea_dir = os.path.join(roi_data_root, args.cell_fea_dir)
    roi_list = [os.path.splitext(ele)[0] for ele in os.listdir(cell_fea_dir) if ele.endswith(".csv")]
    roi_seg_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.roi_seg_dir)
    cellmask_dir = os.path.join(roi_data_root, args.roi_cellmask_dir)

    # organize ROI features
    for ind, ele in enumerate(roi_list):
        print("Extract features on {}/{}".format(ind+1, len(roi_list)))
        # add meta information
        ROIs.append(ele)
        Stages.append(lesion_stage_dict[ele])
        # load information
        cell_fea_path = os.path.join(cell_fea_dir, ele + ".csv") 
        cell_fea_df = pd.read_csv(cell_fea_path)
        cell_ids = cell_fea_df["ID"]
        cell_labels = cell_fea_df["Label"]
       # load segmentation
        roi_seg_path = os.path.join(roi_seg_dir, ele + ".json")
        cur_roi_seg = {}
        with open(roi_seg_path, 'r') as fp:
            cur_roi_seg = json.load(fp)
        if len(cell_ids) != len(cur_roi_seg):
            print("{} - cell number not match in dataset {}.".format(ele, args.dataset))
            sys.exit()
        # cell ratios
        aec_ratio = np.sum(cell_labels == 0) * 1.0 / len(cell_labels)
        lym_ratio = np.sum(cell_labels == 1) * 1.0 / len(cell_labels)
        oc_ratio = np.sum(cell_labels == 2) * 1.0 / len(cell_labels)
        aec_rs.append(aec_ratio)
        lym_rs.append(lym_ratio)
        oc_rs.append(oc_ratio)
        # cell densties
        cur_cellmask_path = os.path.join(cellmask_dir, ele + ".tiff")
        cur_cell_mask = io.imread(cur_cellmask_path)
        pixel_num = cur_cell_mask.shape[0] * cur_cell_mask.shape[1]
        aec_density = np.sum(cell_labels == 0) * 4.0 / pixel_num
        lym_density = np.sum(cell_labels == 1) * 4.0 / pixel_num
        oc_density = np.sum(cell_labels == 2) * 4.0 / pixel_num
        aec_ds.append(aec_density)
        lym_ds.append(lym_density)
        oc_ds.append(oc_density)

        # collect cell centroids information
        cell_centroids2, cell_types2 = [], []
        cell_centroids3, cell_types3 = [], []
        for cell_id, cell_label in zip(cell_ids, cell_labels):
            cell_cnt = cur_roi_seg[str(cell_id)]["contour"]
            # cell_cnt = np.expand_dims(np.asarray(cell_cnt), axis=1)
            # x, y, _, _ = cv2.boundingRect(cell_cnt)
            cen_x, cen_y = np.mean(np.asarray(cell_cnt), axis=0)
            # add cell & locations
            cell_centroids3.append([cen_y, cen_x])
            cell_types3.append(cell_label)
            if cell_label != 2:
                cell_centroids2.append([cen_y, cen_x])
                cell_types2.append(cell_label)
        # calculate altieri2_entropy
        cell_centroids2 = np.asarray(cell_centroids2).astype(np.float64)
        cell_types2 = np.asarray(cell_types2).astype(np.int64)
        lesion_altieri2 = altieri_entropy(cell_centroids2, cell_types2, cut=[30, 60, 100])
        altieri2_entropies.append(lesion_altieri2.entropy) 
        # calculate altieri3_entropy
        cell_centroids3 = np.asarray(cell_centroids3).astype(np.float64)
        cell_types3 = np.asarray(cell_types3).astype(np.int64)
        lesion_altieri3 = altieri_entropy(cell_centroids3, cell_types3, cut=[30, 60, 100])
        altieri3_entropies.append(lesion_altieri3.entropy)                 
 
    # save features
    fea_list = list(zip(ROIs, Stages, aec_rs, lym_rs, oc_rs, aec_ds, lym_ds, oc_ds, altieri2_entropies, altieri3_entropies))
    fea_names = ["Lesions", "Stages", "AEC-Proportion", "LYM-Proportion", "OC-Proportion", "AEC-Density", "LYM-Density", "OC-Density", "Altieri2-Entropy", "Altieri3-Entropy"]
    roi_fea_df = pd.DataFrame(fea_list, columns = fea_names)
    roi_fea_path = os.path.join(roi_data_root, "{}LesionFeatures.csv".format(args.dataset))
    roi_fea_df.to_csv(roi_fea_path, index=False)