# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math, csv, pytz
from datetime import datetime
import collections, pickle
import shutil, copy
import pandas as pd
import numpy as np
from skimage import io, color, filters
import cv2


def set_args():
    parser = argparse.ArgumentParser(description = "Extract lesion cellular ratio  features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--slide_roi_dir",    type=str,       default="SlidesROIs")
    parser.add_argument("--cell_fea_dir",     type=str,       default="CellFeas") 
    parser.add_argument("--roi_seg_dir",      type=str,       default="RegionSegs")     
    parser.add_argument("--embed_dir",        type=str,       default="EmbedMaps")
    parser.add_argument("--reduction_size",   type=int,       default=50)    
    parser.add_argument("--img_coef",         type=int,       default=320)    
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA", "China"]) 
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)

    roi_data_root = os.path.join(args.data_root, args.slide_roi_dir, args.dataset)
    # Embed directory
    roi_embed_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.embed_dir)
    if os.path.exists(roi_embed_dir):
        shutil.rmtree(roi_embed_dir)
    os.makedirs(roi_embed_dir)         
    # traverse all ROIs
    cell_fea_dir = os.path.join(roi_data_root, args.cell_fea_dir)
    roi_list = [os.path.splitext(ele)[0] for ele in os.listdir(cell_fea_dir) if ele.endswith(".csv")]
    roi_seg_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.roi_seg_dir) 

    # organize ROI features
    for ind, ele in enumerate(roi_list):
        print("Embed on {}/{}".format(ind+1, len(roi_list)))
        # initialize embed map size
        w_loc = ele.find("Wlen")
        roi_w = int(ele[w_loc+4:w_loc+10])
        h_loc = ele.find("Hlen")
        roi_h = int(ele[h_loc+4:h_loc+10])
        embed_width = int(np.floor((0.5 + roi_w / args.reduction_size)) + 1)
        embed_height = int(np.floor((0.5 + roi_h / args.reduction_size)) + 1)
        embed_map = np.zeros((embed_height, embed_width, 3), dtype=np.float32)

        # load information
        cell_fea_path = os.path.join(cell_fea_dir, ele + ".csv") 
        cell_fea_df = pd.read_csv(cell_fea_path)
        cell_ids = cell_fea_df["ID"]
        cell_labels = cell_fea_df["Label"]
       # load segmentation
        roi_seg_path = os.path.join(roi_seg_dir, ele + ".json")
        cur_roi_seg = {}
        with open(roi_seg_path, "r") as fp:
            cur_roi_seg = json.load(fp)
        if len(cell_ids) != len(cur_roi_seg):
            print("{} - cell number not match in dataset {}.".format(ele, args.dataset))
            sys.exit()

        # fill the embed map
        for cell_id, cell_label in zip(cell_ids, cell_labels):
            cell_cnt = cur_roi_seg[str(cell_id)]["contour"]
            cen_x, cen_y = np.mean(np.asarray(cell_cnt), axis=0)
            cen_x = int(math.floor(cen_x * 1.0 / args.reduction_size))
            cen_y = int(math.floor(cen_y * 1.0 / args.reduction_size))
            embed_map[cen_y, cen_x, cell_label] += 1.0 / (args.reduction_size * args.reduction_size)

        # Convert to heatmap image
        embed_map = embed_map * args.img_coef
        embed_map[embed_map > 1.0] = 1.0
        embed_map = (embed_map * 255.0).astype(np.uint8)
        embed_map_path = os.path.join(roi_embed_dir, ele + ".png")
        io.imsave(embed_map_path, embed_map)