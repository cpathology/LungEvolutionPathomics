# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math, csv, pytz
from datetime import datetime
import collections, pickle
import shutil, copy
import pandas as pd
import numpy as np
from skimage import io, color, filters
import matplotlib.pyplot as plt
import cv2

def set_args():
    parser = argparse.ArgumentParser(description = "Extract lesion cellular ratio  features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--slide_roi_dir",    type=str,       default="SlidesROIs")    
    parser.add_argument("--embed_dir",        type=str,       default="EmbedMaps")
    parser.add_argument("--heat_dir",         type=str,       default="HeatMaps")
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA", "China"]) 
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)

    roi_data_root = os.path.join(args.data_root, args.slide_roi_dir, args.dataset)
    # Embed directory
    roi_embed_dir = os.path.join(roi_data_root, args.embed_dir)
    heatmap_dir = os.path.join(roi_data_root, args.heat_dir)
    if os.path.exists(heatmap_dir):
        shutil.rmtree(heatmap_dir)
    os.makedirs(heatmap_dir)   

    # traverse all ROIs
    roi_list = [os.path.splitext(ele)[0] for ele in os.listdir(roi_embed_dir) if ele.endswith(".png")]

    # organize ROI features
    for ind, ele in enumerate(roi_list):
        print("Heatmap on {}/{}".format(ind+1, len(roi_list)))
        embed_map_path = os.path.join(roi_embed_dir, ele + ".png")
        embed_map = io.imread(embed_map_path)
        aec_map = embed_map[:,:,0]
        aec_heatmap = cv2.applyColorMap(aec_map, cv2.COLORMAP_MAGMA)
        aec_heat_path = os.path.join(heatmap_dir, ele + "-AEC.png")
        io.imsave(aec_heat_path, aec_heatmap)
        lym_map = embed_map[:,:,1]
        lym_heatmap = cv2.applyColorMap(lym_map, cv2.COLORMAP_MAGMA)
        lym_heat_path = os.path.join(heatmap_dir, ele + "-LYM.png")
        io.imsave(lym_heat_path, lym_heatmap)
        oc_map = embed_map[:,:,2]
        oc_heatmap = cv2.applyColorMap(oc_map, cv2.COLORMAP_MAGMA)
        oc_heat_path = os.path.join(heatmap_dir, ele + "-OC.png")
        io.imsave(oc_heat_path, oc_heatmap)

