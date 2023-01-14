# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math, csv
import collections, pickle
import shutil, copy
import pandas as pd
import numpy as np
from skimage import io, color
import cv2


def set_args():
    parser = argparse.ArgumentParser(description = "Extract lesion cellular ratio  features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--slide_roi_dir",    type=str,       default="SlidesROIs")
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA", "China"]) 
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)

    roi_data_root = os.path.join(args.data_root, args.slide_roi_dir, args.dataset)
    lesion_fea_path = os.path.join(roi_data_root, "{}LesionFeatures.csv".format(args.dataset))

    lesion_fea_df = pd.read_csv(lesion_fea_path)
    lesion_names = lesion_fea_df["Lesions"].tolist()
    heights, widths = [], []
    for cur_lesion in lesion_names:
        w_loc = cur_lesion.find("Wlen")
        widths.append(int(cur_lesion[w_loc+4:w_loc+10]))
        h_loc = cur_lesion.find("Hlen")
        heights.append(int(cur_lesion[h_loc+4:h_loc+10]))
    max_height, min_height = max(heights), min(heights)
    print("Height max: {}, min: {}".format(max_height, min_height))
    max_width, min_width = max(widths), min(widths)
    print("Width max: {}, min: {}".format(max_width, min_width))


