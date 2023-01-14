# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math, csv, pytz
from datetime import datetime
import collections, pickle
import shutil, copy
import pandas as pd
import numpy as np
from skimage import io, color, filters
from skimage.feature import graycomatrix, graycoprops


def set_args():
    parser = argparse.ArgumentParser(description = "Extract lesion cellular ratio  features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--slide_roi_dir",    type=str,       default="SlidesROIs")
    parser.add_argument("--embed_dir",        type=str,       default="EmbedMaps")  
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA", "China"]) 
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)

    roi_data_root = os.path.join(args.data_root, args.slide_roi_dir, args.dataset)
    # obtain lesion stage information
    lesion_stage_dict = {}
    lesion_stage_path = os.path.join(roi_data_root, "{}LesionStages.json".format(args.dataset))
    with open(lesion_stage_path) as fp:
        lesion_stage_dict = json.load(fp)    
            
    # Embed directory
    roi_embed_dir = os.path.join(args.data_root, args.slide_roi_dir, args.dataset, args.embed_dir)
    roi_list = [os.path.splitext(ele)[0] for ele in os.listdir(roi_embed_dir) if ele.endswith(".png")]

    # roi & stage
    ROIs, Stages = [], []
    aec_contrasts, aec_energies = [], []
    lym_contrasts, lym_energies = [], []

    # extract features
    for ind, ele in enumerate(roi_list):
        print("Extract features from {}/{}".format(ind+1, len(roi_list)))
        # add meta information
        ROIs.append(ele)
        Stages.append(lesion_stage_dict[ele])

        cur_embed_path = os.path.join(roi_embed_dir, ele + ".png")
        cur_embed_map = io.imread(cur_embed_path)
        aec_map, lym_map = cur_embed_map[:, :, 0], cur_embed_map[:, :, 1]
        # epithelial  
        aec_glcm = graycomatrix(aec_map, distances=[3], angles=[0], levels=256, symmetric=True, normed=True)
        aec_contrasts.append(graycoprops(aec_glcm, "contrast")[0,0])
        aec_energies.append(graycoprops(aec_glcm, "energy")[0,0])
        # lymphocyte 
        lym_glcm = graycomatrix(lym_map, distances=[3], angles=[0], levels=256, symmetric=True, normed=True)
        lym_contrasts.append(graycoprops(lym_glcm, "contrast")[0,0])
        lym_energies.append(graycoprops(lym_glcm, "energy")[0,0])

    # save features
    fea_list = list(zip(ROIs, Stages, aec_contrasts, aec_energies, lym_contrasts, lym_energies))
    fea_names = ["Lesions", "Stages", "AEC-Contrast", "AEC-Energy", "LYM-Contrast", "LYM-Energy"]
    roi_fea_df = pd.DataFrame(fea_list, columns = fea_names)
    roi_fea_path = os.path.join(roi_data_root, "{}TextureFeatures.csv".format(args.dataset))
    roi_fea_df.to_csv(roi_fea_path, index=False)