# -*- coding: utf-8 -*-

import os, sys
import argparse, shutil, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_args():
    parser = argparse.ArgumentParser(description = "Extract lesion cellular ratio  features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--slide_roi_dir",    type=str,       default="SlidesROIs")
    parser.add_argument("--combine_dir",      type=str,       default="CombineAnalysis")
    parser.add_argument("--pathgenom_dir",    type=str,       default="Pathogenomics")
    parser.add_argument("--vis_dir",          type=str,       default="VisPlots")
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)

    # load all lesion features
    combine_dir = os.path.join(args.data_root, args.combine_dir)
    lesion_fea_path = os.path.join(combine_dir, "CombineROIFeatures.csv")
    lesion_fea_df = pd.read_csv(lesion_fea_path)

    # load NC19 
    genom_lesion_dict = None
    pathgenom_dir = os.path.join(args.data_root, args.pathgenom_dir)
    genom_lesion_path = os.path.join(pathgenom_dir, "AsianLesionStages.json")
    with open(genom_lesion_path, 'r') as fp:
        genom_lesion_dict = json.load(fp)
    genom_lesion_lst = [ele for ele in genom_lesion_dict.keys()]
    
    # filter genom lesion
    genom_lesion_df = lesion_fea_df[lesion_fea_df["Lesions"].isin(genom_lesion_lst)]
    print("There are {} lesions that can be used for genomic analysis.".format(len(genom_lesion_df)))

    # save nc19 lesion features    
    genom_fea_path = os.path.join(pathgenom_dir, "AsianLesionFeatures.csv")
    genom_lesion_df.to_csv(genom_fea_path, index=False) 