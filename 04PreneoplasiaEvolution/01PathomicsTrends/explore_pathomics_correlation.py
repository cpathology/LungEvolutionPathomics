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
    parser.add_argument("--combine_dir",      type=str,       default="CombineAnalysis")
    parser.add_argument("--trend_dir",        type=str,       default="PathomicsTrend")
    parser.add_argument("--plot_format",      type=str,       default=".png", choices=[".png", ".pdf"])        
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)

    combine_dir = os.path.join(args.data_root, args.combine_dir)
    path_trend_dir = os.path.join(combine_dir, args.trend_dir)
    if not os.path.exists(path_trend_dir):
        os.makedirs(path_trend_dir)

    fuse_fea_path = os.path.join(combine_dir, "CombineROIFeatures.csv")
    fuse_fea_df = pd.read_csv(fuse_fea_path)

    ratio_feas = ["AEC-Ratio", "LYM-Ratio",]
    density_feas = ["AEC-Density", "LYM-Density",]
    spatial_feas = ["Altieri3-Entropy", ]
    embed_feas =  ["AEC-Contrast",	"AEC-Energy", "LYM-Contrast", "LYM-Energy"]
    pathomic_markers = ratio_feas + density_feas + spatial_feas + embed_feas

    # correation heatmaps
    marker_df = fuse_fea_df[pathomic_markers]
    corr_mat = marker_df.corr()
    fig, axes = plt.subplots(1, 1, figsize=(10, 6))
    sns.heatmap(corr_mat, annot=True, fmt=".3f", cmap="rocket_r")
    plt.xticks(rotation=30) 
    plt.tight_layout()
    plot_name = "pathomics_correlation_heatmap"
    plot_path = os.path.join(path_trend_dir, plot_name + args.plot_format)
    plt.savefig(plot_path, transparent=False, dpi=300)        
