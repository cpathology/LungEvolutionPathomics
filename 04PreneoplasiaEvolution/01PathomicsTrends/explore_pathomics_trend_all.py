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
    path_trend_dir = os.path.join(combine_dir, args.trend_dir, "All")
    if not os.path.exists(path_trend_dir):
        os.makedirs(path_trend_dir)

    fuse_fea_path = os.path.join(combine_dir, "CombineROIFeatures.csv")
    fuse_fea_df = pd.read_csv(fuse_fea_path)
   
    # compare two cohort features
    pathomics_lst = ["AEC-Ratio", "LYM-Ratio", "AEC-Density", "LYM-Density", "Altieri2-Entropy", "Altieri3-Entropy",
        "AEC-Contrast",	"AEC-Energy", "LYM-Contrast", "LYM-Energy"]
    stage_list = ["Normal", "AAH", "AIS", "MIA", "ADC"]    
    for path_fea in pathomics_lst:
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        sns.violinplot(data=fuse_fea_df, x = "Stages", y=path_fea, order=stage_list,  orient='v', ax=axes[0])
        axes[0].spines[['right', 'top']].set_visible(False)
        sns.violinplot(data=fuse_fea_df, x = "Stages", y=path_fea, hue = "Dataset", order=stage_list,  orient='v', ax=axes[1])
        sns.violinplot(data=fuse_fea_df, x = "Stages", y=path_fea, hue = "Race", order=stage_list,  orient='v', ax=axes[2])
        sns.violinplot(data=fuse_fea_df, x = "Stages", y=path_fea, hue = "SmokeStatus", order=stage_list,  orient='v', ax=axes[3])
        # plt.legend(loc = "best")
        fig.suptitle("{} Trends Comparison on All Data, Three Cohorts, Two Races, and Smoking Status Across Stages".format(path_fea))
        for ind in range(1, 4):
            axes[ind].spines[['left', 'right', 'top']].set_visible(False)
            axes[ind].get_yaxis().set_visible(False)
            axes[ind].legend(frameon=False)
        plt.tight_layout()
        trend_plot_path = os.path.join(path_trend_dir, "{}-Trends{}".format(path_fea, args.plot_format))
        plt.savefig(trend_plot_path, transparent=False, dpi=300)