# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_args():
    parser = argparse.ArgumentParser(description = "Check Cell Annotations")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--celltype_dir",     type=str,       default="CellClassifier")
    parser.add_argument("--vis_dir",          type=str,       default="VisPlots")
    parser.add_argument("--plot_format",      type=str,       default=".png", choices=[".png", ".pdf"])    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    celltype_root = os.path.join(args.data_root, args.celltype_dir)
    vis_dir = os.path.join(celltype_root, args.vis_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)    
    
    nyu_cell_fea_path = os.path.join(celltype_root, "USA", "CellFeas", "cell_feas.csv")
    japan_cell_fea_path = os.path.join(celltype_root, "Japan", "CellFeas", "cell_feas.csv")
    nyu_cell_fea_df = pd.read_csv(nyu_cell_fea_path)
    japan_cell_fea_df = pd.read_csv(japan_cell_fea_path)

    cell_fea_df = pd.concat([nyu_cell_fea_df, japan_cell_fea_df])
    sources = ["Japan",] * len(japan_cell_fea_df) + ["USA",] * len(nyu_cell_fea_df) 
    cell_fea_df["Source"] = sources
    cell_fea_df.rename(columns={"Label": "Cell Type"}, inplace=True)

    # plotting
    f, axes = plt.subplots(1, 3, figsize=(11, 8))
    cell_list = ["AEC", "LYM", "OC"]
    sns.violinplot(y="Area", x="Cell Type", hue="Source", split=True, inner=None, order=cell_list, data=cell_fea_df, orient='v', ax=axes[0])
    sns.violinplot(y="Intensity", x="Cell Type", hue="Source", split=True, inner=None, order=cell_list, data=cell_fea_df, orient='v', ax=axes[1])
    sns.violinplot(y="Roundness", x="Cell Type", hue="Source", split=True, inner=None, order=cell_list, data=cell_fea_df, orient='v', ax=axes[2])
    axes[0].set_ylim(0, 1000)
    axes[1].set_ylim(0, 240)
    axes[2].set_ylim(0, 1.3)

    # save the plot
    plt.tight_layout()
    all_cell_plot_path = os.path.join(vis_dir, "cell_properites_comparison" + args.plot_format)
    plt.savefig(all_cell_plot_path, transparent=True, dpi=300)