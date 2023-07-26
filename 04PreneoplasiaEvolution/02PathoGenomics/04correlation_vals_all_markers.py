# -*- coding: utf-8 -*-

import os, sys
import argparse, shutil, json
import numpy as np
import copy
from scipy.stats import pearsonr
from statsmodels.stats import multitest
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_args():
    parser = argparse.ArgumentParser(description = "Extract lesion cellular ratio  features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--pathgenom_dir",    type=str,       default="Pathogenomics")
    parser.add_argument("--vis_dir",          type=str,       default="VisPlots")
    parser.add_argument("--corr_method",      type=str,       default="spearman", choices=["spearman", "pearson"])
    parser.add_argument("--plot_format",      type=str,       default=".png", choices=[".png", ".pdf"])            
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)

    pathgenom_dir = os.path.join(args.data_root, args.pathgenom_dir)
    vis_dir = os.path.join(pathgenom_dir, args.vis_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    path_genom_fea_path = os.path.join(pathgenom_dir, "LesionPathoGenomics.xlsx")
    path_genom_df = pd.read_excel(path_genom_fea_path)

    pathomic_markers = ["AEC-Proportion", "LYM-Proportion", "AEC-Density", "LYM-Density", "Altieri3-Entropy", 
        "AEC-Contrast", "AEC-Energy", "LYM-Contrast", "LYM-Energy"]
    genomic_markers = ["TMB.nonsynonymous.log2", "AI.burden", "CNV.burden(gain+loss)", "T.cell.fraction", 
        "P1.Total.T.cells", "P1.Cytotoxic.T.cells", "P1.Total.Macrophages", 
        "P2.Cytotoxic.T.cells.CD8.activated", "P2.Regulatory.T.cells",
        "CD4+ T cells",  "CD8+ T cells"]
    genomic_marker_names = ["TMB", "AI-Burden", "CNV-Burden", "T-Cell-Fraction", 
        "Total-T-Cells", "Cytotoxic-T-Cells", "Total-Macrophages", "Cytotoxic-T-Cells-CD8-Activated", "Regulatory-T-Cells",
        "CD4+T-Cells", "CD8+T-Cells"]
    markers = pathomic_markers + genomic_markers

    # correation heatmaps
    marker_df = path_genom_df[markers]
    corr_mat = marker_df.corr(method=args.corr_method)

    fig, axes = plt.subplots(1, 1, figsize=(12, 8))
    show_corr_mat = corr_mat.iloc[len(pathomic_markers):, :len(pathomic_markers)]
    sc = sns.heatmap(show_corr_mat, annot=True, fmt=".3f", cmap="rocket_r")
    sc.set(yticklabels=genomic_marker_names)
    plt.xticks(rotation=30) 
    plt.tight_layout()
    plot_name = "{}-pathogenomics_correlation_heatmap_AllMarkers".format(args.corr_method)
    plot_path = os.path.join(vis_dir, plot_name + args.plot_format)
    plt.savefig(plot_path, transparent=False, dpi=300)  