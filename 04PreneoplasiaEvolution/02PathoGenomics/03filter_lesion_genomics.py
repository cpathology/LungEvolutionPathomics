# -*- coding: utf-8 -*-

import os, sys
import argparse, shutil, json
import numpy as np
from scipy.stats import pearsonr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_args():
    parser = argparse.ArgumentParser(description = "Extract lesion genomics features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--pathgenom_dir",    type=str,       default="Pathogenomics")
    parser.add_argument("--vis_dir",          type=str,       default="VisPlots")
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)

    pathgenom_dir = os.path.join(args.data_root, args.pathgenom_dir)

    # load lesion pathomics features
    lesion_fea_path = os.path.join(pathgenom_dir, "AsianLesionFeatures.csv")
    path_fea_df = pd.read_csv(lesion_fea_path)       
    path_slide_lst = [ele[:ele.rfind("-")] for ele in path_fea_df["Lesions"].tolist()]
    path_fea_df["SlideName"] = path_slide_lst

    # load lesion genomic features
    raw_immune_path = os.path.join(pathgenom_dir, "NC-ImmuneMarkers.xlsx")
    immune_fea_df = pd.read_excel(raw_immune_path)
    immune_fea_df = immune_fea_df[immune_fea_df["Old.name"].isin(path_slide_lst)]
    immune_old_lst = immune_fea_df["Old.name"].tolist()
    immune_fea_df["SlideName"] = immune_old_lst
    # construct new.name and old.name mapping
    immune_new_lst = immune_fea_df["New.name"].tolist()
    new_old_dict = {new_name: old_name for old_name, new_name in zip(immune_old_lst, immune_new_lst)}
    # immune_fea_list = ["SlideName", "TMB.nonsynonymous.log2", "AI.burden", "CNV.burden(gain+loss)",
    #     "T.cell.fraction", "P1.Total.T.cells", "P1.Cytotoxic.T.cells", "P1.Total.Macrophages", "P2.Total.T.cells",
    #     "P2.Cytotoxic.T.cells", "P2.Cytotoxic.T.cells.CD8.activated", "P2.Regulatory.T.cells"]
    immune_fea_list = ["SlideName", "TMB.nonsynonymous.log2", "AI.burden", "CNV.burden(gain+loss)", "T.cell.fraction", 
        "P1.Total.T.cells", "P1.Cytotoxic.T.cells", "P1.Total.Macrophages", "P2.Cytotoxic.T.cells.CD8.activated", "P2.Regulatory.T.cells"]

    lesion_immune_df = immune_fea_df[immune_fea_list]
    path_immnue_df = path_fea_df.merge(lesion_immune_df, how = "left", left_on = "SlideName", right_on = "SlideName")
    
    raw_timer_path = os.path.join(pathgenom_dir, "NC-TimerMarkers.xlsx")
    timer_fea_df = pd.read_excel(raw_timer_path)
    timer_fea_df = timer_fea_df[timer_fea_df["LesionName"].isin(immune_new_lst)]
    timer_lesion_lst = timer_fea_df["LesionName"].tolist()
    timer_slide_lst = [new_old_dict[ele] for ele in timer_lesion_lst]
    timer_fea_df["SlideName"] = timer_slide_lst
    timer_fea_df = timer_fea_df[timer_fea_df["SlideName"].isin(path_slide_lst)]
    timer_fea_list = ["SlideName", "CD4+ T cells", "CD8+ T cells"]
    lesion_timer_df = timer_fea_df[timer_fea_list]
    path_genom_df = path_immnue_df.merge(lesion_timer_df, how = "left", left_on = "SlideName", right_on = "SlideName")
    
    # save pathogenomics 
    path_genom_fea_path = os.path.join(pathgenom_dir, "LesionPathoGenomics.xlsx")
    path_genom_df.to_excel(path_genom_fea_path, index=False)      