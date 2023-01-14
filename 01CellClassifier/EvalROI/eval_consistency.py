# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math, random, pickle
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def set_args():
    parser = argparse.ArgumentParser(description = "Evaluate ROI-Level Cell Classification")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--celltype_dir",     type=str,       default="CellClassifier")
    parser.add_argument("--roi_eval_dir",     type=str,       default="EvaluationROI")    
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA", "China"])
    parser.add_argument("--vis_dir",          type=str,       default="VisPlots")    
    parser.add_argument("--plot_format",      type=str,       default=".png", choices=[".png", ".pdf"])          
    parser.add_argument("--seed",             type=int,       default=1234)

    args = parser.parse_args()
    return args


def score_map(score):
    val = None
    if score == 100:
        val = 0
    elif score >=80 and score < 100:
        val = 1
    elif score >=60 and score < 80:
        val = 2
    else:
        val = 3

    return val


if __name__ == "__main__":
    args = set_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = os.path.join(args.data_root, args.celltype_dir)
    frank_eval_roi_dir = os.path.join(data_root, args.roi_eval_dir, "EvaluationFrank")
    frank_eval_roi_path = os.path.join(frank_eval_roi_dir, "{}-Evaluation.xlsx".format(args.dataset))
    if not os.path.exists(frank_eval_roi_path):
        sys.exit("Frank's evaluation of {} not exist".format(args.dataset))

    serrano_eval_roi_dir = os.path.join(data_root, args.roi_eval_dir, "EvaluationSerrano")
    serrano_eval_roi_path = os.path.join(serrano_eval_roi_dir, "{}-Evaluation.xlsx".format(args.dataset))
    if not os.path.exists(serrano_eval_roi_path):
        sys.exit("Serrano's evaluation of {} not exist".format(args.dataset))


    vis_dir = os.path.join(data_root, args.vis_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)    


    # obtain Frank's evaluation lst
    frank_eval_df = pd.read_excel(frank_eval_roi_path)
    frank_aec_lst = [int(ele) for ele in frank_eval_df["AEC"].tolist()]
    frank_lym_lst = [int(ele) for ele in frank_eval_df["LYM"].tolist()]
    frank_oc_lst = [int(ele) for ele in frank_eval_df["OC"].tolist()]

    # obtain Serrano's evaluation list
    serrano_eval_df = pd.read_excel(serrano_eval_roi_path)
    serrano_aec_lst = [int(ele) for ele in serrano_eval_df["AEC"].tolist()]
    serrano_lym_lst = [int(ele) for ele in serrano_eval_df["LYM"].tolist()]
    serrano_oc_lst = [int(ele) for ele in serrano_eval_df["OC"].tolist()]   

    aec_matrix = np.zeros((4, 4), dtype = np.int32)
    for f1, s1 in zip(frank_aec_lst, serrano_aec_lst):
        ind1 = score_map(f1)
        ind2 = score_map(s1)
        aec_matrix[ind1, ind2] += 1
    lym_matrix = np.zeros((4, 4), dtype = np.int32)
    for f1, s1 in zip(frank_lym_lst, serrano_lym_lst):
        ind1 = score_map(f1)
        ind2 = score_map(s1)
        lym_matrix[ind1, ind2] += 1    
    oc_matrix = np.zeros((4, 4), dtype = np.int32)
    for f1, s1 in zip(frank_oc_lst, serrano_oc_lst):
        ind1 = score_map(f1)
        ind2 = score_map(s1)
        oc_matrix[ind1, ind2] += 1            

    # draw consistensy map
    categories = ["Outstanding", "Good", "Tolerable", "Poor"]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sns.heatmap(aec_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories, ax=axes[0])
    sns.heatmap(lym_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories, ax=axes[1])
    sns.heatmap(oc_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories, ax=axes[2])
    axes[0].set_title("AEC")
    axes[1].set_title("LYM")
    axes[2].set_title("OC")
    # plt.tight_layout()
    fig.suptitle("{} Consistency Matrix of ROI-Level Cellular Recognition Evaluations".format(args.dataset))
    cv_conf_mat_path = os.path.join(vis_dir, "ROI_eval_consistency_{}".format(args.dataset) + args.plot_format)
    plt.savefig(cv_conf_mat_path, transparent=False, dpi=300)   