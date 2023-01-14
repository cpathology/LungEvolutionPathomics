# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math, random, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.simplefilter("ignore", UserWarning)


def set_args():
    parser = argparse.ArgumentParser(description = "Extract lesion cellular ratio  features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--combine_dir",      type=str,       default="CombineAnalysis")
    parser.add_argument("--fea_set",          type=str,       default="1")    
    parser.add_argument("--diag_dir",         type=str,       default="Diag")    
    parser.add_argument("--vis_dir",          type=str,       default="VisPlots")    
    parser.add_argument("--plot_format",      type=str,       default=".png", choices=[".png", ".pdf"])             
    parser.add_argument("--seed",             type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    combine_dir = os.path.join(args.data_root, args.combine_dir)
    diag_dir = os.path.join(combine_dir, args.diag_dir)
    if not os.path.exists(diag_dir):
        os.makedirs(diag_dir)

    lesion_fea_path = os.path.join(combine_dir, "CombineROIFeatures.csv")
    lesion_fea_df = pd.read_csv(lesion_fea_path)
    lesion_fea_df = lesion_fea_df[lesion_fea_df["Stages"] != "Normal"]
    lesion_fea_df.loc[lesion_fea_df["Stages"] == "AAH", "Stages"] = "Precursor"
    lesion_fea_df.loc[lesion_fea_df["Stages"] == "AIS", "Stages"] = "Precursor"
    lesion_fea_df.loc[lesion_fea_df["Stages"] == "MIA", "Stages"] = "Precursor"  
    lesion_fea_df.loc[lesion_fea_df["Stages"] == "Precursor", "Diagnosis"] = 0
    lesion_fea_df.loc[lesion_fea_df["Stages"] == "ADC", "Diagnosis"] = 1

    # split us/asian
    japan_fea_df = lesion_fea_df[lesion_fea_df["Dataset"] == "Japan"]
    china_fea_df = lesion_fea_df[lesion_fea_df["Dataset"] == "China"]
    nyu_fea_df = lesion_fea_df[lesion_fea_df["Dataset"] == "USA"]
    
    fea_lst = {
        "1": ["AEC-Ratio", "LYM-Ratio"],
        "2": ["AEC-Ratio", "LYM-Ratio", "AEC-Density", "LYM-Density"],
        "3": ["AEC-Contrast", "AEC-Energy", "LYM-Contrast", "LYM-Energy"],
        "4": ["AEC-Ratio", "LYM-Ratio", "AEC-Density", "LYM-Density", "Altieri3-Entropy", "AEC-Contrast", "AEC-Energy", "LYM-Contrast", "LYM-Energy"],
        "5": ["LYM-Ratio", "AEC-Contrast", "AEC-Energy", "LYM-Contrast", "LYM-Energy"], # China-Best
        "6": ["LYM-Density", "Altieri3-Entropy", "AEC-Contrast", "AEC-Energy", "LYM-Contrast"], # USA-Best

    }

    # Japan features
    japanX = japan_fea_df[fea_lst[args.fea_set]].to_numpy().astype(np.float64)
    japanY = japan_fea_df["Diagnosis"].to_numpy().astype(np.int64)
    # China features
    chinaX = china_fea_df[fea_lst[args.fea_set]].to_numpy().astype(np.float64)
    chinaY = china_fea_df["Diagnosis"].to_numpy().astype(np.int64)    
    # NYU features
    nyuX = nyu_fea_df[fea_lst[args.fea_set]].to_numpy().astype(np.float64)
    nyuY = nyu_fea_df["Diagnosis"].to_numpy().astype(np.int64)        

    # train on Japan / test on China & USA
    japan_clf = RandomForestClassifier(max_depth=2, n_estimators=30, class_weight="balanced", random_state=args.seed)
    japan_clf.fit(japanX, japanY)

    print("Japan Model Evaluation on China Cohort")
    chinaPred = japan_clf.predict(chinaX)
    china_conf_mat = confusion_matrix(chinaY, chinaPred)
    china_acc = accuracy_score(chinaY, chinaPred)
    china_auc = roc_auc_score(chinaY, chinaPred)
    print("China Accuracy: {:.3f} AUC: {:.3f}".format(china_acc, china_auc))      

    print("Japan Model Evaluation on USA Cohort")
    nyuPred = japan_clf.predict(nyuX)
    nyu_conf_mat = confusion_matrix(nyuY, nyuPred)
    nyu_acc = accuracy_score(nyuY, nyuPred)
    nyu_auc = roc_auc_score(nyuY, nyuPred)
    print("USA Accuracy: {:.3f} AUC: {:.3f}".format(nyu_acc, nyu_auc))          


    # plot the confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    categories = ["Precursor", "ADC"]
    sns.heatmap(china_conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories, ax=axes[0])    
    axes[0].set_title("China - Accuracy: {:.3f} AUC: {:.3f}".format(china_acc, china_auc)) 
    sns.heatmap(nyu_conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories, ax=axes[1])    
    axes[1].set_title("USA - Accuracy: {:.3f} AUC: {:.3f}".format(nyu_acc, nyu_auc))       
    conf_mat_path = os.path.join(diag_dir, "FeaSet{}".format(args.fea_set) + args.plot_format)
    plt.savefig(conf_mat_path, transparent=False, dpi=300) 