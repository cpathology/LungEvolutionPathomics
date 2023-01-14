# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math, random, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns


def set_args():
    parser = argparse.ArgumentParser(description = "Evaluate Cell Classification")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--celltype_dir",     type=str,       default="CellClassifier")
    parser.add_argument("--cellmodel_dir",    type=str,       default="CellModels")    
    parser.add_argument("--vis_dir",          type=str,       default="VisPlots")   
    parser.add_argument("--plot_format",      type=str,       default=".png", choices=[".png", ".pdf"])                
    parser.add_argument("--seed",             type=int,       default=1234)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    vis_dir = os.path.join(args.data_root, args.celltype_dir, args.vis_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)    
    cellmodel_dir = os.path.join(args.data_root, args.celltype_dir, "CellModels")
    if not os.path.exists(cellmodel_dir):
        os.makedirs(cellmodel_dir)

    # load japan features
    japan_data_root = os.path.join(args.data_root, args.celltype_dir, "Japan")
    japan_cell_fea_dir = os.path.join(japan_data_root, "CellFeas")
    japan_cell_fea_path = os.path.join(japan_cell_fea_dir, "cell_feas.csv")
    japan_cell_fea_df = pd.read_csv(japan_cell_fea_path)
    # obtain features
    japan_X = japan_cell_fea_df[["Area", "Intensity", "Roundness"]].to_numpy().astype(np.float64)
    # obtain labels
    japan_cell_fea_df["Class"] = japan_cell_fea_df["Label"]
    japan_cell_fea_df.loc[japan_cell_fea_df["Class"] == "AEC", "Class"] = 0
    japan_cell_fea_df.loc[japan_cell_fea_df["Class"] == "LYM", "Class"] = 1
    japan_cell_fea_df.loc[japan_cell_fea_df["Class"] == "OC", "Class"] = 2
    japan_y = japan_cell_fea_df["Class"].to_numpy().astype(np.int64)

    # load japan classifier
    japan_celltype_model_path = os.path.join(cellmodel_dir, "Japan_cell_classifier.model")
    japan_cell_clf = pickle.load(open(japan_celltype_model_path, "rb"))

    # load NYU features
    nyu_data_root = os.path.join(args.data_root, args.celltype_dir, "USA")
    nyu_cell_fea_dir = os.path.join(nyu_data_root, "CellFeas")
    nyu_cell_fea_path = os.path.join(nyu_cell_fea_dir, "cell_feas.csv")
    nyu_cell_fea_df = pd.read_csv(nyu_cell_fea_path)
    # obtain features
    nyu_X = nyu_cell_fea_df[["Area", "Intensity", "Roundness"]].to_numpy().astype(np.float64)
    # obtain labels
    nyu_cell_fea_df["Class"] = nyu_cell_fea_df["Label"]
    nyu_cell_fea_df.loc[nyu_cell_fea_df["Class"] == "AEC", "Class"] = 0
    nyu_cell_fea_df.loc[nyu_cell_fea_df["Class"] == "LYM", "Class"] = 1
    nyu_cell_fea_df.loc[nyu_cell_fea_df["Class"] == "OC", "Class"] = 2
    nyu_y = nyu_cell_fea_df["Class"].to_numpy().astype(np.int64)

    # load NYU classifier
    nyu_celltype_model_path = os.path.join(cellmodel_dir, "USA_cell_classifier.model")
    nyu_cell_clf = pickle.load(open(nyu_celltype_model_path, "rb"))

    # Evaluate Japan Cell Classifier on NYU Cells
    nyu_pred = japan_cell_clf.predict(nyu_X)
    nyu_conf_mat = confusion_matrix(nyu_y, nyu_pred)
    print("Japan model on USA cells:")
    print(nyu_conf_mat)
    nyu_acc = accuracy_score(nyu_y, nyu_pred)
    print("Accuracy: {:.3f}".format(nyu_acc))
    # Evaluate NYU Cell Classifier on Japan Cells
    japan_pred = nyu_cell_clf.predict(japan_X)
    japan_conf_mat = confusion_matrix(japan_y, japan_pred)
    print("USA model on japan cells:")
    print(japan_conf_mat)
    japan_acc = accuracy_score(japan_y, japan_pred)
    print("Accuracy: {:.3f}".format(japan_acc)) 

    # plot the confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    categories = ["AEC", "LYM", "OC"]
    sns.heatmap(nyu_conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories, ax=axes[0])
    axes[0].set_title("Japan model on USA cells - Acc:{:.3f}".format(nyu_acc))
    sns.heatmap(japan_conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=categories,yticklabels=categories, ax=axes[1])
    axes[1].set_title("USA model on Japan cells - Acc:{:.3f}".format(japan_acc))
    fig.suptitle("Cell Classifier Cross-Cohort Evaluation")

    conf_mat_path = os.path.join(vis_dir, "cross_eval_conf_mat" + args.plot_format)
    plt.savefig(conf_mat_path, transparent=False, dpi=300) 