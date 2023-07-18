# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math, random, pickle
import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix
import seaborn as sns


def set_args():
    parser = argparse.ArgumentParser(description = "Check Cell Annotations")
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

    celltype_root = os.path.join(args.data_root, args.celltype_dir)
    vis_dir = os.path.join(celltype_root, args.vis_dir)
    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)    
    cellmodel_dir = os.path.join(args.data_root, args.celltype_dir, "CellModels")
    if not os.path.exists(cellmodel_dir):
        os.makedirs(cellmodel_dir)

    nyu_cell_fea_path = os.path.join(celltype_root, "USA", "CellFeas", "cell_feas.csv")
    japan_cell_fea_path = os.path.join(celltype_root, "Japan", "CellFeas", "cell_feas.csv")
    nyu_cell_fea_df = pd.read_csv(nyu_cell_fea_path)
    japan_cell_fea_df = pd.read_csv(japan_cell_fea_path)

    cell_fea_df = pd.concat([nyu_cell_fea_df, japan_cell_fea_df])
    sources = ["USA",] * len(nyu_cell_fea_df) + ["Japan",] * len(japan_cell_fea_df)
    cell_fea_df["Sources"] = sources

    # obtain features
    X = cell_fea_df[["Area", "Intensity", "Roundness"]].to_numpy().astype(np.float64)
    # obtain labels
    cell_fea_df["Class"] = cell_fea_df["Label"]
    cell_fea_df.loc[cell_fea_df["Class"] == "AEC", "Class"] = 0
    cell_fea_df.loc[cell_fea_df["Class"] == "LYM", "Class"] = 1
    cell_fea_df.loc[cell_fea_df["Class"] == "OC", "Class"] = 2
    y = cell_fea_df["Class"].to_numpy().astype(np.int64)   

    # cross-validation
    clf = xgb.XGBClassifier()
    scores = cross_val_score(clf, X, y, cv=5)
    print("%0.3f accuracy with a standard deviation of %0.3f" % (scores.mean(), scores.std()))
    # confusion matrix
    y_pred = cross_val_predict(clf, X, y, cv=5)
    cv_conf_mat = confusion_matrix(y, y_pred)
    print("CV Confusion Matrix:")
    print(cv_conf_mat)

    # fit & save model
    clf.fit(X, y)
    celltype_model_path = os.path.join(cellmodel_dir, "fusing_cell_classifier.model")
    pickle.dump(clf, open(celltype_model_path, "wb"))

    # plot the confusion matrix
    fig, axes = plt.subplots(1, 1, figsize=(8, 6))
    categories = ["AEC", "LYM", "OC"]
    sns.heatmap(cv_conf_mat, annot=True, fmt="d", cmap="Blues", xticklabels=categories, yticklabels=categories)

    fig.supxlabel("Predictions")     
    fig.supylabel("Ground Truth")    
    fig.suptitle("Accuracy: {:.3f} Standard Deviation: {:.3f}".format(scores.mean(), scores.std()))
    cv_conf_mat_path = os.path.join(vis_dir, "fusing_cv_conf_mat" + args.plot_format)
    plt.savefig(cv_conf_mat_path, transparent=False, dpi=300)   