# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math, random, pickle
import numpy as np
import pandas as pd


def set_args():
    parser = argparse.ArgumentParser(description = "Evaluate ROI-Level Cell Classification")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--celltype_dir",     type=str,       default="CellClassifier")
    parser.add_argument("--roi_eval_dir",     type=str,       default="EvaluationROI")    
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA", "China"])
    parser.add_argument("--evaluator",        type=str,       default="Frank", choices=["Frank", "Serrano"])
    parser.add_argument("--seed",             type=int,       default=1234)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    data_root = os.path.join(args.data_root, args.celltype_dir, args.roi_eval_dir)
    eval_roi_dir = os.path.join(data_root, "Evaluation{}".format(args.evaluator))

    eval_roi_path = os.path.join(eval_roi_dir, "{}-Evaluation.xlsx".format(args.dataset))
    if not os.path.exists(eval_roi_path):
        sys.exit("Evaluation file not exist")

    roi_eval_df = pd.read_excel(eval_roi_path)
    aec_accs = [int(ele) for ele in roi_eval_df["AEC"].tolist()]
    print("AEC mean accuracy is: {:.1f}".format(np.mean(aec_accs)))
    lym_accs = [int(ele) for ele in roi_eval_df["LYM"].tolist()]
    print("LYM mean accuracy is: {:.1f}".format(np.mean(lym_accs)))    
    oc_accs = [int(ele) for ele in roi_eval_df["OC"].tolist()]
    print("OCD mean accuracy is: {:.1f}".format(np.mean(oc_accs)))        