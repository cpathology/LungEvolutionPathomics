# -*- coding: utf-8 -*-

import os, sys
import argparse, json
import numpy as np


def set_args():
    parser = argparse.ArgumentParser(description = "Count Annotated Cell Numbers")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--celltype_dir",     type=str,       default="CellClassifier")
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA"]) 

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    celltype_root = os.path.join(args.data_root, args.celltype_dir)
    annotation_dir = os.path.join(celltype_root, args.dataset, "AnnotationConsensus")
    if not os.path.exists(annotation_dir):
        sys.exit("Annotations donot exist.")

    # collect cell annotations
    roi_list = sorted([os.path.splitext(ele)[0] for ele in os.listdir(annotation_dir) if ele.endswith(".json")])
    cell_dict = {}
    for cur_roi in roi_list:
        cur_annotation_path = os.path.join(annotation_dir, cur_roi + ".json")
        cur_dict = None
        with open(cur_annotation_path) as fp:
            cur_dict = json.load(fp)
        cell_lst = cur_dict["shapes"]
        for cur_cell in cell_lst:
            cell_label = cur_cell["label"]
            if cell_label not in cell_dict:
                cell_dict[cell_label] = 1
            else:
                cell_dict[cell_label] += 1

    # Summary cell annotations
    cell_lst = ["AEC", "LYM", "OC"]
    for cell_type in cell_lst:
        print("{} has {} cells.".format(cell_type, cell_dict[cell_type]))    