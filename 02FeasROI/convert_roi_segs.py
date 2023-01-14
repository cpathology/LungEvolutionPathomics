# -*- coding: utf-8 -*-

import enum
import os, sys
import argparse, json
import numpy as np


def set_args():
    parser = argparse.ArgumentParser(description = "Filter Segmented Cells")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--lesion_dir",       type=str,       default="SlidesROIs")
    parser.add_argument("--raw_seg_dir",      type=str,       default="RawCellSegs")
    parser.add_argument("--update_seg_dir",   type=str,       default="RegionSegs")
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA", "China"]) 

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    raw_seg_dir = os.path.join(args.data_root, args.lesion_dir, args.dataset, args.raw_seg_dir)
    roi_seg_dir = os.path.join(args.data_root, args.lesion_dir, args.dataset, args.update_seg_dir)
    if not os.path.exists(roi_seg_dir):
        os.makedirs(roi_seg_dir)

    roi_list = [os.path.splitext(ele)[0] for ele in os.listdir(raw_seg_dir) if ele.endswith("json")]
    for ind, cur_roi in enumerate(roi_list):
        print("Extract {}/{}".format(ind+1, len(roi_list)))
        # raw segmentation
        raw_seg_path = os.path.join(raw_seg_dir, cur_roi + ".json")
        raw_seg_dict = {}
        roi_seg_dict = {}
        with open(raw_seg_path) as fp:
            raw_seg_dict = json.load(fp)
        raw_seg_dict = raw_seg_dict["nuc"]
        roi_cid = 0
        for slide_cid in raw_seg_dict.keys():
            cell_seg = raw_seg_dict[slide_cid]
            cell_cnts = np.asarray(cell_seg["contour"])
            # rebuild the cell information
            roi_cid += 1
            cell_dict = {}
            cell_type = None
            cell_color = None
            inst_type = cell_seg["type"]
            if inst_type == 1:
                cell_type = 0
                cell_color = [255, 0, 0]
            elif inst_type == 2 or inst_type == 4:
                cell_type = 1
                cell_color = [0, 255, 0]
            else:
                cell_type = 2
                cell_color = [0, 0, 255]               
            cell_dict["type"] = cell_type
            cell_dict["color"] = cell_color
            cell_dict["contour"] = cell_cnts.tolist()
            roi_seg_dict[str(roi_cid)] = cell_dict
        # save ROI segmenation
        roi_seg_path = os.path.join(roi_seg_dir, cur_roi + ".json")
        with open(roi_seg_path, 'w') as fp:
            json.dump(roi_seg_dict, fp)        


