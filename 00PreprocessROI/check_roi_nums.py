# -*- coding: utf-8 -*-

import os, sys
import argparse, shutil, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def set_args():
    parser = argparse.ArgumentParser(description = "Extract lesion cellular density features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--slide_roi_dir",    type=str,       default="SlidesROIs")
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA", "China"]) 
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)

    roi_data_root = os.path.join(args.data_root, args.slide_roi_dir, args.dataset)
    # obtain lesion stage information
    lesion_stage_dict = {}
    lesion_stage_path = os.path.join(roi_data_root, "{}LesionStages.json".format(args.dataset))
    with open(lesion_stage_path) as fp:
        lesion_stage_dict = json.load(fp)
    
    # collect stage-num
    stage_num_dict = {}
    for lesion, stage in lesion_stage_dict.items():
        if stage not in stage_num_dict.keys():
            stage_num_dict[stage] = 1
        else:
            stage_num_dict[stage] += 1
    # print stage-num
    stage_list = ["Normal", "AAH", "AIS", "MIA", "ADC"]
    print("In dataset {}".format(args.dataset))
    ttl_num = 0
    for stage in stage_list:
        if stage in stage_num_dict.keys():
            print("{} has {} lesions.".format(stage, stage_num_dict[stage]))
            ttl_num += stage_num_dict[stage]
    print("There are {} lesions in total.".format(ttl_num))            
