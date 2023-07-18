# -*- coding: utf-8 -*-

import os, sys
import argparse, shutil, json
import numpy as np
import pandas as pd


def set_args():
    parser = argparse.ArgumentParser(description = "Extract lesion cellular ratio  features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--slide_roi_dir",    type=str,       default="SlidesROIs")
    parser.add_argument("--combine_dir",      type=str,       default="CombineAnalysis")
    parser.add_argument("--rand_seed",        type=int,       default=1234)    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()
    np.random.seed(args.rand_seed)

    combine_dir = os.path.join(args.data_root, args.combine_dir)
    if not os.path.exists(combine_dir):
        os.makedirs(combine_dir)

    datasets = ["USA", "Japan", "China"]
    dset_races = ["White", "Asian", "Asian"]
    dset_race_dict = {dset: race for dset, race in zip(datasets, dset_races)}
    dataset_lst, race_lst = [], []
    fea_dfs = []
    for cur_dataset in datasets:
        roi_data_root = os.path.join(args.data_root, args.slide_roi_dir, cur_dataset)
        lesion_fea_path = os.path.join(roi_data_root, "{}LesionFeatures.csv".format(cur_dataset))
        lesion_fea_df = pd.read_csv(lesion_fea_path)
        texture_fea_path = os.path.join(roi_data_root, "{}TextureFeatures.csv".format(cur_dataset))
        texture_fea_df = pd.read_csv(texture_fea_path)
        texture_fea_df.drop(columns = ["Stages", ], inplace=True)
        cur_fea_df = pd.merge(lesion_fea_df, texture_fea_df, on="Lesions")
        fea_dfs.append(cur_fea_df)
        dataset_lst.extend([cur_dataset, ] * len(cur_fea_df))
        race_lst.extend([dset_race_dict[cur_dataset],] * len(cur_fea_df))
    combine_df = pd.concat(fea_dfs)
    combine_df["Dataset"] = dataset_lst
    combine_df["Race"] = race_lst
    combine_fea_path = os.path.join(combine_dir, "CombineROIFeatures.csv")
    combine_df.to_csv(combine_fea_path, index=False)