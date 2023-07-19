# -*- coding: utf-8 -*-

import os, sys
import argparse, shutil, json, re
import numpy as np
import pandas as pd


def set_args():
    parser = argparse.ArgumentParser(description = "Combine Features")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--slide_roi_dir",    type=str,       default="SlidesROIs")
    parser.add_argument("--demographic_dir",  type=str,       default="ClinicoDemographics")
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
    dataset_lst, race_lst, smoke_lst = [], [], []
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
        dset_clinicodemographic_path = os.path.join(args.data_root, args.demographic_dir, cur_dataset + ".xlsx")
        dset_demographic_df = pd.read_excel(dset_clinicodemographic_path)
        dset_patient_id_lst = [str(ele) for ele in dset_demographic_df["PatientID"].tolist()]
        dset_smoke_stat_lst = [ele for ele in dset_demographic_df["SmokerType"].tolist()]
        pat_smoke_dict = {pat:smoke for pat, smoke in zip(dset_patient_id_lst, dset_smoke_stat_lst)}
        dset_lesion_lst = [ele for ele in cur_fea_df["Lesions"].tolist()]
        for cur_lesion in dset_lesion_lst:
            dash_indices = [match.start() for match in re.finditer("-", cur_lesion)]
            cur_pat = cur_lesion[:dash_indices[-2]]
            smoke_lst.append(pat_smoke_dict[cur_pat])
    combine_df = pd.concat(fea_dfs)
    combine_df["Dataset"] = dataset_lst
    combine_df["Race"] = race_lst
    combine_df["SmokeStatus"] = smoke_lst
    combine_fea_path = os.path.join(combine_dir, "CombineROIFeatures.csv")
    combine_df.to_csv(combine_fea_path, index=False)