# -*- coding: utf-8 -*-

import os, sys
import argparse, json


def set_args():
    parser = argparse.ArgumentParser(description = "Identify lesions for genomics correlations")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--pathgenom_dir",    type=str,       default="Pathogenomics")
    parser.add_argument("--slide_roi_dir",    type=str,       default="SlidesROIs")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    pathgenom_dir = os.path.join(args.data_root, args.pathgenom_dir)

    slide_histology_dict = None
    slide_histology_path = os.path.join(pathgenom_dir, "SlideHistology.json")
    with open(slide_histology_path, 'r') as fp:
        slide_histology_dict = json.load(fp)
    slide_lst = [ele for ele in slide_histology_dict.keys()]

    # genrate lesion dictionary
    china_lesion_dict, japan_lesion_dict = None, None
    china_lesion_stage_path = os.path.join(args.data_root, args.slide_roi_dir, "China", "ChinaLesionStages.json")
    japan_lesion_stage_path = os.path.join(args.data_root, args.slide_roi_dir, "Japan", "JapanLesionStages.json")
    with open(china_lesion_stage_path, 'r') as fp:
        china_lesion_dict = json.load(fp)
    print("There are {} lesions in China Cohort.".format(len(china_lesion_dict)))
    with open(japan_lesion_stage_path, 'r') as fp:
        japan_lesion_dict = json.load(fp)
    print("There are {} lesions in Japan Cohort.".format(len(japan_lesion_dict)))
    ChinaJapan = {**china_lesion_dict, **japan_lesion_dict}
    print("There are {} lesions together.".format(len(ChinaJapan)))
    genom_lesion_dict = {}
    for lesion_name, label in ChinaJapan.items():
        slide_name = lesion_name[:lesion_name.rfind("-")]
        if slide_name in slide_lst and slide_histology_dict[slide_name] == label:
            genom_lesion_dict[lesion_name] = label
    print("There are {} lesions can be used for genomics analysis.".format(len(genom_lesion_dict)))

    # save lesion for genomic correlation analysis
    genom_lesion_path = os.path.join(pathgenom_dir, "AsianLesionStages.json")
    with open(genom_lesion_path, 'w') as fp:
        json.dump(genom_lesion_dict, fp)

    # # Collect and print lesion statistics information
    # lesion_num_dict = {}
    # for lesion_name, label in genom_lesion_dict.items():
    #     lesion_key = lesion_name[0] + "-" + label
    #     if lesion_key not in lesion_num_dict.keys():
    #         lesion_num_dict[lesion_key] = 1
    #     else:
    #         lesion_num_dict[lesion_key] += 1
    # for key, val in lesion_num_dict.items():
    #     print("{} has {} lesions.".format(key, val))


