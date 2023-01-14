# -*- coding: utf-8 -*-

import os, sys
import argparse, shutil
import numpy as np
from skimage import io
from histocartography.preprocessing import MacenkoStainNormalizer


def set_args():
    parser = argparse.ArgumentParser(description = "Mencenko Stain Normalization")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--lesion_dir",       type=str,       default="SlidesROIs")
    parser.add_argument("--block_dir",        type=str,       default="RegionROIs")
    parser.add_argument("--norm_dir",         type=str,       default="MacenkoROIs")    
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA", "China"])    

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    lesion_root_dir = os.path.join(args.data_root, args.lesion_dir, args.dataset)
    # identify all images need to normalize
    roi_img_root = os.path.join(lesion_root_dir, args.block_dir)
    img_list = sorted([ele for ele in os.listdir(roi_img_root) if ele.endswith(".png")])

    # setup stain normalization results location
    roi_norm_root = os.path.join(lesion_root_dir, args.norm_dir)
    if os.path.exists(roi_norm_root):
        shutil.rmtree(roi_norm_root)
    os.makedirs(roi_norm_root)

    # normalize image one-by-one
    normalizer = MacenkoStainNormalizer()
    for num, img_name in enumerate(img_list):
        print("Normalize {}/{} name: {}".format(num+1, len(img_list), img_name))
        roi_img_path = os.path.join(roi_img_root, img_name)
        image = io.imread(roi_img_path)
        img_norm = normalizer.process(image)
        norm_img_path = os.path.join(roi_norm_root, img_name)
        io.imsave(norm_img_path, img_norm)