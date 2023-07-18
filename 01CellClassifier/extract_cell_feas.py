# -*- coding: utf-8 -*-

import os, sys
import argparse, json, math
import shutil, copy
import numpy as np
from skimage import io
import cv2
import pandas as pd

from seg_utils import bounding_box


def set_args():
    parser = argparse.ArgumentParser(description = "Extract Cell Features from Annotations")
    parser.add_argument("--data_root",        type=str,       default="/Data")
    parser.add_argument("--celltype_dir",     type=str,       default="CellClassifier")
    parser.add_argument("--roi_img_dir",      type=str,       default="MacenkoROIs")
    parser.add_argument("--roi_seg_dir",      type=str,       default="RegionSegs")
    parser.add_argument("--dataset",          type=str,       default="Japan", choices=["Japan", "USA"]) 

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = set_args()

    data_root = os.path.join(args.data_root, args.celltype_dir, args.dataset)
    annotation_dir = os.path.join(data_root, "AnnotationConsensus")
    roi_img_dir = os.path.join(data_root, args.roi_img_dir)
    roi_seg_dir = os.path.join(data_root, args.roi_seg_dir)
    cell_fea_dir = os.path.join(data_root, "CellFeas")
    if os.path.exists(cell_fea_dir):
        shutil.rmtree(cell_fea_dir)
    os.makedirs(cell_fea_dir)

    # collect cell annotations
    roi_list = sorted([os.path.splitext(ele)[0] for ele in os.listdir(annotation_dir) if ele.endswith(".json")])
    cell_labels, cell_areas, cell_intensities, cell_circularities = [], [], [], []

    for ind, cur_roi in enumerate(roi_list):
        # print("Extract {}/{} annotation on {}".format(ind+1, len(roi_list), cur_roi))
        roi_img_path = os.path.join(roi_img_dir, cur_roi + ".png")
        if not os.path.exists(roi_img_path):
            sys.exit("{} do not have image.".format(cur_roi))
        roi_img = io.imread(roi_img_path)
        roi_seg_mask = np.zeros((roi_img.shape[0], roi_img.shape[1]), dtype = np.uint16)
        # load HoVer-Net reference
        roi_seg_path = os.path.join(roi_seg_dir, cur_roi + ".json")
        if not os.path.exists(roi_seg_path):
            sys.exit("{} do not have segmentation.".format(cur_roi))
        cell_reference_dict = {}
        with open(roi_seg_path, 'r') as fp:
            cell_reference_dict = json.load(fp)
        cell_keys = [key for key in cell_reference_dict]
        cell_label_dict = {}
        for ind, key in enumerate(cell_keys):
            cur_cell_dict = cell_reference_dict[key]
            cur_cell_phenotype = cur_cell_dict["type"]
            cell_val = int(key)
            cell_label_dict[cell_val] = cur_cell_phenotype
            cell_cnt = np.asarray(cur_cell_dict["contour"])
            cell_cnt = np.expand_dims(cell_cnt, axis=1)
            cv2.drawContours(roi_seg_mask, contours=[cell_cnt, ], contourIdx=0, color=cell_val, thickness=-1)
        # load annotations
        cell_annotation_path = os.path.join(annotation_dir, cur_roi + ".json")
        cell_annotation_dict = None
        with open(cell_annotation_path) as fp:
            cell_annotation_dict = json.load(fp)
        cell_lst = cell_annotation_dict["shapes"]
        for ind, cur_cell in enumerate(cell_lst):
            cell_label = cur_cell["label"]
            cur_points = cur_cell["points"][0]
            point_h = int(math.floor(cur_points[1] + 0.5))
            point_w = int(math.floor(cur_points[0] + 0.5))
            point_val = roi_seg_mask[point_h, point_w]
            # get mask
            inst_map = np.array(roi_seg_mask == point_val, np.uint8)
            y1, y2, x1, x2  = bounding_box(inst_map)
            y1 = y1 - 2 if y1 - 2 >= 0 else y1
            x1 = x1 - 2 if x1 - 2 >= 0 else x1
            x2 = x2 + 2 if x2 + 2 <= roi_seg_mask.shape[1] - 1 else x2
            y2 = y2 + 2 if y2 + 2 <= roi_seg_mask.shape[0] - 1 else y2
            inst_cell_crop = inst_map[y1:y2, x1:x2]
            cell_mask = np.zeros((inst_cell_crop.shape[0], inst_cell_crop.shape[1], 3), dtype=np.uint8)
            contours, hierarchy = cv2.findContours(inst_cell_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
            cell_cnt = contours[0]
            cv2.drawContours(cell_mask, contours=[cell_cnt, ], contourIdx=0, color=(1, 1, 1), thickness=-1)
            cell_img = roi_img[y1:y2, x1:x2]
            mask_cell_img = cell_img * cell_mask
            cur_cell_type_dir = os.path.join(cell_fea_dir, cell_label)
            if not os.path.exists(cur_cell_type_dir):
                os.makedirs(cur_cell_type_dir)
            cur_cell_type_name = cur_roi + "-X" + str(point_w) + "-Y" + str(point_h)
            cur_cell_type_path = os.path.join(cur_cell_type_dir, cur_cell_type_name + ".png")
            io.imsave(cur_cell_type_path, mask_cell_img)

            # cell features
            cell_area = cv2.contourArea(cell_cnt)
            cell_intensity = np.mean(cv2.mean(cell_img, mask=cell_mask[:,:,0])[:3])
            # Roundness
            cnt_perimeter = cv2.arcLength(cell_cnt, True)
            cell_circularity = 4 * 3.14 * cell_area / (cnt_perimeter * cnt_perimeter)
            # add features
            cell_labels.append(cell_label)
            cell_areas.append(cell_area)
            cell_intensities.append(cell_intensity)
            cell_circularities.append(cell_circularity)

    cell_fea_df = pd.DataFrame(list(zip(cell_labels, cell_areas, cell_intensities, cell_circularities)),
        columns =["Label", "Area", "Intensity", "Roundness"])
    cell_fea_path = os.path.join(cell_fea_dir, "cell_feas.csv")
    cell_fea_df.to_csv(cell_fea_path, index=False)
    