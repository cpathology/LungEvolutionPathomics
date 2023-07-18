# -*- coding: utf-8 -*-

import os, sys
import xml.etree.ElementTree as ET
import numpy as np


def parse_imagescope_rects(xml_path):
    xml_tree = ET.parse(xml_path)
    anno_dict = {}
    annotations_xml = xml_tree.findall('.//Annotation')
    anno_num = len(annotations_xml)
    for ianno in range(anno_num):
        annotation_xml = annotations_xml[ianno]
        annotation_name = annotation_xml.attrib["Name"]
        regions = annotation_xml.findall('.//Region')
        region_num = len(regions)
        roi_boxes = []
        for idx in range(region_num):
            region_xml = regions[idx]
            vertices = []
            for vertex_xml in region_xml.findall('.//Vertex'):
                attrib = vertex_xml.attrib
                vertices.append([float(attrib['X']) + 0.5,
                                float(attrib['Y']) + 0.5])
            vertices = np.asarray(vertices, dtype=np.int32)
            if vertices.shape[0] != 4 or vertices.shape[1] != 2:
                continue
            xs, ys = vertices[:, 0], vertices[:, 1]
            w_start, h_start = np.min(xs), np.min(ys)
            w_len = np.max(xs) - np.min(xs)
            h_len = np.max(ys) - np.min(ys)
            if w_len <= 0 or h_len <= 0:
                continue
            roi_boxes.append([(w_start, h_start), (w_len, h_len)])
        anno_dict[annotation_name] = roi_boxes

    return anno_dict