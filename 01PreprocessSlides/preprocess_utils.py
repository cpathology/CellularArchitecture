# -*- coding: utf-8 -*-

import os, sys
import numpy as np
import itertools, struct
import xml.etree.ElementTree as ET


def get_splitting_coors(wsi_w, wsi_h, block_size):
    def split_coor(ttl_len, sub_len):
        split_num = int(np.floor((ttl_len + sub_len / 2) / sub_len))
        split_pair = [(num * sub_len, sub_len) for num in range(split_num-1)]
        split_pair.append(((split_num-1) * sub_len, ttl_len - (split_num-1) * sub_len))
        return split_pair

    w_pairs = split_coor(wsi_w, block_size)
    h_pairs = split_coor(wsi_h, block_size)
    coors_list = [(ele[0][0], ele[1][0], ele[0][1], ele[1][1]) for ele in list(itertools.product(w_pairs, h_pairs))]

    return coors_list


def parse_imagescope_rects(xml_path):
    xml_tree = ET.parse(xml_path)
    region_num = len(xml_tree.findall('.//Region'))
    roi_boxes = []
    for idx in range(region_num):
        region_xml = xml_tree.findall('.//Region')[idx]
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

    return roi_boxes


def numpy2tiff(image, path):
    def tiff_tag(tag_code, datatype, values):
        types = {'<H': 3, '<L': 4, '<Q': 16}
        datatype_code = types[datatype]
        number = 1 if isinstance(values, int) else len(values)
        if number == 1:
            values_bytes = struct.pack(datatype, values)
        else:
            values_bytes = struct.pack('<' + (datatype[-1:] * number), *values)
        tag_bytes = struct.pack('<HHQ', tag_code, datatype_code, number) + values_bytes
        tag_bytes += b'\x00' * (20 - len(tag_bytes))
        return tag_bytes

    image_bytes = image.shape[0] * image.shape[1] * image.shape[2]
    with open(path, 'wb+') as f:
        image.reshape((image_bytes,))
        f.write(b'II')
        f.write(struct.pack('<H', 43))  # Version number
        f.write(struct.pack('<H', 8))  # Bytesize of offsets
        f.write(struct.pack('<H', 0))  # always zero
        f.write(struct.pack('<Q', 16 + image_bytes))  # Offset to IFD
        for offset in range(0, image_bytes, 2 ** 20):
            f.write(image[offset:offset + 2 ** 20].tobytes())
        f.write(struct.pack('<Q', 8))  # Number of tags in IFD
        f.write(tiff_tag(256, '<L', image.shape[1]))  # ImageWidth tag
        f.write(tiff_tag(257, '<L', image.shape[0]))  # ImageLength tag
        f.write(tiff_tag(258, '<H', (8, 8, 8)))  # BitsPerSample tag
        f.write(tiff_tag(262, '<H', 2))  # PhotometricInterpretation tag
        f.write(tiff_tag(273, '<H', 16))  # StripOffsets tag
        f.write(tiff_tag(277, '<H', 3))  # SamplesPerPixel
        f.write(tiff_tag(278, '<Q', image_bytes // 8192))  # RowsPerStrip
        f.write(tiff_tag(279, '<Q', image_bytes))  # StripByteCounts
        f.write(struct.pack('<Q', 0))  # Offset to next IFD