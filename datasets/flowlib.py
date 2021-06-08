#!/usr/bin/python
"""
# ==============================
# flowlib.py
# library for optical flow processing
# Author: Ruoteng Li
# Date: 6th Aug 2016
# ==============================
"""
import numpy as np
from PIL import Image
import cv2

UNKNOWN_FLOW_THRESH = 1e7
SMALLFLOW = 0.0
LARGEFLOW = 1e8
"""
=============
Flow Section
=============
"""


def read_flow(filename):
    """
    read optical flow data from flow file
    :param filename: name of the flow file
    :return: optical flow data in numpy array (dtype: np.float32)
    """
    if filename.endswith('.flo'):
        flow = read_flo_file(filename)
    else:
        raise Exception('Invalid flow file format!')

    return flow


def write_flow(flow, filename):
    """
    write optical flow in Middlebury .flo format
    :param flow: optical flow map
    :param filename: optical flow file path to be saved
    :return: None
    """
    f = open(filename, 'wb')
    magic = np.array([202021.25], dtype=np.float32)
    (height, width) = flow.shape[0:2]
    w = np.array([width], dtype=np.int32)
    h = np.array([height], dtype=np.int32)
    magic.tofile(f)
    w.tofile(f)
    h.tofile(f)
    flow.tofile(f)
    f.close()


def save_flow_image(flow, image_file):
    """
    save flow visualization into image file
    :param flow: optical flow data
    :param flow_fil
    :return: None
    """
    # print flow.shape
    flow_img = flow_to_image(flow)
    img_out = Image.fromarray(flow_img)
    img_out.save(image_file)


def flowfile_to_imagefile(flow_file, image_file):
    """
    convert flowfile into image file
    :param flow: optical flow data
    :param flow_fil
    :return: None
    """
    flow = read_flow(flow_file)
    save_flow_image(flow, image_file)


def flow_error(tu, tv, u, v):
    """
    Calculate average end point error
    :param tu: ground-truth horizontal flow map
    :param tv: ground-truth vertical flow map
    :param u:  estimated horizontal flow map
    :param v:  estimated vertical flow map
    :return: End point error of the estimated flow
    """
    smallflow = 0.0
    '''
    stu = tu[bord+1:end-bord,bord+1:end-bord]
    stv = tv[bord+1:end-bord,bord+1:end-bord]
    su = u[bord+1:end-bord,bord+1:end-bord]
    sv = v[bord+1:end-bord,bord+1:end-bord]
    '''
    stu = tu[:]
    stv = tv[:]
    su = u[:]
    sv = v[:]

    idxUnknow = (abs(stu) > UNKNOWN_FLOW_THRESH) | (
        abs(stv) > UNKNOWN_FLOW_THRESH)
    stu[idxUnknow] = 0
    stv[idxUnknow] = 0
    su[idxUnknow] = 0
    sv[idxUnknow] = 0

    ind2 = [(np.absolute(stu) > smallflow) | (np.absolute(stv) > smallflow)]
    index_su = su[ind2]
    index_sv = sv[ind2]
    an = 1.0 / np.sqrt(index_su**2 + index_sv**2 + 1)

    index_stu = stu[ind2]
    index_stv = stv[ind2]
    tn = 1.0 / np.sqrt(index_stu**2 + index_stv**2 + 1)
    '''
    angle = un * tun + vn * tvn + (an * tn)
    index = [angle == 1.0]
    angle[index] = 0.999
    ang = np.arccos(angle)
    mang = np.mean(ang)
    mang = mang * 180 / np.pi
    '''

    epe = np.sqrt((stu - su)**2 + (stv - sv)**2)
    epe = epe[ind2]
    mepe = np.mean(epe)
    return mepe


def flow_to_image(flow, maxrad=-1):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    if maxrad == -1:
        rad = np.sqrt(u**2 + v**2)
        maxrad = max(-1, np.max(rad))

    u = u / (maxrad + np.finfo(float).eps)
    v = v / (maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)


def evaluate_flow_file(gt_file, pred_file):
    """
    evaluate the estimated optical flow end point error according to ground truth provided
    :param gt_file: ground truth file path
    :param pred_file: estimated optical flow file path
    :return: end point error, float32
    """
    # Read flow files and calculate the errors
    gt_flow = read_flow(gt_file)  # ground truth flow
    eva_flow = read_flow(pred_file)  # predicted flow
    # Calculate errors
    average_pe = flow_error(gt_flow[:, :, 0], gt_flow[:, :, 1],
                            eva_flow[:, :, 0], eva_flow[:, :, 1])
    return average_pe


def evaluate_flow(gt_flow, pred_flow):
    """
    gt: ground-truth flow
    pred: estimated flow
    """
    average_pe = flow_error(gt_flow[:, :, 0], gt_flow[:, :, 1],
                            pred_flow[:, :, 0], pred_flow[:, :, 1])
    return average_pe


"""
==============
Others
==============
"""


def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2 + v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a + 1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols + 1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel, 1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0 - 1] / 255
        col1 = tmp[k1 - 1] / 255
        col = (1 - f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nanIdx)))

    return img


def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255 * np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col + YG, 0] = 255 - np.transpose(
        np.floor(255 * np.arange(0, YG) / YG))
    colorwheel[col:col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col + GC, 1] = 255
    colorwheel[col:col + GC, 2] = np.transpose(
        np.floor(255 * np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col + CB, 1] = 255 - np.transpose(
        np.floor(255 * np.arange(0, CB) / CB))
    colorwheel[col:col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col + BM, 2] = 255
    colorwheel[col:col + BM, 0] = np.transpose(
        np.floor(255 * np.arange(0, BM) / BM))
    col += +BM

    # MR
    colorwheel[col:col + MR, 2] = 255 - np.transpose(
        np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col + MR, 0] = 255

    return colorwheel


def read_flo_file(filename):
    """
    Read from Middlebury .flo file
    :param flow_file: name of the flow file
    :return: optical flow data in matrix
    """
    f = open(filename, 'rb')
    magic = np.fromfile(f, np.float32, count=1)
    data2d = None

    if 202021.25 != magic:
        print('Magic number incorrect. Invalid .flo file')
    else:
        w = np.fromfile(f, np.int32, count=1)
        h = np.fromfile(f, np.int32, count=1)
        # print "Reading %d x %d flow file in .flo format" % (h, w)
        data2d = np.fromfile(f, np.float32, count=2 * int(w) * int(h))
        # reshape data into 3D array (columns, rows, channels)
        data2d = np.resize(data2d, (int(h), int(w), 2))
    f.close()
    return data2d


def resize_flow(flow, des_width, des_height, method='bilinear'):
    # improper for sparse flow
    src_height = flow.shape[0]
    src_width = flow.shape[1]
    if src_width == des_width and src_height == des_height:
        return flow
    ratio_height = float(des_height) / float(src_height)
    ratio_width = float(des_width) / float(src_width)
    if method == 'bilinear':
        flow = cv2.resize(
            flow, (des_width, des_height), interpolation=cv2.INTER_LINEAR)
    elif method == 'nearest':
        flow = cv2.resize(
            flow, (des_width, des_height), interpolation=cv2.INTER_NEAREST)
    else:
        raise Exception('Invalid resize flow method!')
    flow[:, :, 0] = flow[:, :, 0] * ratio_width
    flow[:, :, 1] = flow[:, :, 1] * ratio_height
    return flow
