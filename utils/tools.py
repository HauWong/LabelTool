# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import cv2


def rc2xy(box):

    """
    (row_min, row_max, col_min, col_max) -> (x_min, y_min, x_max, y_max)
    """

    return box[2], box[0], box[3], box[1]


def xy2rc(box):

    """
    (x_min, y_min, x_max, y_max) -> (row_min, row_max, col_min, col_max)
    """

    return box[1], box[3], box[0], box[2]


def get_extent(points):

    """

    Args:
        points: np.ndarray, 点集
    Return:
        边界点集
    """
    points = np.array(points)
    min_0 = points[:, 0].min()
    max_0 = points[:, 0].max()
    min_1 = points[:, 1].min()
    max_1 = points[:, 1].max()
    return min_0, min_1, max_0, max_1


def limit_boundary(rect, boundary):

    """ 限制矩形边界

    Args:
        rect: (xmin, ymin, xmax, ymax)
        boundary: (xmin, ymin, xmax, ymax)

    """

    res = list(rect)
    if rect[0] < boundary[0]:
        res[0] = boundary[0]
    if rect[1] < boundary[1]:
        res[1] = boundary[1]
    if rect[2] > boundary[2]:
        res[2] = boundary[2]
    if rect[3] > boundary[3]:
        res[3] = boundary[3]
    return tuple(res)


def paint_rectangles(image, box_ls, color=(0, 0, 255), width=2, show_conf=True):

    """ 在图像上批量绘制矩形框

    Args：
        image: array, 图像数组, shape=(w, h, c)
        box_ls: list, 矩形框列表, [(prob, xmin, ymin, xmax, ymax), ...]
        color: tuple, 8-bit RGB形式的框体颜色
        width: int, 框线宽度
        show_conf: bool, 是否显示置信度
    Return：
        image: array, 结果图像
    """

    img_w, img_h, _ = image.shape
    boxes = np.array(box_ls)
    confs, bboxes = boxes[:, :-4], boxes[:, -4:]
    for i, box in enumerate(bboxes):
        first_point = (int(box[0]), int(box[1]))  # 正确方式
        last_point = (int(box[2]), int(box[3]))
        if show_conf and confs[i]:
            cv2.putText(image, '%.3f' % box[0], first_point, cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, width)
        cv2.rectangle(image, first_point, last_point, color, width)
    return image
