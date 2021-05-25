# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np

from config.configs import *
from utils.link_table import DoubleLinkTable


class Compass(object):

    def __init__(self):

        self.is_corner = False
        self.direct = (0, 1)
        self._d_idx = 4
        self._array = np.zeros((3, 3), dtype=np.uint8)

    def update(self, window):
        self._array = window
        self.__seek()

    def __seek(self):

        if self._array[1, 1] != 1:
            self.is_corner = False
            return

        a = self._array
        loosen = [a[0, 0], a[0, 1], a[0, 2], a[1, 2],
                  a[2, 2], a[2, 1], a[2, 0], a[1, 0]]
        view_link = DoubleLinkTable(loosen)
        view = (self._d_idx+4) % 8
        node = view_link.get_node(view)
        if node.next.value == 0:
            node = node.next
            view = (view+1) % 8
            while node.value == 0:
                node = node.next
                view = (view+1) % 8
        else:
            node = node.prev
            view = (view-1) % 8
            while node.value == 0:
                node = node.prev
                view = (view-1) % 8
        # todo: 添加对短尾线的识别支持

        if self._d_idx != view and node.value:
            self.is_corner = True
            self._d_idx = view
            self.direct = DIRECTS[self._d_idx]
        else:
            self.is_corner = False


def find_lowest(points):
    points = np.array(points)
    idx = np.argmax(points[:, 0])
    return points[idx].tolist()


def calculate_ctan(base_point, end_point):
    delta_x = end_point[1] - base_point[1]
    delta_y = base_point[0] - end_point[0]
    return delta_x/(delta_y + 1e-16)


def sort_points(arr, points):
    if len(arr) < 2:
        return points
    else:
        pivot = arr[0]
        base_point = points[0]
        less = [i for i in arr[1:] if i <= pivot]
        left_points = [points[arr.index(i)] for i in less]
        greater = [i for i in arr[1:] if i > pivot]
        right_points = [points[arr.index(i)] for i in greater]
        return sort_points(less, left_points) + [base_point] + sort_points(greater, right_points)


def is_anticlockwise(sta_point, mid_point, end_point):
    front_vector = [mid_point[0] - sta_point[0], mid_point[1] - sta_point[1]]
    after_vector = [end_point[0] - mid_point[0], end_point[1] - mid_point[1]]
    mul = front_vector[0]*after_vector[1] - after_vector[0]*front_vector[1]
    return mul > 0


def calculate_convex_hull(points):
    """
    根据提供二值图像的角点坐标提取凸包

    Args:
        points: 待提取凸包二值图像的角点列表，
                list([row1, col1], [row2, col2], ...)
    Returns:
         res: 凸包角点列表
              list([row1, col1], ...)
    """

    lowest_point = find_lowest(points)
    points.remove(lowest_point)
    ctan_ls = []
    for point in points:
        if point == lowest_point:
            continue
        ctan_ls.append(calculate_ctan(lowest_point, point))
    points_sorted = sort_points(ctan_ls, points)

    res = [lowest_point, points_sorted[0]]
    for i in range(1, len(points_sorted)):
        point = points_sorted[i]
        while is_anticlockwise(res[-2], res[-1], point):
            res.pop(-1)
            if len(res) <= 2:
                break
        res.append(point)
    res.append(lowest_point)
    return res


def array2vector(array):

    """
    根据实心二值数组提取角点

    Args:
        array: 二维实心二值数组
               array([[0, 0, 1, 1, 0],
                      [0, 1, 1, 1, 1]
                      ...
                      []])
               shape(height, width)
    Returns:
        plg: 角点坐标列表
             list[[row1, col1], ...]

    """
    h, w = array.shape
    corners = []
    cps = Compass()
    p = [0, 0]
    while p not in corners:
        win_arr = np.zeros((3, 3), dtype='uint8')
        for i0 in range(-1, 2):
            for j0 in range(-1, 2):
                if p[0] + i0 < 0 or p[1] + j0 < 0 or p[0] + i0 >= h or p[1] + j0 >= w:
                    continue
                win_arr[i0 + 1, j0 + 1] = array[p[0] + i0, p[1] + j0]
        cps.update(win_arr)
        if cps.is_corner:
            corners.append(p)
        p = [p[0]+cps.direct[0], p[1]+cps.direct[1]]

    return corners


if __name__=="__main__":

    arr = np.array([[0, 0, 0, 1, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 1, 1, 1, 1, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1, 1, 0, 0],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1, 1, 1, 0],
                    [0, 0, 1, 1, 0, 0, 0, 0, 0],
                    [0, 0, 0, 1, 0, 0, 0, 0, 0]])

    h, w = arr.shape

    corners = array2vector(arr)
    # win = Window()
    # p = [0, 0]
    # while p not in corners:
    #     win_arr = np.zeros((3, 3), dtype='uint8')
    #     for i0 in range(-1, 2):
    #         for j0 in range(-1, 2):
    #             if p[0] + i0 < 0 or p[1] + j0 < 0 or p[0] + i0 >= h or p[1] + j0 >= w:
    #                 continue
    #             win_arr[i0 + 1, j0 + 1] = arr[p[0] + i0, p[1] + j0]
    #     win.update(win_arr)
    #     if win.is_corner:
    #         corners.append(p)
    #     p = [p[0]+win.direct[0], p[1]+win.direct[1]]

    print(corners)