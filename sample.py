# !/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import random
from osgeo import gdal
import argparse

import label
from utils.tools import *
from utils import voc_sample_tool as vst
from config.configs import *


def operate_batch(func, image_dir, label_dir, save_dir, **pars):

    """ 批量操作

    Args:
        func: 批量操作的函数
        image_dir: 原始影像目录
        label_dir: 标记文件目录，要求内部标记文件名与原始影像名对应
        save_dir: 保存目录
        **pars: 与func函数对应的其他参数
    """

    files = os.listdir(label_dir)
    for f in files:
        lab_p = os.path.join(label_dir, f)
        envi_lab = label.EnviLabel(lab_p)
        img_p = os.path.join(image_dir, '%s.tif' % os.path.splitext(f)[0])
        if func.__name__ == "make_voc":
            df_pars = {"display_dir": None, "split": True}
            df_pars.update(pars)
            dsp_p = None
            if df_pars["display_dir"]:
                dsp_p = os.path.join(df_pars["display_dir"], '%s.png' % os.path.splitext(f)[0])
            save_p = os.path.join(save_dir, '%s.xml' % os.path.splitext(f)[0])
            func(envi_lab, img_p, save_p, dsp_p, split=df_pars["split"])
        elif func.__name__ == "cut_with_label":
            par_n = ("mask", "size", "center", "count", "bias", "save_pos")
            df_v = (False, None, False, 1, (0, 100), False)
            df_pars = dict(zip(par_n, df_v))
            df_pars.update(pars)
            save_d = os.path.join(save_dir, os.path.splitext(f)[0])
            func(envi_lab, img_p, save_d, df_pars["mask"], df_pars["size"], df_pars["center"],
                 df_pars["count"], df_pars["bias"], df_pars["save_pos"])


def make_voc(envi_label, image_path, save_path,
             display_path=None, basic_info=None, split=True):

    """ 根据EnviLabel标注生成VOC格式的目标检测样本

    Args:
        envi_label: EnviLabel对象，标注数据
        image_path: str, 原始图像路径
        save_path: str, 保存路径
        display_path: str, 显示路径，默认为None，此时不保存显示图像
        basic_info: dict, 样本基本信息, 默认为None
        split: bool, 标注是否分离, 默认为True
    """

    if basic_info is None:
        basic_info = INFO

    img = cv2.imread(image_path)
    h, w = img.shape[0], img.shape[1]
    dataset = gdal.Open(image_path)
    regions = envi_label.regions(split, dataset)
    boxes = list(map(get_extent, regions.values()))
    boxes = list(map(limit_boundary, boxes, [(0, 0, w, h)]*len(boxes)))

    dir, file_name = os.path.split(image_path)
    folder = dir.split('\\')[-1]

    if display_path:
        res_img = paint_rectangles(img, boxes)
        cv2.imwrite(display_path, res_img)

    new_boxes = list(map(lambda box: {'name': basic_info['name'],
                                      'bndbox': box,
                                      'truncated': basic_info['truncated'],
                                      'difficult': basic_info['difficult']},
                         boxes))
    img_info = {'folder': folder,
                'filename': file_name,
                'size': img.shape if len(img.shape) == 3 else (h, w, 1),
                'segmented': 0}
    vst.write_xml(img_info, new_boxes, save_path)
    print("Saved at %s" % save_path)


def cut_with_label(envi_label, image_path, save_dir,
                   mask=False, size=None, center=False,
                   count=1, bias=(0, 100), save_pos=False):

    """ 根据EnviLabel标注裁剪影像

    Args:
        envi_label: EnviLabel对象，标注数据
        image_path: str, 原始影像路径
        save_dir: str, 保存目录
        mask: bool, 是否利用标注进行掩膜, 默认为True
        size: tuple, 限定裁剪区域的大小, 默认为None, 即以标注区域的最小外接框为限定区域
        center: bool, 是否以区域中心坐标为基准进行裁剪, 默认为False, 即随机偏移裁剪
                若size为None, 则该参数无效
        count: int, 随机裁剪数量, 默认为1,
               若center为True, 则该参数默认为1
        bias: tuple, 随机偏移范围, 默认为(0, 100),
              若center为True, 则该参数无效
        save_pos: bool, 是否保存区域的坐标，默认为False
                  若size为None，则该参数无效
    """

    if not size:
        center = False
    if center:
        count = 1

    img = cv2.imread(image_path)
    h, w = img.shape[0], img.shape[1]
    c = img.shape[2] if len(img.shape) == 3 else 1

    dataset = gdal.Open(image_path)
    regions = envi_label.regions(True, dataset)

    if save_pos:
        pos_f = open(os.path.join(save_dir, "positions.txt"), 'w')
    for r_name, region in regions.items():
        save_name = r_name
        plg_points = np.array(region[:-1], dtype="int32")
        x_min, y_min, x_max, y_max = limit_boundary(get_extent(plg_points), (0, 0, w, h))
        w_rect = x_max - x_min
        h_rect = y_max - y_min

        plg_data = img[y_min:y_max, x_min:x_max].copy()
        x_org_img, y_org_img = x_min, y_min
        tmp_w, tmp_h = x_max-x_min, y_max-y_min
        if mask:
            plg_points -= [x_min, y_min]
            m = np.zeros((h_rect, w_rect), dtype="uint8")
            cv2.polylines(m, [plg_points], True, 255)
            cv2.fillPoly(m, [plg_points], 255)
            plg_data[m == 0] = 0

        if size:
            for i in range(count):
                if i > 0:
                    save_name = "%s_%d" % (r_name, i)
                res = np.zeros((size[1], size[0], c), dtype=img.dtype)

                # 计算结果框与标注框的坐标偏差
                c_rect = int(w_rect/2), int(h_rect/2)
                c_res = int(size[0]/2), int(size[1]/2)
                x_diff = c_res[0] - c_rect[0]
                y_diff = c_res[1] - c_rect[1]
                x_rnd, y_rnd = 0, 0
                if not center:
                    x_rnd = random.randint(bias[0], bias[1])
                    y_rnd = random.randint(bias[0], bias[1])
                x_diff = x_diff+x_rnd if x_diff+x_rnd < size[0] else size[0]-1
                y_diff = y_diff+y_rnd if y_diff+y_rnd < size[1] else size[1]-1

                tmp_w, tmp_h = size
                if mask:
                    x_org = x_diff if x_diff >= 0 else 0
                    y_org = y_diff if y_diff >= 0 else 0
                    x_org_plg = -x_diff if x_diff < 0 else 0
                    y_org_plg = -y_diff if y_diff < 0 else 0
                    tmp_w = size[0]-x_org if (x_org_plg+size[0])<(x_org+w_rect) else w_rect-x_org_plg
                    tmp_h = size[1]-y_org if (y_org_plg+size[1])<(y_org+h_rect) else h_rect-y_org_plg
                    res[y_org:y_org+tmp_h, x_org:x_org+tmp_w] = plg_data[y_org_plg:y_org_plg+tmp_h, x_org_plg:x_org_plg+tmp_w]
                    x_org_img = x_min + x_org_plg
                    y_org_img = y_min + y_org_plg
                else:
                    x_org_img = x_min-x_diff if x_min-x_diff > 0 else 0
                    y_org_img = y_min-y_diff if y_min-y_diff > 0 else 0
                    x_org_img = w-size[0] if x_org_img+size[0] > w else x_org_img
                    y_org_img = h-size[1] if y_org_img+size[1] > h else y_org_img
                    res[0:size[1], 0:size[0]] = img[y_org_img:y_org_img+size[1], x_org_img:x_org_img+size[0]]

                save_path = os.path.join(save_dir, '%s.tif' % save_name)
                cv2.imwrite(save_path, res)
                print("Saved at %s" % save_path)
        else:
            save_path = os.path.join(save_dir, '%s.tif' % save_name)
            cv2.imwrite(save_path, plg_data)
            print("Saved at %s" % save_path)
        if save_pos:
            pos_f.write("%s %d %d %d %d\n" % (save_name, x_org_img, y_org_img, tmp_w, tmp_h))
    if save_pos:
        pos_f.close()


def divide_samples(file_dir, save_dir, name='',
                   types=('train', 'val', 'test'),
                   percentages=(0.6, 0.3, 0.1),
                   shuffle=True):
    """ Divide samples into several sets (train, val, test), and save them into some text files

    Args:
        file_dir: str, directory of all samples
        save_dir: str, directory for saving divided results
        name: str, custom name
        types: tuple, types of sample set ready to divide
        percentages: tuple, percentages corresponding types above
        shuffle: bool, shuffle or not shuffle the original samples
    """
    if not os.path.isdir(file_dir):
        raise NotADirectoryError('Please check out the directory: %s'
                                 % file_dir)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
        print('Create a new directory: %s' % save_dir)

    for root, dirs, files in os.walk(file_dir):
        if shuffle:
            random.shuffle(files)
            print('Shuffled')

        count = len(files)
        nums = []
        for i in range(len(percentages)-1):
            nums.append(int(percentages[i]*count))
        nums.append(count-sum(nums))
        print('Total: %d' % count)
        print('Types:', types)
        print('Numbers:', nums)

        idx = 0
        for type_i in range(len(nums)):
            file_name = os.path.join(save_dir, '%s_%s.txt' % (name, types[type_i]))
            end_idx = idx+nums[type_i]
            with open(file_name, 'w') as f:
                while idx < end_idx:
                    sample_name = files[idx].split('.')[0]
                    f.write('%s\n' % sample_name)
                    idx += 1


if __name__ == "__main__":

    def str_to_bool(v):
        return v.lower() in ("true", "t", "yes", "y", "1")

    parser = argparse.ArgumentParser()
    parser.add_argument('-f', dest="func", type=str, choices=["bs", "s", "bc", "c", "d"],
                        help="Choose function from cut_with_label[c], make_samples_from_batch[bs],"
                             "make_voc[s] and divide_samples[d].")
    parser.add_argument('-i', dest="img", type=str, help="Image path or image directory")
    parser.add_argument('-l', dest="lab", type=str, default=None, help="[bs][s][bc][c]Label path or label directory")
    parser.add_argument('-s', dest="save", type=str, help="Save path or save directory")

    parser.add_argument('-d', dest="disp", type=str, default=None, help="[bs][s]Display path or display directory")
    parser.add_argument('-sp', dest="split", type=str, default=True, help="[bs][s]Split or not")

    parser.add_argument('--mask', type=str_to_bool, default=True, help="[bc][c]Mask or not")
    parser.add_argument('--size', nargs=2, type=int, default=None, help="[bc][c]Size of cutting region")
    parser.add_argument('--center', type=str_to_bool, default=False, help="[bc][c]Center of cutting region")
    parser.add_argument('--count', type=int, default=1, help="[bc][c]Count of cutting regions")
    parser.add_argument('--bias', nargs=2, type=float, default=(0, 100), help="[bc][c]Bias of center of cutting region")
    parser.add_argument('--pos', type=bool, default=False, help="[bc][c]Save positions of region or not")

    parser.add_argument('--name', type=str, default='', help="[d]Category name")
    parser.add_argument('--types', nargs=2, type=str, default=('train', 'val', 'test'),
                        help="[d]The purpose of the sample set")
    parser.add_argument('--pct', nargs=2, type=float, default=(0.6, 0.3, 0.1),
                        help="[d]The proportion of the sample set")
    parser.add_argument('--shuffle', type=str_to_bool, default=True, help="[d]Shuffle or not")
    args = parser.parse_args()

    img = args.img
    lab = args.lab
    save = args.save

    if args.func == "bc":
        if not lab:
            raise ValueError("Must give one label to cut at least.")
        mask, size, center, count, bias, pos = args.mask, args.size, args.center, args.count, args.bias, args.pos
        operate_batch(cut_with_label, img, lab, save,
                      mask=mask, size=size, center=center,
                      count=count, bias=bias, save_pos=pos)
    elif args.func == "c":
        if not lab:
            raise ValueError("Must give one label to cut at least.")
        mask, size, center, count, bias, pos = args.mask, args.size, args.center, args.count, args.bias, args.pos
        envi_lab = label.EnviLabel(lab)
        cut_with_label(envi_lab, img, save, mask, size, center, count, bias, pos)
    elif args.func == "bs":
        if not lab:
            raise ValueError("Must give one label to make sample at least.")
        disp = args.disp
        sp = args.split
        operate_batch(make_voc, img, lab, save,
                      display_dir=disp, split=sp)
    elif args.func == "s":
        if not lab:
            raise ValueError("Must give one label to make VOC sample.")
        disp = args.disp
        sp = args.split
        envi_lab = label.EnviLabel(lab)
        make_voc(envi_lab, img, save, display_path=disp, split=sp)
    elif args.func == "d":
        name, types, pct, shuffle = args.name, args.types, args.pct, args.shuffle
        divide_samples(img, save, name, types, pct, shuffle)
    else:
        raise ValueError("Must assign a method.")