# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re

import numpy as np
import cv2
from osgeo import gdal
from skimage import measure
from xml.dom.minidom import Document

from config.configs import *
from utils.tools import *
from utils.convex_hull import array2vector
from utils import geo_transform as trs


class EnviLabel(object):

    def __init__(self, par=None, **pars):

        """ 用来自ENVI的标注文件定义EnviLabel对象，标注文件可选xml或tif格式。

        Args:
            par: 实例化EnviLabel对象的数据来源，可选两种方式或缺省
                1. EnviLabel(path), path: str, 来自标注文件路径，文件格式包括：
                    *.xml(来自ENVI标注)、*.tif(来自ENVI标注)、*.png、*.jpg
                2. EnviLabel(regions), regions: dict, 所有Region的字典文件；
                    {"shape": (,), "proj": str, "regions": {"ROI #1": [[,],]}}
            pars: 关键字格式
                1. EnviLabel(path='/exp.xml')
                2. EnviLabel(regions={})
                3. EnviLabel(shape=(400, 400)), shape: int or tuple, 标注文件的大小
        """

        self._regions = {}
        self._regions_splited = {}
        self._proj = "none"
        self._with_projcs = False
        self._geotrans = None
        self._raster_data = None
        self._raster_data_splited = None
        self._height = None
        self._width = None

        if isinstance(par, str):
            if not os.path.exists(par):
                raise FileNotFoundError("Wrong *.xml file path: {}".format(par))
            self._extract_info(par)
        elif isinstance(par, dict):
            self._define_with_dict(par)
        elif par is None:
            pass
        else:
            print("Warning! Cannot parse type: {}".format(type(par)))

        for k, p in pars.items():
            if k not in ["path", "regions", "shape"]:
                print("Warning! No parameter named: {}".format(k))
            else:
                if k == "path":
                    self._extract_info(p)
                elif k == "regions":
                    self._define_with_dict(p)
                elif k == "shape":
                    self.set_shape(p)

        assert self._regions, "No region found."

    def _extract_info(self, path):

        ext = os.path.splitext(path)[1]
        if ext == ".xml":
            self._extract_from_xml(path)
        elif ext in [".tif", ".png", ".jpg"]:
            self._extract_from_img(path)
        else:
            raise ValueError("Cannot parse file type: {}".format(ext))

    def _extract_from_xml(self, path):

        """
        从.xml文件中提取ROI信息

        Args:
            path: str 来自ENVI的xml格式标注文件路径

            self._regions: dict, 所有Region的多边形信息
                     {'rgn_name': [[(x_1, y_1), (x_2, y_2), (x_3, y_3)], ]}
        """

        def str_to_tuple(plg_ls):
            rgn = []
            for plg in plg_ls:
                coords = plg.split(' ')
                x_list = [float(_) for _ in coords[::2]]
                y_list = [float(_) for _ in coords[1::2]]
                rgn.append(list(zip(x_list, y_list)))
            return rgn

        name_pattern = re.compile(r'name="(.*)"\s')
        proj_pattern = re.compile(r'<CoordSysStr>(.*)</CoordSysStr>')
        coord_pattern = re.compile(r'<Coordinates>\n([-*\d*.\d*\s?]+)\n\s*</Coordinates>')

        with open(path, 'r') as rgn_f:
            xml_info_str = rgn_f.read()
            self._proj = re.findall(proj_pattern, xml_info_str)[0]
            if not self._proj:
                self._proj = "none"
            if "PROJCS" in self._proj:
                self._with_projcs = True

            region_list = xml_info_str.split(r'</Region>')
            for region_str in region_list:
                name = re.findall(name_pattern, region_str)
                if len(name) == 0:
                    continue
                coord_str_list = re.findall(coord_pattern, region_str)
                self._regions[name[0]] = str_to_tuple(coord_str_list)

    def _define_with_dict(self, dicts):
        pass

    def _extract_from_img(self, path):

        # todo: 添加对地理坐标的适配

        # reading
        ext = os.path.splitext(path)[1]
        if ext == ".tif":
            dataset = gdal.Open(path)
            if dataset is None:
                raise ("Read nothing from {}".format(path))

            self._proj = dataset.GetProjection()
            self._geotrans = dataset.GetGeoTransform()
            if not self._proj:
                self._proj = "none"
            if "PROJCS" in self._proj:
                self._with_projcs = True
            self._raster_data = dataset.ReadAsArray()
        else:
            self._raster_data = cv2.imread(path)

        h, w = self._raster_data.shape[-2:]
        if self._width is not None and self._height is not None \
                and (self._height != h or self._width != w):
            print("Warning! The shape ({0}, {1}) is replaced by ({0}, {1})"
                  .format(self._height, self._width,h, w))
        self._height, self._width = h, w

        # extraction
        num_rgn = np.max(self._raster_data)
        for i in range(1, num_rgn+1):
            rgn_name = "ROI #{}".format(i)
            self._regions[rgn_name] = []
            rgn_arr = self._raster_data == i  # True or False
            cur_lab = measure.label(rgn_arr, connectivity=1)  # Integers
            num_plg = np.max(cur_lab)
            for j in range(1, num_plg+1):
                plg_arr = cur_lab == j  # True or False
                plg_row_min, plg_col_min, plg_row_max, plg_col_max = get_extent(np.argwhere(plg_arr))
                plg_arr = plg_arr[plg_row_min:plg_row_max+1, plg_col_min:plg_col_max+1]
                plg = np.array(array2vector(plg_arr)) + [plg_row_min, plg_col_min]
                plg = [list(_[::-1]) for _ in plg]  # points [[row, col], ...] --> [[x, y], ...]
                self._regions[rgn_name].append(plg)

    def _create_array(self, split=False):
        if self._width is None or self._height is None:
            raise Exception("Size not found, please use the following function: set_shape().")
        i = 1
        data = np.zeros((self._height, self._width), dtype=np.uint8)
        for plg_list in self._regions.values():
            for plg in plg_list:
                points = np.array(plg[:-1], dtype='int32')[np.newaxis]
                cv2.fillPoly(data, points, i)
                i += split
            i += (1-split)
        if split:
            self._raster_data_splited = data.copy()
        else:
            self._raster_data = data.copy()
        del data

    def _geo_transformer(self, regions, dataset):

        """ 将各区域的地理/经纬度坐标转换为图像坐标

        Args:
            regions: dict, 区域数据
            dataset: GDAL数据集, 与envi_label对应的影像数据集
        Return:
            regions/new_regions: dict, 转换为图像坐标后的区域
        """

        if self._proj != "none" and not self._raster_data:
            new_regions = {}
            if not self._geotrans or self._geotrans == dataset.GetGeoTransform():
                for name, region in regions.items():
                    r = np.array(region)
                    trs_func = trs.geo2imagexy if self._with_projcs else trs.lonlat2imagexy
                    new_r = list(map(trs_func, [dataset]*len(r), r[:, 0], r[:, 1]))
                    new_regions[name] = new_r
            else:
                raise ValueError("GeoTransform not match.")
            return new_regions
        return regions

    def set_shape(self, *pars):
        if not len(pars):
            raise Exception("Should deliver one or two integers.")
        elif len(pars) < 2:
            if pars[0] < 3:
                raise ValueError("Size should equal to or greater than 3.")
            self._height = self._width = pars[0]
        else:
            if pars[0] < 3 or pars[1] < 3:
                raise ValueError("Width and height should equal to or greater than 3.")
            self._height, self._width = pars[:2]

    def array(self, split=False):
        if split:
            if self._raster_data_splited is None:
                pass  # todo: 添加数据分离形式
            return self._raster_data_splited
        if self._raster_data is None:
            self._create_array(False)
        return self._raster_data

    def regions(self, split=False, dataset=None):
        if split:
            if len(self._regions_splited) == 0:
                for r_n, regs in self._regions.items():
                    for i, reg in enumerate(regs):
                        self._regions_splited["%s_%d" % (r_n, i+1)] = reg
            if dataset:
                return self._geo_transformer(self._regions_splited, dataset)
            return self._regions_splited
        if dataset:
            return self._geo_transformer(self._regions, dataset)
        return self._regions

    def projection(self):
        return self._proj

    def geotransform(self):
        return self._geotrans

    def with_projcs(self):
        return self._with_projcs

    def write_xml(self, path, split=False):

        doc = EnviDoc()
        root = doc.createElement('RegionsOfInterest')
        root.setAttribute("version", "1.0")
        doc.appendChild(root)
        if split:
            rgns = self._regions_splited
        else:
            rgns = self._regions
        for i, (k, v) in enumerate(rgns.items()):
            rgn = doc.createElement("Region")
            root.appendChild(rgn)
            clr = COLORS[i % len(COLORS)]
            rgn.setAttribute("name", k)
            rgn.setAttribute("color", clr)
            geo = doc.createElement("GeometryDef")
            rgn.appendChild(geo)
            crs = doc.add_one_node(geo, "CoordSysStr", self._proj)
            for plg in v:
                doc.add_one_polygon(geo, plg)

        with open(path, 'wb') as f:
            f.write(doc.toprettyxml(indent='\t', encoding='UTF-8'))

    def write_img(self, path, split=False):
        pass


class EnviDoc(Document):

    def __init__(self):
        super(EnviDoc, self).__init__()
        pass

    def add_one_node(self, parent, name, content=None):
        info = self.createElement(name)
        if content is not None:
            info_content = self.createTextNode(str(content))
            info.appendChild(info_content)
        parent.appendChild(info)
        return info

    def add_one_polygon(self, parent, polygon):
        plg = self.add_one_node(parent, "Polygon")
        ext = self.add_one_node(plg, "Exterior")
        lnr = self.add_one_node(ext, "LinearRing")
        plg_str = ""
        for p in polygon:
            plg_str += "{} {} ".format(p[0], p[1])
        crd = self.add_one_node(lnr, "Coordinates", plg_str.strip())
        return crd