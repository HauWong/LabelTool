# !/usr/bin/env python
# -*- coding: utf-8 -*-

import re

from osgeo import osr


def define(description):

    sr = osr.SpatialReference()
    if type(description) == str:
        proj_name_pat = re.compile(r'PROJCS\["(.*)",GEO')
        geog_name_pat = re.compile(r'GEOGCS\["(.*)",DAT')
        proj_name = proj_name_pat.search(description)
        geog_name = geog_name_pat.search(description)
        if geog_name:
            if "WGS" in geog_name and "1984" in geog_name:
                sr.SetWellKnownGeogCS("WGS84")
                no = proj_name.split("_")[-1][:-1]
                ns = True if proj_name.split("_")[-1][-1] == "N" else False
                sr.SetUTM(no, ns)
            else:
                pass
        else:
            pass
    else:
        pass

    return sr