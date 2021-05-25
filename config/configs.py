# !/usr/bin/env python
# -*- coding: utf-8 -*-

COLORS = ["255,0,0", "0,255,0", "0,0,255", "255,255,0", "255,0,255", "0,255,255"]

DIRECTS = {0: (-1, -1), 1: (-1, 0), 2: (-1, 1), 3: (0, 1),
           4: (1, 1), 5: (1, 0), 6: (1, -1), 7: (0, -1)}

INFO = {"name": "positive", "truncated": False, "difficult": False}  # 默认样本基本信息