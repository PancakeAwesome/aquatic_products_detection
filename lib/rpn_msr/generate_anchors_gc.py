#-*- coding:utf-8 -*-
# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import numpy as np

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])
#ratios=[0.5, 1, 2]表示1：2,1：1,2：1。
#scales=2**np.arange(3, 6)表示2^3,2^4,2^5分别为(8,16,32)
#结果：
'''
[[ -84.  -40.   99.   55.]
 [-176.  -88.  191.  103.]
 [-360. -184.  375.  199.]
 [ -56.  -56.   71.   71.]
 [-120. -120.  135.  135.]
 [-248. -248.  263.  263.]
 [ -36.  -80.   51.   95.]
 [ -80. -168.   95.  183.]
 [-168. -344.  183.  359.]]'''
def generate_anchors(base_size=16, ratios=[0.5, 1, 2],
                     scales=2**np.arange(3, 6)):
    """
    Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, 15, 15) window.
    """
    #新建一个数组：base_anchor:[0 0 15 15]
    base_anchor = np.array([1, 1, base_size, base_size]) - 1
    #枚举各种宽高比，生成三个比例的anchor
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in xrange(ratio_anchors.shape[0])])
    return anchors
#返回anchor的宽，高，x中心，y中心
def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr
#建立anchor
def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """
    #ws与hs都为（3,），np.newaxis操作后都变为（3,1）大小
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    #hstack是将numpy数组进行组合，_ratio_enum组合结果：
    #[[ -3.5   2.   18.5  13. ]
    #[  0.    0.   15.   15. ]
    #[  2.5  -3.   12.5  18. ]]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors
#anchor为[0 0 15 15],ratios为[0.5, 1, 2]
def _ratio_enum(anchor, ratios):
    #该函数就是要生成1：2,1：1，2：1的anchor
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """
    #返回宽高和中心坐标，w:16,h:16,x_ctr:7.5,y_ctr:7.5
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    #计算一个基础size，w*h=256
    size = w * h
    #得到比例的size_ratios，type：np.array
    #为（512,256,128）
    size_ratios = size / ratios
    #np.sqrt开方
    #ws:[23 16 11]  与  hs:[12 16 22]
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors
#_scale_enum就是将一个anchor扩展scale倍，scale是一个数组
#也就是说一个anchor可以生成scale.shape[0]个anchor
def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

if __name__ == '__main__':
    import time
    t = time.time()
    a = generate_anchors()
    print time.time() - t
    print a
    #from IPython import embed; embed()