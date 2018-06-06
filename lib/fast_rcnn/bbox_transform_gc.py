# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

import numpy as np
#函数作用：返回anchor相对于GT的（dx,dy,dw,dh）四个回归值，shape（len（anchors），4）
def bbox_transform(ex_rois, gt_rois):
    #计算每一个anchor的width与height
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    #计算每一个anchor中心点x，y坐标
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights
    #注意：当前的GT不是最一开始传进来的所有GT，而是与对应anchor最匹配的GT，可能有重复信息
    #计算每一个GT的width与height
    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    #计算每一个GT的中心点x，y坐标
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights
    #要对bbox进行回归需要4个量，dx、dy、dw、dh，分别为横纵平移量、宽高缩放量
    #此回归与fast-rcnn回归不同，fast要做的是在cnn卷积完之后的特征向量进行回归，dx、dy、dw、dh都是对应与特征向量
    #此时由于是对原图像可视野中的anchor进行回归，更直观
    #定义 Tx=Pwdx(P)+Px Ty=Phdy(P)+Py Tw=Pwexp(dw(P)) Th=Phexp(dh(P))
    #P为anchor，T为target，最后要使得T～G，G为ground-True
    #回归量dx(P)，dy(P)，dw(P)，dh(P)，即dx、dy、dw、dh
    targets_dx = (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)
    #targets_dx, targets_dy, targets_dw, targets_dh都为（anchors.shape[0]，）大小
    #所以targets为（anchors.shape[0]，4）
    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets
#boxes为anchor信息，deltas为'rpn_bbox_pred'层信息
#函数作用:得到改善后的anchor的信息（x1,y1,x2,y2）
def bbox_transform_inv(boxes, deltas):
    #boxes.shape[0]=K*A=Height*Width*A
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)
    # boxes/anchors:[x1, y1, x2, y2]
    #得到Height*Width*A个anchor的宽，高，中心点的x，y坐标
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights
    #deltas本来就只有4列，依次存（dx,dy,dw,dh）,每一行表示一个anchor
    #0::4表示先取第一个元素，以后每4个取一个，所以取的index为（0,4,8,12,16...），但是deltas本来就只有4列，所以只能取到一个值
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]
    #预测后的中心点，与w与h
    #先做平移变换
    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    # 再做缩放
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]
    #预测后的（x1,y1,x2,y2）存入 pred_boxes
    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes
#函数作用：使得boxes位于图片内
def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries.
    """
    #im_shape[0]为图片高，im_shape[1]为图片宽
    #使得boxes位于图片内
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes