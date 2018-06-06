# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick and Sean Bell
# --------------------------------------------------------

import os
import yaml
from fast_rcnn.config import cfg
import numpy as np
import numpy.random as npr
from generate_anchors import generate_anchors
from utils.cython_bbox import bbox_overlaps
from fast_rcnn.bbox_transform import bbox_transform
import pdb

DEBUG = False
#输入分别为rpn_cls_score层输出，GT信息，image信息，输入data，_feat_stride = [16,],anchor_scales = [8, 16, 32]
def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, data, _feat_stride = [16,], anchor_scales = [4 ,8, 16, 32]):
    """
    Assign anchors to ground-truth targets. Produces anchor classification
    labels and bounding-box regression targets.
    """
    _anchors = generate_anchors(scales=np.array(anchor_scales))
    #_num_anchors等于9
    _num_anchors = _anchors.shape[0]

    if DEBUG:
        print 'anchors:'
        print _anchors
        print 'anchor shapes:'
        print np.hstack((
            _anchors[:, 2::4] - _anchors[:, 0::4],
            _anchors[:, 3::4] - _anchors[:, 1::4],
        ))
        _counts = cfg.EPS
        _sums = np.zeros((1, 4))
        _squared_sums = np.zeros((1, 4))
        _fg_sum = 0
        _bg_sum = 0
        _count = 0
    # allow boxes to sit over the edge by a small amount
    #不允许boxes超出图片
    _allowed_border =  0
    # map of shape (..., H, W)
    #height, width = rpn_cls_score.shape[1:3]

    im_info = im_info[0]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate 9 anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the 9 anchors
    # filter out-of-image anchors
    # measure GT overlap
    #rpn_cls_score.shape[0]为1
    assert rpn_cls_score.shape[0] == 1, \
        'Only single item batches are supported'
    #rpn_cls_score.shape的第二位第三位分别存储高与宽
    #rpn_cls_score.shape=[1,height,width,depth],按前提来看，depth应为18,height与width分别为原图高/16,原图宽/16
    # map of shape (..., H, W)
    height, width = rpn_cls_score.shape[1:3]

    if DEBUG:
        print 'AnchorTargetLayer: height', height, 'width', width
        print ''
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])
        print 'height, width: ({}, {})'.format(height, width)
        print 'rpn: gt_boxes.shape', gt_boxes.shape
        print 'rpn: gt_boxes', gt_boxes

    # 1. Generate proposals from bbox deltas and shifted anchors
    #产生横向偏移值，偏移值的个数为width，以600 × 1000的图像为例，会有64个偏移值，因为width=1000/16=64
    shift_x = np.arange(0, width) * _feat_stride
    # 产生纵向偏移值，偏移值的个数为height，以600 × 1000的图像为例，会有39个偏移值，因为height=600/16=39（？？有异议）
    shift_y = np.arange(0, height) * _feat_stride
    #将坐标向量转换为坐标矩阵，新的shift_x行向量为旧shift_x，有dim（shift_y）行，新的shift_y列向量为旧shift_y，有dim（shift_x）列
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # shift_x，shift_y均为39×64的二维数组，对应位置的元素组合即构成图像上需要偏移量大小（偏移量大小是相对与图像最
    # 左上角的那9个anchor的偏移量大小），也就是说总共会得到2496个偏移值对。这些偏移值对与初始的anchor相加即可得到
    # 所有的anchors，所以对于600×1000的图像，总共会产生2496×9个anchors，且存储在all_anchors变量中
    #note: _feat_stride的值不是随便确定的，在经过vgg卷积神经网络后，一共有4个maxpool层，其余conv层pad方式为SAME，可以找到当前featuremap点对应原图像点
    #即featuremap每个点的可视野为（2^4）*（2^4）=16*16,根据featuremap找anchor，即在原图像中以16*16的像素块中找9个比例大小anchor
    #要定位原图像的anchor区域，只需定义以左上角16*16区域所形成的9个anchor相对与所有16*16区域anchor的偏移量，下代码可以实现
    #对于一个width=4,height=3的实例，可以实现：
    #[[ 0 0 0 0]
    # [16 0 16 0]
    # [32 0 32 0]
    # [ 0 16 0 16]
    # [16 16 16 16]
    # [32 16 32 16]
    # [ 0 32 0 32]
    # [16 32 16 32]
    # [32 32 32 32]
    # [ 0 48 0 48]
    # [16 48 16 48]
    # [32 48 32 48]]
    #对应与各个像素块的偏移量
    # numpy.ravel()多维数组降为一维，组合得到一个（width*height，4）的数组
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    #A=_num_anchors等于9
    A = _num_anchors
    #K等于width*height
    K = shifts.shape[0]
    #(1, A, 4)与(K, 1, 4)的数组进行相加，得到(K, A, 4)数组，实验得证，每个(K, 1, 4)的4元素都依次与(1, A, 4)中的每一个4元素相加，最后得到(K, A, 4)数组
    #这样是合理的，因为_anchors中记录的是对用于左上角可视野的9个anchor的左上角坐标与右下角坐标的4个值，而shifts中记录width*height个可视野相对于左上角可视野的偏移量
    #两者相加可得到width*height*9个预测anchor的左上角与右下角坐标信息
    all_anchors = (_anchors.reshape((1, A, 4)) +
                   shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, 4))
    #每个偏移量对应9（A）个不同比例anchor，所以anchor一共有K*A个
    total_anchors = int(K * A)

    # only keep anchors inside the image
    #只保存图像区域内的anchor，超出图片区域的舍弃
    #目前im_info还是placehold，需要以后通过feed_dict来判断是什么信息，目前来看，im_info[0]存的是图片像素行数即高，im_info[1]存的是图片像素列数即宽
    #_allowed_border目前定义为0,其实他规定了一个（-_allowed_border，-_allowed_border）（im_info[1] + _allowed_border，im_info[0] + _allowed_border）
    #这个范围就是图片pad一个_allowed_border的距离
    #[0]表示,np.where取出的是tuple，里面是一个array，array里是符合的引索，所以[0]就是要取出array
    #这一步骤就可以减少掉约2/3的anchor
    inds_inside = np.where(
        (all_anchors[:, 0] >= -_allowed_border) &
        (all_anchors[:, 1] >= -_allowed_border) &
        (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
        (all_anchors[:, 3] < im_info[0] + _allowed_border)    # height
    )[0]

    if DEBUG:
        print 'total_anchors', total_anchors
        print 'inds_inside', len(inds_inside)
    #把上一步得到的符合要求的anchor从all_anchors选出来，存入anchor变量，格式还是一个ndarray
    # keep only inside anchors
    anchors = all_anchors[inds_inside, :]
    if DEBUG:
        print 'anchors.shape', anchors.shape

    # label: 1 is positive, 0 is negative, -1 is dont care
    #生成一个具有符合条件的anchor数个数的未初始化随机数的ndarray
    labels = np.empty((len(inds_inside), ), dtype=np.float32)
    #将这些随机数初始化为-1
    labels.fill(-1)

    # overlaps between the anchors and the gt boxes
    # overlaps (ex, gt)
    #此时假设通过筛选的anchor的个数为N，GT个数为K
    #产生一个(N,K)array，此K与上面说的K不同.里面每一项存的是第N个anchor相对于第K个GT的IOU（重叠面积/（anchor+GT-重叠面积））
    overlaps = bbox_overlaps(
        np.ascontiguousarray(anchors, dtype=np.float),
        np.ascontiguousarray(gt_boxes, dtype=np.float))
    #以横向相比较，取最大值引索，对比结果为每一个anchor找到与其重叠最好的GT
    argmax_overlaps = overlaps.argmax(axis=1)
    #max_overlaps大小（N，），存的是每一个anchor与其重叠最好的GT的IOU（重叠面积/（anchor+GT-重叠面积））
    max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps]
    #以纵向相比较，取最大值引索，对比结果为每一个GT找到与其重叠最好的anchor
    gt_argmax_overlaps = overlaps.argmax(axis=0)
    #gt_max_overlaps大小（K，），存的是每一个GT与其重叠最好的anchor的IOU（重叠面积/（anchor+GT-重叠面积））
    gt_max_overlaps = overlaps[gt_argmax_overlaps,
                               np.arange(overlaps.shape[1])]
    #np.where返回的是一个tuple，tuple里存array，array里为符合的引索，故用[0]取array
    #这句代码没意义，gt_argmax_overlaps已经是通过overlaps.argmax得到的，再用gt_argmax_overlaps得到 gt_max_overlaps，再用gt_max_overlaps得gt_argmax_overlaps，还是原来的gt_argmax_overlaps
    gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0]
    #cfg.TRAIN.RPN_CLOBBER_POSITIVES为False
    if not cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels first so that positive labels can clobber them
        #cfg.TRAIN.RPN_NEGATIVE_OVERLAP=0.3
        #将max_overlaps（与lables大小相同，其实都是对应与anchor）小于0.3的都认为是bg（back ground），设置标签为0
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # fg label: for each gt, anchor with highest overlap
    #与gt有最佳匹配的anchor，labels设置为1（gt_argmax_overlaps虽然与labels形状不同，但是gt_argmax_overlaps存的是anchor的index，就对该index的anchor进行赋值）
    #多个gt可能有同一个最佳匹配的anchor，此时lebals的该anchor引索位置被重复赋值为1
    labels[gt_argmax_overlaps] = 1
    # fg label: above threshold IOU
    #cfg.TRAIN.RPN_POSITIVE_OVERLAP=0.7
    #与gt重叠参数大于等于0.7的anchor，labels设置为1
    labels[max_overlaps >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1
    #这个参数就是看positive与negative谁比较强，先设置0说明positive强，因为0可能转1,而后设置0说明negative强，设置完1还可以设置成0
    if cfg.TRAIN.RPN_CLOBBER_POSITIVES:
        # assign bg labels last so that negative labels can clobber positives
        labels[max_overlaps < cfg.TRAIN.RPN_NEGATIVE_OVERLAP] = 0

    # subsample positive labels if we have too many
    #减少前景样本，如果我们有太多前景样本
    #cfg.TRAIN.RPN_FG_FRACTION=0.5,cfg.TRAIN.RPN_BATCHSIZE=256,因此num_fg=128
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCHSIZE)
    #找到前景样本的引索
    fg_inds = np.where(labels == 1)[0]
    #如果前景样本的引索大于128
    if len(fg_inds) > num_fg:
        #从fg_inds随机挑选出size个元素，存入disable_inds中
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        #对应disable_inds的引索设置为-1,即随机将一部分正样本设置为-1标签样本
        labels[disable_inds] = -1

    # subsample negative labels if we have too many
    #减少背景样本，如果我们有太多背景样本
    #num_bg设置为256-正样本个数。
    num_bg = cfg.TRAIN.RPN_BATCHSIZE - np.sum(labels == 1)
    #找到背景样本引索
    bg_inds = np.where(labels == 0)[0]
    #如果背景样本的引索大于128
    if len(bg_inds) > num_bg:
        # 从fg_inds随机挑选出size个元素，存入disable_inds中
        disable_inds = npr.choice(
            bg_inds, size=(len(bg_inds) - num_bg), replace=False)
        # 对应disable_inds的引索设置为-1,即随机将一部分背景样本设置为-1标签样本
        labels[disable_inds] = -1
        #print "was %s inds, disabling %s, now %s inds" % (
            #len(bg_inds), len(disable_inds), np.sum(labels == 0))
    #创建一个(len(inds_inside)，4)大小的全零数组，存储标签为-1，0,1的anchor。anchor刚才的操作只是对inds_inside里的anchor进行分类（-1,0，1）
    bbox_targets = np.zeros((len(inds_inside), 4), dtype=np.float32)
    #argmax_overlaps为（N，），存的是N个anchor对应匹配最好的GT的引索
    #每个GT的存储为5个元素，前四个与anchor相同，为左上角右下角坐标，最后一个为标签
    #_compute_targets函数返回一个用于anchor回归成target的包含每个anchor回归值(dx、dy、dw、dh)的array,形状（(len(inds_inside), 4），即（anchors.shape[0],4）
    bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])
    #再次初始化一个(len(inds_inside)，4)大小的全零数组
    bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    #对应labels==1的引索,全零的四个元素变为(1.0, 1.0, 1.0, 1.0)
    bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)
    #再再次初始化一个(len(inds_inside)，4)大小的全零数组
    bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
    #cfg.TRAIN.RPN_POSITIVE_WEIGHT=-1,执行if
    if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
        # uniform weighting of examples (given non-uniform sampling)
        #记录需要训练的anchor，即标签为0与1的，-1的舍弃不训练
        num_examples = np.sum(labels >= 0)
        #标签为0的与标签为1的anchor，权重初始化都为（1/num_examples，1/num_examples，1/num_examples，1/num_examples）
        positive_weights = np.ones((1, 4)) * 1.0 / num_examples
        negative_weights = np.ones((1, 4)) * 1.0 / num_examples
    else:
        assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
                (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
        positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
                            np.sum(labels == 1))
        negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
                            np.sum(labels == 0))
    #对应位置放入初始化权重
    bbox_outside_weights[labels == 1, :] = positive_weights
    bbox_outside_weights[labels == 0, :] = negative_weights

    if DEBUG:
        _sums += bbox_targets[labels == 1, :].sum(axis=0)
        _squared_sums += (bbox_targets[labels == 1, :] ** 2).sum(axis=0)
        _counts += np.sum(labels == 1)
        means = _sums / _counts
        stds = np.sqrt(_squared_sums / _counts - means ** 2)
        print 'means:'
        print means
        print 'stdevs:'
        print stds

    # map up to original set of anchors
    #之后可能还会用到第一次被筛选出的anchor信息，所以对labels信息进行扩充，添加进去了第一次筛选出的anchor的标签（都为-1）
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    #以下三个相同，都是把原始anchor信息添加进去，但是信息都是0
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

    if DEBUG:
        print 'rpn: max max_overlap', np.max(max_overlaps)
        print 'rpn: num_positive', np.sum(labels == 1)
        print 'rpn: num_negative', np.sum(labels == 0)
        _fg_sum += np.sum(labels == 1)
        _bg_sum += np.sum(labels == 0)
        _count += 1
        print 'rpn: num_positive avg', _fg_sum / _count
        print 'rpn: num_negative avg', _bg_sum / _count

    # labels
    #pdb.set_trace()
    #height与width分别为feature-map的高和宽，也为原图片横纵可视野个数
    #由之前anchor产生可知，anchor产生的排序方式与卷积的顺序相同，一行一行的出，每个位置产生9个anchor
    #NOTE：由于越往后信息归类越精确，所以labels.reshape((1, height, width, A))顺序正常的
    #之后transpose(0, 3, 1, 2)，此时最精确信息为width，此时以width信息进行fastest聚类
    #测试
    '''
    # aa = np.array(range(1, 37))
    # print aa
    # kk = aa.reshape(1, 3, 3, 4)
    # print kk
    # 
    # cc = kk.transpose(0, 3, 1, 2)
    # print cc
    # kkk = cc.reshape(12, 3)
    # print kkk
    '''
    labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
    labels = labels.reshape((1, 1, A * height, width))
    rpn_labels = labels
    # bbox_targets
    bbox_targets = bbox_targets \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

    rpn_bbox_targets = bbox_targets
    # bbox_inside_weights
    bbox_inside_weights = bbox_inside_weights \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    #assert bbox_inside_weights.shape[2] == height
    #assert bbox_inside_weights.shape[3] == width

    rpn_bbox_inside_weights = bbox_inside_weights

    # bbox_outside_weights
    bbox_outside_weights = bbox_outside_weights \
        .reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
    #assert bbox_outside_weights.shape[2] == height
    #assert bbox_outside_weights.shape[3] == width

    rpn_bbox_outside_weights = bbox_outside_weights

    return rpn_labels,rpn_bbox_targets,rpn_bbox_inside_weights,rpn_bbox_outside_weights


def _unmap(data, count, inds, fill=0):
    """ Unmap a subset of item (data) back to the original set of items (of
    size count) """
    #判断label是否为一维的，执行if
    if len(data.shape) == 1:
        #建立一个（A*K，）大小的一维数组，fill：-1
        ret = np.empty((count, ), dtype=np.float32)
        ret.fill(fill)
        #图片内的anchor属于第一次筛选，筛选出去的label都为-1
        #第一次筛选后的anchor，其中符合条件的anchor分别被赋予0与1，其余的都为-1
        #第二次筛选：可能标签为1与0的太多了，随机排除一些，标签设置为-1
        #所以inds_inside与labels一一对应，但是其中还存在有大量不训练的标签为-1的anchor
        ret[inds] = data
    else:
        #产生一个（A*K，4）ndarray，fill=0
        ret = np.empty((count, ) + data.shape[1:], dtype=np.float32)
        ret.fill(fill)
        #对于标签为0与1的填入信息
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    #要求anchor与对应匹配最好GT个数相同
    assert ex_rois.shape[0] == gt_rois.shape[0]
    #要有anchor左上角与右下角坐标，有4个元素
    assert ex_rois.shape[1] == 4
    #GT有标签位，所以为5个
    assert gt_rois.shape[1] == 5
    #返回一个用于anchor回归成target的包含每个anchor回归值(dx、dy、dw、dh)的array
    return bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)