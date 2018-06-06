import numpy as np
from .generate_anchors import generate_anchors
from ..fast_rcnn.config import cfg_from_file
from ..fast_rcnn.bbox_transform import bbox_transform_inv, clip_boxes
from ..fast_rcnn.nms_wrapper import nms


DEBUG = False


def proposal_layer(rnc_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, _feat_stride = [16,], anchor_scales = [16,]):
    """
    Proposal Layer负责综合所有[dx(A)，dy(A)，dw(A)，dh(A)]变换量和foreground anchors，计算出精准的proposal，送入后续RoI Pooling Layer
    Proposal Layer有3个输入：
    fg/bg anchors分类器结果rnc_cls_prob_reshape:(1 , H , W , Ax2)
    对应的bbox reg---[dx(A)，dy(A)，dw(A)，dh(A)]:(1 , H , W , Ax4)变换量rpn_bbox_pred，
    im_info:[M, N, scale_factor]
    
    另外还有参数feat_stride=16
    Parameters

    输出：
    roi层的输入之一：
    rpn_rois : shape为（proposals.shape[0]，4） e.g. [0, x1, y1, x2, y2]
    ----------
    rnc_cls_prob_reshape: (1 , H , W , Ax2) outputs of RPN, prob of bg or fg
                         NOTICE: the old version is ordered by (1, H, W, 2, A) !!!!
    rpn_bbox_pred: (1 , H , W , Ax4), rgs boxes output of RPN
    im_info: a list of [image_height, image_width, scale_ratios]
    cfg_key: 'TRAIN' or 'TEST'
    _feat_stride: the downsampling ratio of feature map to the original input image
    anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
    ----------
    Returns
    ----------
    rpn_rois : (1 x H x W x A, 5) e.g. [0, x1, y1, x2, y2]

    # Algorithm:
    #
    # for each (H, W) location i
    #   generate A anchor boxes centered on cell i
    #   apply predicted bbox deltas at cell i to each of the A anchors
    # clip predicted boxes to image
    # remove predicted boxes with either height or width < threshold
    # sort all (proposal, score) pairs by score from highest to lowest
    # take top pre_nms_topN proposals before NMS
    # apply NMS with threshold 0.7 to remaining proposals
    # take after_nms_topN proposals after NMS
    # return the top proposals (-> RoIs top, scores top)
    #layer_params = yaml.load(self.param_str_)
    """
    cfg_key = cfg_key.decode('ascii')
    _anchors = generate_anchors(scales = np.array(anchor_scales))# 生成基本的9个anchor
    _num_anchors = _anchors.shape[0] # 9
    rnc_cls_prob_reshape = np.transpose(rnc_cls_prob_reshape, [0, 3, 1, 2])
    rpn_bbox_pred = np.transpose(rpn_bbox_pred, [0, 3, 1, 2])
    im_info = im_info[0] #原始图像的高宽、缩放尺度

    assert rnc_cls_prob_reshape.shape[0] == 1,  \
        'Only single item batches are supported'

    pre_nms_topN = cfg[cfg_key].RPN_PRE_NMS_TOP_N
    # 12000,在做nms之前，最多保留的候选box数目
    post_nms_topN = cfg[cfg_key].RPN_POST_NMS_TOP_N
    # 2000，做完nms之后，最多保留的box的数目
    nms_thresh = cfg[cfg_key].RPN_NMS_THRESH
    # nms用参数，阈值是0.7
    min_size = cfg[cfg_key].RPN_MIN_SIZE

    #从深度方向切片，前一半是每个中心i对应可视野的9个anchor的为bg的分类得分，后一半是每个中心i对应可视野的9个anchor为fg的分类得分
    #目前取的是fg部分
    scores = rnc_cls_prob_reshape[:, _num_anchors:, :, :]

    bbox_deltas = rpn_bbox_pred

    if DEBUG:
        print 'im_size: ({}, {})'.format(im_info[0], im_info[1])
        print 'scale: {}'.format(im_info[2])

    # 1. Generate proposals from bbox deltas and shifted anchors
    # #取出feature-map的高和宽
    height, width = scores.shape[-2:]

    if DEBUG:
        print 'score map size: {}'.format(scores.shape)

    # 同anchor-target-layer-tf这个文件一样，生成anchor的shift，进一步得到整张图像上的所有anchor
    # 产生横向偏移值，偏移值的个数为width，以600 × 1000的图像为例，会有64个偏移值，因为width=1000/16=64
    shift_x = np.arange(0, width) * _feat_stride
    # 产生纵向偏移值，偏移值的个数为height，以600 × 1000的图像为例，会有39个偏移值，因为height=600/16=39
    shift_y = np.arange(0, height) * _feat_stride
    # 将坐标向量转换为坐标矩阵，新的shift_x行向量为旧shift_x，有dim（shift_y）行，新的shift_y列向量为旧shift_y，有dim（shift_x）列
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    # shift_x，shift_y均为39×64的二维数组，对应位置的元素组合即构成图像上需要偏移量大小（偏移量大小是相对与图像最
    # 左上角的那9个anchor的偏移量大小），也就是说总共会得到2496个偏移值对。这些偏移值对与初始的anchor相加即可得到
    # 所有的anchors，所以对于600×1000的图像，总共会产生2496×9个anchors，且存储在all_anchors变量中
    # note: _feat_stride的值不是随便确定的，在经过vgg卷积神经网络后，一共有4个maxpool层，其余conv层pad方式为SAME，可以找到当前featuremap点对应原图像点
    # 即featuremap每个点的可视野为（2^4）*（2^4）=16*16,根据featuremap找anchor，即在原图像中以16*16的像素块中找9个比例大小anchor
    # 要定位原图像的anchor区域，只需定义以左上角16*16区域所形成的9个anchor相对与所有16*16区域anchor的偏移量，下代码可以实现
    # 对于一个width=4,height=3的实例，可以实现：
    # [[ 0 0 0 0]
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
    # 对应与各个像素块的偏移量
    # numpy.ravel()多维数组降为一维，组合得到一个（width*height，4）的数组
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()

    # 产生feature maps上所有像素点的anchors
    # Enumerate all shifted anchors:
    #
    # add A anchors (1, A, 4) to
    # cell K shifts (K, 1, 4) to get
    # shift anchors (K, A, 4)
    # reshape to (K*A, 4) shifted anchors
    A = _num_anchors
    # K等于width*height
    K = shifts.shape[0]
    # (1, A, 4)与(K, 1, 4)的数组进行相加，得到(K, A, 4)数组，实验得证，每个(K, 1, 4)的4元素都依次与(1, A, 4)中的每一个4元素相加，最后得到(K, A, 4)数组
    # 这样是合理的，因为_anchors中记录的是对用于左上角可视野的9个anchor的左上角坐标与右下角坐标的4个值，而shifts中记录width*height个可视野相对于左上角可视野的偏移量
    # 两者相加可得到width*height*9个预测anchor的左上角与右下角坐标信息
    anchors = _anchors.reshape((1, A, 4) + shifts.reshape(1, K, 4)).transpose((1, 0, 2))
    anchors = anchors.reshape((K * A, 4))
    # 这里得到的anchor就是整张图像上的所有anchor

    # 变换predict_box的shape
    # (1, 4 * A, H, W)变成(1, H, W, 4 * A)，再变成(1 * H * W * A, 4)
    # 将bbox信息(1, H, W, 4 * A)转化为 (1 * H * W * A, 4)形式，使得与anchor信息order相同
    bbox_deltas = bbox_deltas.transpose((0, 2, 3, 1)).reshape((-1, 4))

    # 变换cls_box的shape
    # (1, 4 * A, H, W)变成(1, H, W, 4 * A)，再变成(1 * H * W * A, 4)
    # # 将score信息(1, A, H, W) 转化为 (1 * H * W * A, 1)形式，使得与anchor信息order相同
    scores = scores.transpose((0, 2, 3, 1)).tf.reshape((-1, 1))

    #通过bbox转换将anchor转换成proposals
    #这里的过程就是每个anchor都是通过各自的（dx,dy,dw,dh）来到G‘即proposal，使得G’～GT（ground-true），其中（dx,dy,dw,dh）来自'rpn_bbox_pred'层
    #shape:[num_proposals, 4]
    proposals = bbox_transform_inv(anchors, bbox_deltas)#做逆变换，得到box在图像上的真实坐标

    # 2. clip predicted boxes to image
    #裁剪预测框
    #im_info[0]存的是图片像素行数即高，im_info[1]存的是图片像素列数即宽
    #使得boxes位于图片内
    proposals = clip_boxes(proposals, im_info[:2])

    # 3. remove predicted boxes with either height or width < threshold
    #移除小于阈值的boxes
    # (NOTE: convert min_size to input image scale stored in im_info[2])
    #推断的话 im_info[2]=1/16
    keep = _filter_boxes(proposals, min_size * im_info[2])
    #保存符合条件的proposal与对应scores
    proposals = proposals[keep, :]
    scores = scores[keep]

    # 4. sort all (proposal, score) pairs by score from highest to lowest
    # 5. take top pre_nms_topN (e.g. 6000)
    #numpy.argsort()返回的是数值从小到大的引索值，[::-1]是反序排列。所以order是score从大到小的引索值
    order = scores.ravel().argsort()[::-1]
    if pre_nms_topN > 0:
        # 保存符合条件的proposal与对应scores
        order = order[:pre_nms_topN]
    # 保存符合条件的proposal与对应scores
    proposals = proposals[order, :]
    scores = scores[order]

    # 6. apply nms (e.g. threshold = 0.7)
    # 7. take after_nms_topN (e.g. 300)
    # 8. return the top proposals (-> RoIs top)
    #返回的是nms提纯后的引索，已经是按照score从大到小排序了
    keep = nms(np.hstack((proposals, scores)), nms_thresh)
    if post_nms_topN > 0:
        #post_nms_topN（TRAIN：2000 TEST：300）
        #取前两千个score高的引索
        keep = keep[:post_nms_topN]
    #进一步提纯
    proposals = proposals[keep, :]
    scores = scores[keep]
    # Output rois blob
    # Our RPN implementation only supports a single input image, so all
    # batch inds are 0
    #建立一个proposal引索，proposals.shape[0]为还剩的proposal的个数
    batch_inds = np.zeros((proposals.shape[0], 1), dtype = np.float32)
    #生成blob，由[proposal引索(全0)，proposal]构成，shape为（proposals.shape[0]，4）
    blob = np.hstack((batch_inds, proposals.astype(np.float32, copy = False)))
    return blob
    #top[0].reshape(*(blob.shape))
    #top[0].data[...] = blob

    # [Optional] output scores blob
    #if len(top) > 1:
    #    top[1].reshape(*(scores.shape))
    #    top[1].data[...] = scores

def _filter_boxes(boxes, min_size):
    #这个就是找到符合条件的boxes，引索存入keep
    """Remove all boxes with any side smaller than min_size."""
    ws = boxes[:, 2] - boxes[:, 0] + 1
    hs = boxes[:, 3] - boxes[:, 1] + 1
    keep = np.where((ws >= min_size) & (hs >= min_size))[0]
    return keep