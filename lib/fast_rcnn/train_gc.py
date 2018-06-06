# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network."""

from fast_rcnn.config import cfg
import gt_data_layer.roidb as gdl_roidb
import roi_data_layer.roidb as rdl_roidb
from roi_data_layer.layer import RoIDataLayer
from utils.timer import Timer
import numpy as np
import os
import tensorflow as tf
import sys
from tensorflow.python.client import timeline
import time

class SolverWrapper(object):
    """A simple wrapper around Caffe's solver.
    This wrapper gives us control over he snapshotting process, which we
    use to unnormalize the learned bounding-box regression weights.
    """
    def __init__(self, sess, saver, network, imdb, roidb, output_dir, pretrained_model=None):
        self.net = network
        self.imdb = imdb
        self.roidb = roidb
        self.output_dir = output_dir
        self.pretrained_model = pretrained_model

        print 'Computing bounding-box regression targets...'

        # True
        if cfg.TRAIN.BBOX_REG:
            #不同类的均值与方差，返回格式means.ravel(), stds.ravel()
            self.bbox_means, self.bbox_stds = rdl_roidb.add_bbox_regression_targets(roidb)
        print('done')

        # for checkpoint
        self.saver = saver

    def snapshot(self, sess, iter):
        """Take a snapshot of the network after unnormalizing the learned
        bounding-box regression weights. This enables easy use at test-time.
        """
        net = self.net

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            # save original values
            with tf.variable_scope('bbox_pred', reuse=True):
                weights = tf.get_variable("weights")
                biases = tf.get_variable("biases")

            orig_0 = weights.eval()
            orig_1 = biases.eval()

            # scale and shift with bbox reg unnormalization; then save snapshot
            weights_shape = weights.get_shape().as_list()
            sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0 * np.tile(self.bbox_stds, (weights_shape[0], 1))})
            sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1 * self.bbox_stds + self.bbox_means})

        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        infix = ('_' + cfg.TRAIN.SNAPSHOT_INFIX
                 if cfg.TRAIN.SNAPSHOT_INFIX != '' else '')
        filename = (cfg.TRAIN.SNAPSHOT_PREFIX + infix +
                    '_iter_{:d}'.format(iter+1) + '.ckpt')
        filename = os.path.join(self.output_dir, filename)

        self.saver.save(sess, filename)
        print 'Wrote snapshot to: {:s}'.format(filename)

        if cfg.TRAIN.BBOX_REG and net.layers.has_key('bbox_pred'):
            with tf.variable_scope('bbox_pred', reuse=True):
                # restore net to original state
                sess.run(net.bbox_weights_assign, feed_dict={net.bbox_weights: orig_0})
                sess.run(net.bbox_bias_assign, feed_dict={net.bbox_biases: orig_1})

    #sigma为3，计算smooth_l1损失
    def _modified_smooth_l1(self, sigma, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights):
        """
            ResultLoss = outside_weights * SmoothL1(inside_weights * (bbox_pred - bbox_targets))
            SmoothL1(x) = 0.5 * (sigma * x)^2,    if |x| < 1 / sigma^2
                          |x| - 0.5 / sigma^2,    otherwise
        """
        #9
        sigma2 = sigma * sigma
        #tf.subtract(bbox_pred, bbox_targets)，从bbox_pred减去bbox_targets，再与bbox_inside_weights相乘
        inside_mul = tf.multiply(bbox_inside_weights, tf.subtract(bbox_pred, bbox_targets))
        #判断abs（inside_mul）是否小于1/9,如果小于对应位置返回True，否则为False，再tf.cast转换为0和1
        smooth_l1_sign = tf.cast(tf.less(tf.abs(inside_mul), 1.0 / sigma2), tf.float32)
        smooth_l1_option1 = tf.multiply(tf.multiply(inside_mul, inside_mul), 0.5 * sigma2)
        smooth_l1_option2 = tf.subtract(tf.abs(inside_mul), 0.5 / sigma2)
        #结果就是实现上面的SmoothL1(x)结果
        smooth_l1_result = tf.add(tf.multiply(smooth_l1_option1, smooth_l1_sign),
                                  tf.multiply(smooth_l1_option2, tf.abs(tf.subtract(smooth_l1_sign, 1.0))))
        #实现：bbox_outside_weights*SmoothL1(x)
        outside_mul = tf.multiply(bbox_outside_weights, smooth_l1_result)

        return outside_mul

    def train_model(self, sess, max_iters):
        """Network training loop."""
        #返回一个RoIDataLayer类对象，内容self._roidb ,self._num_classes ,self._perm,self._cur
        data_layer = get_data_layer(self.roidb, self.imdb.num_classes)

        # RPN
        # classification loss
        #将'rpn_cls_score_reshape'层的输出（1,n，n，18）reshape为（-1,2）,其中2为前景与背景的多分类得分（）
        #self.net.get_output得到self.layers
        #self.layers在VGGnet_train.py中重载了，为一个dict，记录的是每一层的输出
        rpn_cls_score = tf.reshape(self.net.get_output('rpn_cls_score_reshape'), [-1, 2])
        #'rpn-data'层输出的[0]为rpn_label,shape为(1, 1, A * height, width)，中存的是所有anchor的label（-1,0,1）
        # 问题1：目前感觉有异议，数据读取方向labels有问题################################
        rpn_label = tf.reshape(self.net.get_output('rpn-data')[0], [-1])
         # 剔除不参加训练的anchors
         #把rpn_label不等于-1对应引索的rpn_cls_score取出，重新组合成rpn_cls_score
         rpn_cls_score = tf.reshape(tf.gather(rpn_cls_score, tf.where(tf.not_equal(rpn_label, -1))), [-1, 2])
         #把rpn_label不等于-1对应引索的rpn_label取出，重新组合成rpn_label
         rpn_label = tf.reshape(tf.gather(rpn_label, tf.where(tf.not_equal(rpn_label, -1))), [-1])

         # faster RCNN损失之一：rpn网络的cls损失
         #score损失：tf.nn.sparse_softmax_cross_entropy_with_logits函数的两个参数logits，labels数目相同（shape[0]相同），分别为最后一层的输出与标签
         #NOTE：这个函数返回的是一个向量，要求交叉熵就tf.reduce_sum，要求损失就tf.reduce_mean
         #问题2：logits，labels应该shape相同的，但这里不同，有异议
         rpn_cross_entropy = tf.reduce_mean(tf.losses.sparse_softmax_cross_entropy_with_logits(logits = rpn_cls_score, labels = rpn_label))

         # faster RCNN损失之二：rpn网络的pred损失
         # bounding box regression L1 loss
         #'rpn_bbox_pred'层为了回归bbox,存的是（dx,dy,dw,dh）
         rpn_bbox_pred = self.net.get_output('rpn_bbox_pred')
         #'rpn-data'[1]返回一个用于anchor回归成target的包含每个anchor回归值(dx、dy、dw、dh)的array,形状（(len(inds_inside), 4），即（anchors.shape[0],4）
         #重新reshape成(1, height, width, A * 4)
         rpn_bbox_targets = tf.transpose(self.net.get_output('rpn-data')[1], [0, 2, 3, 1])
         #rpn_bbox_inside_weights：标签为1的anchor,对应(1.0, 1.0, 1.0, 1.0)
         #重新reshape成(1, height, width, A * 4)
         rpn_bbox_inside_weights = tf.transpose(self.net.get_output('rpn-data')[2], [0, 2, 3, 1])
         #rpn_bbox_outside_weights:标签为0或者1的，权重初始化都为（1/num_examples，1/num_examples，1/num_examples，1/num_examples），num_examples为标签为0或者1的anchor总数
         #重新reshape成(1, height, width, A * 4)
         rpn_bbox_outside_weights = tf.transpose(self.net.get_output('rpn-data')[3], [0, 2, 3, 1])
         # 计算smooth_l1损失
         rpn_smooth_l1 = self._modified_smooth_l1(3.0 , rpn_bbox_pred, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)
         # rpn_smooth_l1计算出的为一个向量，现在要合成loss形式
         rpn_loss_box = tf.reduce_mean(tf.reduce_sum(rpn_smooth_l1, reduction_indices = [1, 2, 3]))

         # R-CNN
         # classification loss
         #得到最后一个score分支fc层的输出
         # faster RCNN损失之三：final分类cls损失
         cls_score = self.net.get_output('cls_score')
         #label：筛选出的proposal与GT结合形成all_roi,从all_roi中筛选出符合的roi，得到这些roi的label
         label = tf.reshape(self.net.get_output('roi-data')[1], [-1])
         #用这些roi的label与最后一个score分支fc层的输出相比较，得到loss
         cross_entropy = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = cls_score, labels = label))

         # faster RCNN损失之四：final回归reg损失
         # bounding box regression L1 loss
         #得到最后一个bbox分支fc层的输出
         bbox_pred = self.nn.get_output('bbox_pred')
         bbox_targets = self.nn.get_output('roi-data')[2]
         bbox_inside_weights = self.nn.get_output('roi-data')[3]
         bbox_outside_weights = self.nn.get_output('roi-data')[4]
         # 计算smooth_l1损失
         smooth_l1 = self._modified_smooth_l1(1.0, bbox_pred, bbox_targets, bbox_inside_weights, bbox_outside_weights)
         # smooth_l1计算出的为一个向量，现在要合成loss形式
         loss_box = tf.reduce_mean(tf.reduce_sum(smooth_l1, reduction_indices = [1]))

         # final loss
         # 总loss
         loss = cross_entropy + loss_box + rpn_cross_entropy + rpn_loss_box

         # optimizer and learning rate
         global_step = tf.Variable(0, trainable = False)

         #cfg.TRAIN.LEARNING_RATE为0.001,  cfg.TRAIN.STEPSIZE为50000
         #tf.train.exponential_decay（初始lr，初始步数，多少步进入下一平台值，总步数，下一次平台值是多少（基于上次的比率），staircase）
         #staircase为True则遵循刚才规则，如为False则每一次迭代更新一次
         lr = tf.train.exponential_decay(cfg.TRAIN.LEARNING_RATE, global_step, cfg.TRAIN.STEPSIZE, 0.1, staircase = True)
         # cfg.Train.MOMENTUN为0.9
         momentum = cfg.TRAIN.MOMENTUN
         # 动态系数为0.9的梯度下降法
         train_op = tf.train.MomentumOptimizer(lr, momentum).minimize(loss, global_step = global_step)

         # initialize variables
         sess.run(tf.global_variables_initializer())
         # 如果有预训练模型，则加载
         if self.pretrained_model is not None:
            print ('Loading pretrained model '
                   'weights from {:s}').format(self.pretrained_model)
            self.net.load(self.pretrained_model, sess, self.saver, True)

        last_snaphot_iter = -1
        # 记录当前时间
        timer = Timer()
        # 在最大循环次数内
        for iter in range(max_iters):
            # 得到一个batch信息
            # 依次取batch大小个roi的index
            blobs = data_layer.forward()

            # 给定placehold信息
            feed_dict = {self.net.data: blobs['data'], self.net.im_info: blobs['im_info'], self.net.keep_prob 0.5, self.net.gt_boxes: blobs['gt_boxes']}

            run_options = None
            run_metadata = None
            #False
            if cfg.TRAIN.DEBUG_TIMELINE:
                run_options = tf.RunOptions(trace_level = tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

            time.tic()

            rpn_loss_cls_value, rpn_loss_box_value, loss_cls_value, loss_box_value, _ = sess.run([rpn_cross_entropy, rpn_loss_box, cross_entropy, loss_box, train_op], feed_dict = feed_dict, options = run_options, run_metadata = run_metadata)

            timer.toc()
            #False
            if cfg.TRAIN.DEBUG_TIMELINE:
                trace = timeline.Timeline(step_stats=run_metadata.step_stats)
                trace_file = open(str(long(time.time() * 1000)) + '-train-timeline.ctf.json', 'w')
                trace_file.write(trace.generate_chrome_trace_format(show_memory=False))
                trace_file.close()

            if (iter+1) % (cfg.TRAIN.DISPLAY) == 0:
                print 'iter: %d / %d, total loss: %.4f, rpn_loss_cls: %.4f, rpn_loss_box: %.4f, loss_cls: %.4f, loss_box: %.4f, lr: %f'%\
                        (iter+1, max_iters, rpn_loss_cls_value + rpn_loss_box_value + loss_cls_value + loss_box_value ,rpn_loss_cls_value, rpn_loss_box_value,loss_cls_value, loss_box_value, lr.eval())
                print 'speed: {:.3f}s / iter'.format(timer.average_time)

            if (iter+1) % cfg.TRAIN.SNAPSHOT_ITERS == 0:
                last_snapshot_iter = iter
                self.snapshot(sess, iter)

        if last_snaphot_iter != iter:
            self.snapshot(sess, iter)

#返回一个RoIDataLayer类对象，内容self._roidb ,self._num_classes ,self._perm,self._cur
def get_data_layer(roidb, num_classes):
    """return a data layer."""
    #False
    if cfg.TRAIN.HAS_RPN:
        #False
        if cfg.IS_MULTISCALE:
            layer = GtDataLayer(roidb)
        else:
            layer = RoIDataLayer(roidb, num_classes)
    else:
        layer = RoIDataLayer(roidb, num_classes)

    return layer

def get_training_roidb(imdb):
    """Returns a roidb (Region of Interest database) for use in training."""
    #cfg.TRAIN.USE_FLIPPED已经定义为TRUE，表示使用水平反转图像（数据增强），防止过拟合
    if cfg.TRAIN.USE_FLIPPED:
        print 'Appending horizontally-flipped training examples...'
        # append_flipped_images：将图片水平对称变换，增加样本数量
        # roidb是imdb的一个成员变量，roidb是一个list（每个元素对应一张图片），list中的元素是一个字典，字典中存放了5个key，分别是boxes信息，每个box的class信息，是否是flipped的标志位，重叠信息gt_overlaps，以及seg_areas；分析该函数可知，将box的值按照水平对称，原先roidb中只有5011个元素，经过水平对称后通过append增加到5011*2=10022个
        imdb.append_flipped_images()
        print('done')

    print('Preparing training data...')
    #cfg.TRAIN.HAS_RPN为false
    if cfg.TRAIN.HAS_RPN:
        if cfg.IS_MULTISCALE:
            gdl_roidb.prepare_roidb(imdb)
        else:
            rdl_roidb.prepare_roidb(imdb)
    else:
        #就是对roidb进行进一步的操作，添加了image.weight.height.max_classes.max_overlaps
        rdl_roidb.prepare_roidb(imdb)
    print 'done'

    return imdb.roidb

def filter_roidb(roidb):
    """Remove roidb entries that have no usable RoIs."""
    def is_valid(entry):
        """
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        """
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        #overlaps就是一个one-hot编码，有分类物体的就在该分类位置上置1（包括背景），所以可以通过一下函数找到有只是一个前景背景物体的图片
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
         # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        #如果至少有一个前景或者背景即返回True
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        return valid

    # roidb列表中元素（字典）的长度，即有多少个图片信息
    num = len(roidb)
    # 记录筛选过后的roidb
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    # 筛选过后的roid数目
    num_after = len(filered_roidb)
    print ('Filtered {} roidb entries: {} -> {}'.format(num - num_after,
                                                           num, num_after))

    return filtered_roidb

#network为VGGnet_train一个对象，imdb为pascal_voc对象，roidb为一个列表
def train_net(network, imdb, roidb, output_dir, pretrained_model = None, max_iters = 40000):
    """Train a Fast R-CNN network."""
    #筛选roidb（至少有一个前景或者背景的图片）
    roidb = filter_roidb(roidb)
    # 对参数进行保存，100次迭代更新一次
    saver = tf.train.Saver(max_to_keep = 100)
    #建立对话，对于tf.ConfigProto有以下选项
    #log_device_placement=True : 是否打印设备分配日志
    #allow_soft_placement=True ： 如果你指定的设备不存在，允许TF自动分配设备
    with tf.Session(config = tf.ConfigProto(allow_soft_placement = True)) as sess:
        #建立SolverWrapper一个对象,添加了self.saver, self.net, self.imdb, self.roidb,self.output_dir, self.pretrained_model，
        #以及roidb['bbox_targets'](标准化后的), self.bbox_means, self.bbox_stds信息
        sw = SolverWrapper(sess, saver, network, imdb, roidb, output_dir, pretrained_model = pretrained_model)
        print 'Solving...'
        sw.train_model(sess, max_iters)
        print 'done solving'