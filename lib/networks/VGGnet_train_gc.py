import tensorflow as tf
from networks.network import Network


#define

n_classes = 21
_feat_stride = [16,]
anchor_scales = [8, 16, 32]

class VGGnet_train(Network):
    """docstring for VGGnet_train
    VGGnet_train实例化的时候，会传入一个Network参数，这个Network参数相当于是VGGnet_train的子类
    """
    def __init__(self, trainable = True):
        self.trainable = trainable
        self.data = tf.placeholder(tf.float32, shape = [None, None, None, 3])
        self.im_info = tf.placeholder(tf.float32, shape = [None, 3])
        self.gt_boxes = tf.placeholder(tf.float32, shape = [None, 5])
        # 该参数定义dropout比例
        self.keep_prob = tf.placeholder(tf.float32)
        self.layers = dict({'data': self.data, 'im_info': self.im_info, 'gt_boxes': self.gt_boxes})
        self.trainable = trainable
        # 子类的方法重载了父类的方法
        # 调用子类的方法
        self.setup()

        # create ops and placeholders for bbox normalization process
        #建立weights,biases变量，用tf.assign来更新
        with tf.variable_scope('bbox_pred', reuse = True):
            weights = tf.get_variable("weights")
            biases = tf.get_variable("biases")

            self.bbox_weights = tf.placeholder(weights.dtype, shape = weights.get_shape())
            self.bbox_biases = tf.placeholder(biases.dtype, shape = biases.get_shape())
            # tf.assign用来更新参数值
            self.bbox_weights_assign = weights.assgin(self.bbox_weights)
            self.bbox_bias_assign = biases.assgin(self.bbox_biases)

    def setup(self):
        #feed就是从self.layers中的信息提取出来（包括self.data,self.im_info,self.gt_boxes）存入self.input
        #feed返回的是self，为conv第一个参数,conv参数（self,卷积核高，宽，深度，stride高，宽）
        #return self 用法：
        #class human（object）
        #   def __init__(self,weight):
        #       self.weight=weight
        #   def get_weight(self):
        #       return self.weight
        #想要调用get_weight,直接用human.get_weight(45)会告知调用之前要先实例化，weight为未绑定函数
        #而   human.get_weight(human(45))就可以正常输出，说明human(45)将get_weight绑定了
        #其实human（45）作为一个self传给get_weight（）
        #conv:卷积高、宽、输出深度、步长高、宽
        #VGG层
        (self.feed('data')
             .conv(3, 3, 64, 1, 1, name = 'conv1_1', trainable = False)
             .conv(3, 3, 64, 1, 1, name = 'conv1_2',trainable = False)
             .max_pool(2, 2, 2, 2, padding = 'VALID', name = 'pool1')
             .conv(3, ,3 ,128, 1, 1, name = 'conv2_1', trainable = False)
             .conv(3, ,3 ,128, 1, 1, name = 'conv2_2', trainable = False)
             .max_pool(2, 2, 2, 2, padding = 'VALID', name = 'pool2')
             .conv(3, ,3 ,256, 1, 1, name = 'conv3_1', trainable = False)
             .conv(3, ,3 ,256, 1, 1, name = 'conv3_2', trainable = False)
             .conv(3, ,3 ,256, 1, 1, name = 'conv3_3', trainable = False)
             .max_pool(2, 2, 2, 2, padding = 'VALID', name = 'pool3')
             .conv(3, ,3 ,512, 1, 1, name = 'conv4_1', trainable = False)
             .conv(3, ,3 ,512, 1, 1, name = 'conv4_2', trainable = False)
             .conv(3, ,3 ,512, 1, 1, name = 'conv4_3', trainable = False)
             .max_pool(2, 2, 2, 2, padding = 'VALID', name = 'pool4')
             .conv(3, 3, 512, 1, 1, name='conv5_1')
             .conv(3, 3, 512, 1, 1, name='conv5_2')
             .conv(3, 3, 512, 1, 1, name='conv5_3')
            )
        #========= RPN ============
        #feed操作就是保证self.inputs中只存有下一步操作所需要的上一层的输出数据，每一个op操作都会包含一个feed操作（定义在layer函数里）
        #在初始化feed时候可能需要好几种数据，如feed('rpn_cls_score','gt_boxes','im_info','data')
        #rpn_conv/3x3和rpn_cls_score层
        (self.feed('conv5_3')
             .conv(3, 3, 512, 1, 1, name = 'rpn_conv/3x3')
             .conv(1, 1, len(anchor_scales) * 3 * 2, 1, 1, padding = 'VALID', relu = False, name = 'rpn_cls_score')
            )
        #feed的返回值为参数层的输出，都是从self.layers这个字典中取的，layer函数中有将各层的输出存进self.layers的操作
        #'rpn-data'层给出了rpn中的所有信息，包括图片anchor标签（-1,0,1），回归值（dx，dy，dw，dh），两个权重
        #data：为图片像素信息，info：im_info[0]存的是图片像素行数即高，im_info[1]存的是图片像素列数即宽
        #gt_boxes：GT信息，为5维度，前四个为（xmin，ymin）（xmax，ymax），最后一个为标签
        #rpn_cls_score就把他看成一个简单的feature-map，深度为3*3*2,对应3比例*3尺度*2分类值
        #anchor_target_layer层:
        #对应rpn-data,训练阶段用来产生计算rpn_cls_loss和rpn_loss_bbox的参数：
        (self.feed('rpn_cls_score', 'gt_boxes', 'im_info', 'data')
             .anchor_target_layer(_feat_stride, anchor_scales, name = 'rpn-data')
            )

        # Loss of rpn_cls & rpn_boxes
        #回归bbox,存的是（dx,dy,dw,dh）
        #rpn_cls_loss和rpn_loss_bbox
        
        # rpn_loss_bbox
        (self.feed('rpn_conv/3x3')
             .conv(1, 1, len(anchor_scales) * 3 * 4, 1, 1, padding = 'VALID', relu = False, name = 'rpn_bbox_pred')
            )

        # rpn_cls_loss
        #========= RoI Proposal ============
        #先reshape后softmax激活
        (self.feed('rpn_cls_score')
             .reshape_layer(2, name = 'rpn_cls_score_reshape')# 形状shape(1, 9n, n, 2)
             .softmax(name = 'rpn_cls_prob')
             #形状shape(1,9n,n,2)
            )
        # 再次reshape成blob的格式
        (self.feed('rpn_cls_prob')# 形状shape（1, 9n, n, 2)
             .reshape_layer(len(anchor_scales) * 3 * 2, name = 'rpn_cls_prob_reshape')
            )#形状shape(1,n,n,18),信息还原成'rpn_cls_score'，刚才两步reshape_layer操作：1、修改为softmax格式。2、还原rpn_cls_score信息位置格式，只不过内容变为sotfmax得分

        # proposal层
        (self.feed('rpn_cls_prob_reshape', 'rpn_bbox_pred', 'im_info')
            # 初始得到blob，内容为[proposal引索(全0)，proposal]，shape（proposals.shape[0](即暂存的proposals个数),5）
             .proposal_layer(_feat_stride, anchor_scales, 'TRAIN', name= 'rpn_rois')
            )

        #产生筛选后的roi，对应labels，三个（len(rois),4*21）大小的矩阵，其中一个对fg-roi对应引索行的对应类别的4个位置填上（dx,dy,dw,dh），另两个对fg-roi对应引索行的对应类别的4个位置填上（1,1,1,1）
        (self.feed('rpn_rois', 'gt_boxes'))
             .proposal_target_layer(n_classes, name = 'roi-data')


        #========= RCNN ============
        # 分类结果
        (self.feed('conv5_3', 'roi-data')
             .roi_pool(7, 7, 1.0 / 16, name = 'pool_5')
             .fc(4096, name = 'fc6')
             .dropout(0.5, name = 'drop6')
             .fc(4096, name = 'fc7')
             .dropout(0.5, name = 'drop7')
             .fc(n_classes, relu = False, name = 'cls_score')
             .softmax(name = 'cls_prob')
            )

        # 回归结果（bbox）
        (self.feed('fc7')
             .fc(n_classes * 4, relu = False, name = 'bbox_pred')
            )

