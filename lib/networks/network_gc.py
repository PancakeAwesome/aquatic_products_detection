import numpy as np
import tensorflow as tf
import roi_pooling_layer.roi_pooling_op as roi_pool_op
import roi_pooling_layer.roi_pooling_op_grad
from rpn_msr.proposal_layer_tf import proposal_layer as proposal_layer_py
from rpn_msr.anchor_target_layer_tf import anchor_target_layer as anchor_target_layer_py
from rpn_msr.proposal_target_layer_tf import proposal_target_layer as proposal_target_layer_py



DEFAULT_PADDING = 'SAME'

def layer(op):
    def layer_decorated(self, *args, **kwargs):
        # Automatically set a name if not provided.
        #op.__name__的是各个操作函数名，如conv、max_pool
        #get_unique_name返回类似与conv_4，以name：'conv_4'存在kwargs字典
        name = kwargs.setdefault('name', self.get_unique_name(op.__name__))
        # Figure out the layer inputs.
        if len(self.inputs)==0:
            raise RuntimeError('No input variables found for layer %s.'%name)
        #此情况说明刚有输入层，即取输入数据即可
        elif len(self.inputs)==1:
            layer_input = self.inputs[0]
        else:
            layer_input = list(self.inputs)
        # Perform the operation and get the output.
        #开始做卷积，做pool操作！！！！正式开始做操作的是这里，而不是函数定义，会发现下面函数定义中与所给参数个数不符合，原因在于input没给定
        layer_output = op(self, layer_input, *args, **kwargs)
        # Add to layer LUT.
        #在self.layer中添加该name操作信息
        self.layers[name] = layer_output
        # This output is now the input for the next layer.
        #将该output添加到inputs中
        self.feed(layer_output)
        # Return self for chained calls.
        return self
    return layer_decorated

class Network(object):
    def __init__(self, inputs, trainable=True):
        self.inputs = []
        self.layers = dict(inputs)
        self.trainable = trainable
        self.setup()

    def setup(self):
        raise NotImplementedError('Must be subclassed.')

    def load(self, data_path, session, saver, ignore_missing=False):
        if data_path.endswith('.ckpt'):
            saver.restore(session, data_path)
        else:
            data_dict = np.load(data_path).item()
            for key in data_dict:
                with tf.variable_scope(key, reuse=True):
                    for subkey in data_dict[key]:
                        try:
                            var = tf.get_variable(subkey)
                            session.run(var.assign(data_dict[key][subkey]))
                            print "assign pretrain model "+subkey+ " to "+key
                        except ValueError:
                            print "ignore "+key
                            if not ignore_missing:

                                raise
    #*args中存的是多余的变量，且无标签，存在tuple中，如果有标签，则需要将函数改为feed(self, *args，**kwargs):
    #**kwargs为一个dict
    #layers为一个dict，inputs为一个list
    def feed(self, *args):
        #如果没给参数，就raise一个error
        assert len(args)!=0
        self.inputs = []
        for layer in args:
            #先判断如果给定参数是一个str
            if isinstance(layer, basestring):
                #self.layers在VGGnet_train 重载，为一个有值的dict
                try:
                    #将layer改为真实的variable，虽然目前还只是数据流图的一部分，还没有真正的开始运作
                    layer = self.layers[layer]
                    print layer
                except KeyError:
                    print self.layers.keys()
                    raise KeyError('Unknown layer name fed: %s'%layer)
            #将取出的layer数据存入input列表
            self.inputs.append(layer)
        return self

    def get_output(self, layer):
        try:
            #self.layers在VGGnet_train.py中重载了，为一个dict，记录的是每一层的输出
            layer = self.layers[layer]
        except KeyError:
            print self.layers.keys()
            raise KeyError('Unknown layer name fed: %s'%layer)
        return layer

    #得到唯一的名字，prefix传回来的是conv、max_pool..
    #self.layers为一个dict，item将其转换为可迭代形式
    def get_unique_name(self, prefix):
        # startswith() 方法用于检查字符串是否是以指定子字符串开头，返回true与false
        #即查看有没有conv开头的key，记录有的个数（true），相加再加1为id
        id = sum(t.startswith(prefix) for t,_ in self.layers.items())+1
        #返回的就是类似与conv_4
        return '%s_%d'%(prefix, id)

    #此函数就是在tensorflow格式下建立变量
    def make_var(self, name, shape, initializer=None, trainable=True):
        return tf.get_variable(name, shape, initializer=initializer, trainable=trainable)
    #判断padding类型是否符合要求
    def validate_padding(self, padding):
        assert padding in ('SAME', 'VALID')

    @layer
    #就因为上面的属性函数，是的真正的conv操作没有在这里进行，而是在上面的layer函数中进行
    def conv(self, input, k_h, k_w, c_o, s_h, s_w, name, relu=True, padding=DEFAULT_PADDING, group=1, trainable=True):
        #判断padding是否为same与valid的一种
        self.validate_padding(padding)
        #shape最后一位为深度
        #input形状为[batch, in_height, in_width, in_channels]
        #c_i.c_o分别为输入激活图层的深度，与输入激活图层深度，即卷积核个数
        c_i = input.get_shape()[-1]
        assert c_i%group==0
        assert c_o%group==0
        ##conv2d中stride[]第一位与最后一位都必须为1，第一位表示在batch上的位移，第四位表示在深度方向上的位移,i与k目前没找到定义，应该为input与卷积核
        #目前问题解决，lambda相当与一个def，只是定义函数
        ##1.将参数filter变为一个二维矩阵，形状为：[filter_height*filter_width*in_channels,output_channels]

        #2.将输入（input）转化为一个具有如下形状的Tensor，形状为：[batch,out_height,out_width,filter_height * filter_width * in_channels]

        #3.操作sum_{di, dj, q} input[b, strides[1] * i + di, strides[2] * j + dj, q] *filter[di, dj, q, k]

        convolve = lambda i, k: tf.nn.conv2d(i, k, [1, s_h, s_w, 1], padding=padding)
        with tf.variable_scope(name) as scope:
            #采取截断是正态初始化权重，这只是一种initializer方法，mean=0,stddev=0.01
            init_weights = tf.truncated_normal_initializer(0.0, stddev=0.01)
            #这也只是定义initializer的方法，初始化为0
            init_biases = tf.constant_initializer(0.0)
            #make_var就是用get_variable来建立变量
            #weight.shape[高，宽，深度，多少个]
            kernel = self.make_var('weights', [k_h, k_w, c_i/group, c_o], init_weights, trainable)
            biases = self.make_var('biases', [c_o], init_biases, trainable)

            if group==1:
                conv = convolve(input, kernel)
            else:
                #如果group不为0,将input与kernel第4个维度，即深度信息平分为group组
                input_groups = tf.split(3, group, input)
                kernel_groups = tf.split(3, group, kernel)
                output_groups = [convolve(i, k) for i,k in zip(input_groups, kernel_groups)]
                #分开的group组合起来
                conv = tf.concat(3, output_groups)
            if relu:
                bias = tf.nn.bias_add(conv, biases)
                return tf.nn.relu(bias, name=scope.name)
            return tf.nn.bias_add(conv, biases, name=scope.name)

    @layer
    def relu(self, input, name):
        return tf.nn.relu(input, name=name)

    @layer
    def max_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.max_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)

    @layer
    def avg_pool(self, input, k_h, k_w, s_h, s_w, name, padding=DEFAULT_PADDING):
        self.validate_padding(padding)
        return tf.nn.avg_pool(input,
                              ksize=[1, k_h, k_w, 1],
                              strides=[1, s_h, s_w, 1],
                              padding=padding,
                              name=name)10635000012334

    @layer
    def proposal_layer(self, input, _feat_stride, anchor_scales, cfg_key, name):
        """
        对应proposal层：结合anchor_pred_reg和fg anchors输出proposals，并进一步剔除无效anchors
        """
        #cfg_key为TRAIN
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        #就是返回blob，内容为[proposal引索(全0)，proposal]，shape（proposals.shape[0],5）,引索(全0)占一列，proposal占4列
        return tf.reshape(tf.py_func(proposal_layer_py,[input[0],input[1],input[2], cfg_key, _feat_stride, anchor_scales], [tf.float32]),[-1,5],name =name)

    @layer
    def roi_pool(self, input, pooled_height, pooled_width, spatial_scale, name):
        """
        Rol pooling层有2个输入：
        1. 原始的feature maps
        2. RPN输出的proposal boxes（大小各不相同）
        """
        # only use the first input
        if isinstance(input[0], tuple):
            input[0] = input[0][0]

        if isinstance(input[1], tuple):
            input[1] = input[1][1]

        print input
        return roi_pool_op.roi_pool(input[0], input[1], pooled_height, pooled_width, spatial_scale, name = name)[0]

    @layer
    def anchor_target_layer(self, input, _feat_stride, anchor_scales, name):
        """
        anchor_target_layer:对应rpn-data,训练阶段用来产生计算rpn_cls_loss和rpn_loss_bbox的参数：
        rpn_labels:[achors.shape[0], 1],所有anchors的fg/bg信息，（-1，0，1），-1代表不参加训练
        rpn_bbox_targets:[anchors.shape[0], 4]，所有anchors与gtbox之间的回归值
        rpn_inside_weights,rpn_outside_weights分别代表fg/bg anchors的初始化权重

        input为'rpn_cls_score','gt_boxes','im_info','data'信息组成的一个列表，input[0]为rpn_cls_score信息
        """
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        
        with tf.variable_scope(name) as scope:
            #tf.py_func将任意的python函数func转变为TensorFlow op,格式tf.py_func(func, inp, Tout, stateful=True, name=None)
            #func为python函数，inp为输入（ndarray），Tout为自定义输出格式，下面的输入输出分别为[input[0],input[1],input[2],input[3], _feat_stride, anchor_scales],[tf.float32,tf.float32,tf.float32,tf.float32]
            #rpn_label中存的是所有anchor的label（-1,0,1），
            #rpn_bbox_targets是所有anchor的4回归值,对于标签为-1的anchor，4个回归值全是0，
            #rpn_bbox_inside_weights,rpn_bbox_outside_weights是两个权重，初始化方式不一样
            #要将tf.float32类型转换为tf.Tensor类型
            #tf.convert_to_tensor函数，以确保我们处理张量而不是其他类型
            rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = tf.py_func(anchor_target_layer_py, [input[0], input[1], input[2], input[3], _feat_stride, anchor_scales], [tf.float32, tf.float32, tf.float32, tf.float32])
            rpn_labels = tf.convert_to_tensor(tf.cast(rpn_labels), name = 'rpn_labels')
            rpn_bbox_targets = tf.convert_to_tensor(tf.cast(rpn_bbox_targets), name = 'rpn_bbox_targets')
            rpn_bbox_inside_weights = tf.convert_to_tensor(tf.cast(rpn_bbox_inside_weights), name = 'rpn_bbox_inside_weights')
            rpn_bbox_outside_weights = tf.convert_to_tensor(tf.cast(rpn_bbox_outside_weights), name = 'rpn_bbox_outside_weights')

            return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

    @layer
    def proposal_target_layer(self, input, classes, name):
        """
        proposal_target_layer:训练阶段用来产生预测的分类好的最终proposals，三个（len(rois),4*21）大小的矩阵，和bbox_targets（和gt之间的差别，用来精修proposals），bbox_inside_weights和bbox_outside_weights
        """
        if isinstance(input[0], tuple):
            input[0] = input[0][0]
        with tf.variable_scope(name) as scope:
            #产生筛选后的roi，对应labels，三个（len(rois),4*21）大小的矩阵，其中一个对fg-roi对应引索行的对应类别的4个位置填上（dx,dy,dw,dh），另两个对fg-roi对应引索行的初始权重对应类别的4个位置填上（1,1,1,1)
            rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights = tf.py_func(proposal_target_layer_py, [input[0], input[1], classes], [tf.float32,tf.float32,tf.float32,tf.float32,tf.float32])
            rois = tf.reshape(rois, [-1, 5], name = 'rois')
            # 要将tf.float32类型转换为tf.Tensor类型
            # tf.convert_to_tensor函数，以确保我们处理张量而不是其他类型
            labels = tf.convert_to_tensor(tf.cast(labels,tf.int32), name = 'labels')
            bbox_targets = tf.convert_to_tensor(bbox_targets, name = 'bbox_targets')
            bbox_inside_weights = tf.convert_to_tensor(bbox_inside_weights, name = 'bbox_inside_weights')
            bbox_outside_weights = tf.convert_to_tensor(bbox_outside_weights, name = 'bbox_outside_weights')

            return rois, labels, bbox_targets, bbox_inside_weights, bbox_outside_weights

    @layer
    def reshape_layer(self, input, d, name):
        """
        在caffe基本数据结构blob中以如下形式保存数据：blob=[batch_size, channel*2，height，width]
        而在softmax分类时需要进行fg/bg二分类，所以reshape layer会将其变为[1, 2, channel*H, W]大小
        即单独“腾空”出来一个维度以便softmax分类，之后再reshape回复原状
        """
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob_reshape':
            # 还原回rpn_cls_prob_reshape的信息位置格式
            return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
                    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)/tf.cast(d,tf.float32)*tf.cast(input_shape[3],tf.float32),tf.int32),input_shape[2]]),[0,2,3,1],name=name)
        else:
            # 假设rpn_cls_score.shape为[1,n,n,18],最后reshape成[1,9n,n,2]
            #假如rpn_cls_score.shape为[1,3,3,18]，元素内容为range（3*3*18），最后得到的形状为[0,81],[1,82],[2,83]..意思为前81个元素（3*3*9）为bg，后81个元素对应fg，0与36对应着该featuremap的位置i，对应原图的可视野为前景或者背景的概率
            # 当然需要再一步softmax才能给出该可视野为fg与bg的概率
            return tf.transpose(tf.reshape(tf.transpose(input,[0,3,1,2]),[input_shape[0],
                    int(d),tf.cast(tf.cast(input_shape[1],tf.float32)*(tf.cast(input_shape[3],tf.float32)/tf.cast(d,tf.float32)),tf.int32),input_shape[2]]),[0,2,3,1],name=name)

    @layer
    def feature_extrapolating(self, input, scales_base, num_scale_base, num_per_octave, name):
        return feature_extrapolating_op.feature_extrapolating(input, scales_base, num_scale_base, num_per_octave, name = name)

    @layer
    def lrn(self, input, raadius, alpha, beta, name, bias = 1.0):
        """
        一种正则归一化方法
        """
        return tf.nn.local_response_normalization(input, depth_radius=radius, alpha=alpha, beta=beta, bias=bias, name=name) 

    @layer
    def concat(self, inputs, axis, name):
        """
        返回tensor
        """
        return tf.concat(concat_dim = axis, values = inputs, name = name)

    @layer
    def fc(self, input, num_out, name, relu = True, trainable = True):
        with tf.variable_scope(name) as scope:
            # only use the first input
            if isinstance(input, tuple):
                input = input[0]

            input_shape = input.get_shape()
            if input_shape.ndims == 4:
                # 将特征图转换成特征向量
                dim = 1
                for d in input_shape[1:].as_list():
                    dim *= d
                feed_in = tf.reshape(tf.transpose(input, [0, 3, 1, 2]), [-1, dim])
            else:
                # 输入即特征向量
                feed_in, dim = (input, int(input_shape[-1]))

            if name == 'bbox_pred':
                init_weights = tf.truncated_normal_initializer(0.0, stddev = 0.001)
                init_biases = tf.constant_initializer(0.0)
            else:
                init_weights = tf.truncated_normal_initializer(0.0, stddev = 0.01)
                init_biases = tf.constant_initializer(0.0)

            weights = self.make_var('weights', [dim, num_out], init_weights, trainable)
            biases = self.make_var('biases', [num_out], init_biases, trainable)

            op = tf.nn.relu_layer if relu else tf.nn.xw_plus_b
            fc = op(feed_in, weights, biases, name = scope.name)

            return fc

    @layer
    # 多项逻辑斯特回归
    def softmax(self, input, name):
        input_shape = tf.shape(input)
        if name == 'rpn_cls_prob':
            # 就是对原始数据进行softmax激活
            return tf.reshape(tf.nn.softmax(tf.reshape(input,[-1,input_shape[3]])),[-1,input_shape[1],input_shape[2],input_shape[3]],name=name)
        else:
            return tf.nn.softmax(input, name = name)

    @layer
    # dropout防止过拟合
    def dropout(self, input, keep_prob, name):
        return tf.nn.dropout(input, keep_prob, name = name)