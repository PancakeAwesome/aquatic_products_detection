#!/usr/bin/env python

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Train a Fast R-CNN network on a region of interest database."""

import _init_paths
from fast_rcnn.train import get_training_roidb, train_net
from fast_rcnn.config import cfg,cfg_from_file, cfg_from_list, get_output_dir
from datasets.factory import get_imdb
from networks.factory import get_network
import argparse
import pprint
import numpy as np
import sys
import pdb


if __name__ == '__main__':
    args = parse_args()

    print('Called')
    print(args)
    # 如果还有其他配置文件，就加载
    if args.cf_file is not None:
        cfg_from_file(args.cfg_file)
    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs)

    print('Using config:')
    # 已知类型的前提下，可以使用ppring来标准打印
    pprint.pprint(cfg)

    if not args.randomize:
        # fix the random seeds (numpy and caffe) for reproducibility
        np.random.seed(cfg.RNG_SEED)
    #imdb为存在一个字典(easydict)里的pascal_voc类的一个对象，e.g.{voc_2007_train:内容，voc_2007_val:内容，voc_2007_test:内容,voc_2007_test:内容,voc_2012_train:内容...}
    #内容里有该类里的各种self名称与操作，包括roi信息等等
    #字典中存放了4个key，分别是boxes信息，每个box的class信息，是否是flipped的标志位，重叠信息gt_overlaps
    imdb = get_imdb(args.imdb_name)
    print('Loaded dataset `{:s}` for training'.format(imdb.name))
    #get_training_roidb函数返回imdb对象的各种roi与图片信息，用于训练
    #这是一个列表，列表中存的是各个图片的字典，字典中存roi信息，字典引索为图片引索
    roidb = get_training_roidb(imdb)
    # 输出全路径
    output_dir = get_output_dir(imdb, None)
    print 'Output will be saved to `{:s}`'.format(output_dir)
    device_name = '/{}:{:d}'.format(args.device,args.device_id)
    print (device_name)
    #得到网络结构，参数（包括rpn）
    network = get_network(args.network_name)
    print 'Use network `{:s}` in training'.format(args.network_name)

    train_net(network, imdb, roidb, output_dir, pretrained_model = args.pretrained_model, max_iters = args.max_iters)