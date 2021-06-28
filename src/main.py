#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from model.train_val import train_val_run
from model.test import test_run
from utils.arg_parser import init_parser
import setproctitle

if __name__ == '__main__':
    # TODO: distributed data parallel
    # TODO: use our own mean and std for normalization

    setproctitle.setproctitle('STCNet')
    parser = init_parser()
    args = parser.parse_args()

    '''DEVICE'''
    gpu = args.gpu
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu)
    torch.cuda.set_device(0)
    device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")

    '''REPRODUCIBILITY'''
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
        # set cudnn.benchmark as False for REPRODUCIBILITY, at the cost of reduced performance.

    func_args = {
        'device': device,
        'img_height': args.img_height,
        'img_width': args.img_width,
    }

    if not args.test:
        train_val_run(**func_args)
    else:
        func_args.update({'mode': args.mode})
        test_run(**func_args)