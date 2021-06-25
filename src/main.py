#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from model.train_val import train_val_run
from utils.arg_parser import init_parser

if __name__ == '__main__':
    # TODO: multiple gpus

    # TODO: use our own mean and std for normalization

    parser = init_parser()
    args = parser.parse_args()

    gpu = args.gpu
    seed = args.seed
    img_height = args.img_height
    img_width = args.img_width

    '''DEVICE'''
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu)
    torch.cuda.set_device(gpu)
    device = torch.device("cuda:{}".format(gpu) if torch.cuda.is_available() else "cpu")

    '''REPRODUCIBILITY'''
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
        # set cudnn.benchmark as False for REPRODUCIBILITY, at the cost of reduced performance.

    train_val_run(
        device=device, img_height=img_height, img_width=img_width
    )