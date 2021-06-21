#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn
from model.train_val import train_val_run

if __name__ == '__main__':
    # TODO: multiple gpus
    # TODO: arg parser

    '''DEVICE'''
    gpu_device = 0
    os.environ['CUDA_VISIBLE_DEVICES'] = '{}'.format(gpu_device)
    torch.cuda.set_device(gpu_device)
    device = torch.device("cuda:{}".format(gpu_device) if torch.cuda.is_available() else "cpu")

    '''REPRODUCIBILITY'''
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
        # set cudnn.benchmark as False for REPRODUCIBILITY, at the cost of reduced performance.

    train_val_run(device)