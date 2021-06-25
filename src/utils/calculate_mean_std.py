#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import timeit


def cal_dir_stat(root):
    CHANNEL_NUM = 3
    pixel_num = 0
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)
    img_size = (224, 224)

    im_pths = glob(join(root, "*.npy"))
    for path in im_pths:
        ims = np.load(path)
        nums = ims.shape[0]
        for i in range(nums):
            im = ims[i]
            im = cv2.resize(im, img_size, interpolation=cv2.INTER_LINEAR)
            im = im / 255.0
            pixel_num += (im.size / CHANNEL_NUM)
            channel_sum += np.sum(im, axis=(0, 1))
            channel_sum_squared += np.sum(np.square(im), axis=(0, 1))

    rgb_mean = channel_sum / pixel_num
    rgb_std = np.sqrt(channel_sum_squared / pixel_num - np.square(rgb_mean))

    return rgb_mean, rgb_std

if __name__ == '__main__':
    train_root = '../data/rgb/'
    start = timeit.default_timer()
    mean, std = cal_dir_stat(train_root)
    end = timeit.default_timer()
    print("elapsed time: {}".format(end - start))
    print("mean:{}\nstd:{}".format(mean, std))
