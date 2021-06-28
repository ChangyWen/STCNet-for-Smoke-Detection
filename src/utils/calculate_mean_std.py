#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from os import listdir
from os.path import join, isdir
from glob import glob
import cv2
import timeit
from util import load_json
import setproctitle


def cal_dir_stat(im_pths):
    CHANNEL_NUM = 3
    pixel_num = 0
    channel_sum = np.zeros(CHANNEL_NUM)
    channel_sum_squared = np.zeros(CHANNEL_NUM)
    img_size = (224, 224)

    # im_pths = glob(join(root, "*.npy"))
    for j in range(len(im_pths)):
        if j % 100 == 0: print('*' * 18, '\t',  j, '\t', '*' * 18)
        path = im_pths[j]
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
    setproctitle.setproctitle('cal-mean-std')

    metadata = []
    json_name = [
        '../data/split/metadata_{}_split_0_by_camera.json',
        '../data/split/metadata_{}_split_1_by_camera.json',
        '../data/split/metadata_{}_split_2_by_camera.json',
        '../data/split/metadata_{}_split_by_date.json',
        '../data/split/metadata_{}_split_3_by_camera.json',
        '../data/split/metadata_{}_split_4_by_camera.json',
    ]
    file_name = [f.format('train') for f in json_name]
    for f in file_name:
        metadata.extend(load_json(f))
    rgb_dir = "../data/rgb/"
    im_pths = []
    for data in metadata:
        file_name = data['file_name'].replace('-180-180-', '-320-320-')
        im_pths.append(rgb_dir + file_name + '.npy')
    print('len(im_pths):', len(im_pths))
    start = timeit.default_timer()
    mean, std = cal_dir_stat(im_pths)
    end = timeit.default_timer()
    print("elapsed time: {}".format(end - start))
    print("mean:{}\nstd:{}".format(mean, std))
