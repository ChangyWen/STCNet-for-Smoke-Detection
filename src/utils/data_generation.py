#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from util import load_json, split_list, random_scale, generate_random_crop_pos, \
    random_crop_pad_to_shape, normalize, get_pretrained_settings
from torch.utils.data import Dataset, DataLoader


class BaseDataset(Dataset):
    def __init__(
            self, split_name='all', img_height=224, img_width=224, mode='train',
            segs_per_clip=8, backbone='se_resnext50_32x4d', pretrained='imagenet'
    ):
        assert split_name in ['all', 's0', 's1', 's2', 's3', 's4', 's5'], split_name
        mapping = {'s0': 0, 's1': 1, 's2': 2, 's3': 3, 's4': 4, 's5': 5}
        self.rgb_path = '../data/rgb/'
        self.mode = mode
        json_name = [
            '../data/split/metadata_{}_split_0_by_camera.json',
            '../data/split/metadata_{}_split_1_by_camera.json',
            '../data/split/metadata_{}_split_2_by_camera.json',
            '../data/split/metadata_{}_split_by_date.json',
            '../data/split/metadata_{}_split_3_by_camera.json',
            '../data/split/metadata_{}_split_4_by_camera.json',
        ]
        file_name = [f.format(self.mode) for f in json_name]
        self.data_list = []
        if split_name == 'all':
            for f in file_name:
                self.data_list.extend(load_json(f))
        else:
            self.data_list.extend(load_json(file_name[mapping[split_name]]))
        self.img_height = img_height
        self.img_width = img_width
        self.img_size = (self.img_height, self.img_width)

        '''for convenience'''
        self.segs_per_clip = segs_per_clip
        num_frames = 36
        self.segs = split_list(num_frames, self.segs_per_clip + 1)
        self.alpha_array = [0.5, 0.75, 1, 1.25, 1.5]
        self.beta_array = [-70, -35, 0, 35, 70]
        self.scale_array = [0.5, 0.75, 1, 1.5, 1.75, 2.0]
        self.flip_array = [-1, 0, 1]
        self.rotate_array = [0, 2]
        pretrained_settings = get_pretrained_settings()
        self.mean = pretrained_settings[backbone][pretrained]['mean']
        self.std = pretrained_settings[backbone][pretrained]['std']

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        info = self.data_list[index]
        video_frames = np.load(self.rgb_path + info['file_name'] + '.npy')
        indices = [np.random.choice(seg) for seg in self.segs]
        temp = self.preprocess(video_frames[indices, :, :, :])
        frames = temp[:-1]
        res_frames = temp[1:] - frames
        frames = frames.transpose(0, 3, 1, 2).astype(np.float32)
        res_frames = res_frames.transpose(0, 3, 1, 2).astype(np.float32)
        label = info['label']
        return dict(frames=frames, res_frames=res_frames, label=label, filename=info['file_name'])

    def preprocess(self, frames):
        processed_frames = []
        use_augment = np.random.random() <= 0.3333
        if use_augment and self.mode == 'train':
            alpha = np.random.choice(self.alpha_array) if self.alpha_array else 1
            beta = np.random.choice(self.beta_array) if self.beta_array else 0
            scale = np.random.choice(self.scale_array) if self.scale_array else 1
            flip = np.random.choice(self.flip_array)
            rotate = np.random.choice(self.rotate_array)
            use_flip = np.random.random() <= 0.5
        for i in range(frames.shape[0]):
            frame = frames[i]
            frame = cv2.resize(frame, self.img_size, interpolation=cv2.INTER_LINEAR)
            if use_augment and self.mode == 'train':
                frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
                frame = cv2.flip(frame, flip) if use_flip else cv2.rotate(frame, rotate)
                frame = random_scale(frame, scale)
                crop_pos = generate_random_crop_pos(frame.shape[:2], self.img_size)
                frame, _ = random_crop_pad_to_shape(frame, crop_pos, self.img_size, 0)
            frame = normalize(frame, self.mean, self.std)
            processed_frames.append(frame)
        return np.stack(processed_frames, axis=0)


def get_DataLoader(
        mode='train', backbone='se_resnext50_32x4d', pretrained='imagenet',
        batch_size=4, drop_last=True, is_shuffle=True, pin_memory=True,
        img_height=224, img_width=224,
):
    dataset = BaseDataset(
        mode=mode, backbone=backbone, pretrained=pretrained, img_height=img_height, img_width=img_width
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=is_shuffle,
        pin_memory=pin_memory
    )
    return data_loader