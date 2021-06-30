#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import sys
from tqdm import tqdm
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from network import STCNet
from utils.util import ensure_dir
from utils.data_generation import get_DataLoader
import pandas as pd


def test_run(
        device, img_height=224, img_width=224, mode='test',
        backbone='50', pretrained='imagenet', batch_size=32,
        model_dir='../trained_model/', output_dir='../pred_out/'
):
    assert backbone in ['50', '101']
    backbone = 'se_resnext{}_32x4d'.format(backbone)
    model = STCNet(backbone=backbone, pretrained=pretrained)

    model_path = model_dir + '/state_dict.pth'
    model.load_state_dict(torch.load(model_path))

    model.to(device)
    model.eval()

    assert mode in ['test', 'validation']
    test_dataloader = get_DataLoader(
        mode=mode, batch_size=batch_size, backbone=backbone, pretrained=pretrained,
        img_height=img_height, img_width=img_width, is_shuffle=False,
    )
    iter_test_dataloader = iter(test_dataloader)
    niters = len(test_dataloader)

    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    pbar = tqdm(range(niters), file=sys.stdout, bar_format=bar_format)

    all_label = []
    all_preds = []
    all_filename = []
    all_frame_idx = []
    ensure_dir(output_dir)

    with torch.no_grad():
        for idx in pbar:
            mini_batch = iter_test_dataloader.next()
            frames = mini_batch['frames']
            res_frames = mini_batch['res_frames']
            label = mini_batch['label']
            filename = mini_batch['filename']
            frame_idx = mini_batch['frame_idx']

            if device.type == 'cuda':
                frames = frames.cuda(device=device, non_blocking=True)
                res_frames = res_frames.cuda(device=device, non_blocking=True)
                label = label.cuda(device=device, non_blocking=True)

            preds, _ = model(rgb=frames, residual=res_frames, target=label, is_testing=True)
            preds = torch.argmax(preds, dim=1).cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            all_label.append(label)
            all_preds.append(preds)
            all_filename.append(filename)
            all_frame_idx.append(frame_idx)

            print_str = 'Iter{}/{}:'.format(idx + 1, niters)
            pbar.set_description(print_str, refresh=False)

    all_filename = np.concatenate(all_filename)
    all_frame_idx = np.concatenate(all_frame_idx)
    all_label = np.concatenate(all_label)
    all_preds = np.concatenate(all_preds)
    accuracy = accuracy_score(all_label, all_preds)
    precision = precision_score(all_label, all_preds)
    recall = recall_score(all_label, all_preds)
    f1 = f1_score(all_label, all_preds)

    df = pd.DataFrame({'filename': all_filename, 'frame_idx':all_frame_idx, 'label': all_label, 'preds': all_preds})
    df.to_csv(
        output_dir + '{}-New-ACC{:.4f}-PRE{:.4f}-REC{:.4f}-F{:.4f}.csv'.format(mode, accuracy, precision, recall, f1),
        sep=',', index=False, header=True
    )
    print('Saved, MODE:{}, ACC{:.6f}-PRE{:.6f}-REC{:.6f}-F{:.6f}'.format(mode, accuracy, precision, recall, f1))

    '''
    Performance on Testing set and Validation set
    test: ACC0.917490-PRE0.900443-REC0.876863-F0.888497
    validation: ACC0.925699-PRE0.923586-REC0.913358-F0.918443
    '''
