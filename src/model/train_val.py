#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import sys
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import accuracy_score, precision_score, recall_score
from network import STCNet
from utils.data_generation import get_DataLoader
from utils.lr_policy import PolyLR, ExponentialLR
from utils.util import ensure_dir, get_pretrained_settings, \
    init_business_weight, initialize_pretrained_weight, group_weight


def train_val_run(
        device, img_height=224, img_width=224,
        backbone='50', pretrained = 'imagenet', bn_eps=1e-5, bn_momentum=0.1,
        lr=1e-3, lr_power=0.8, backbone_lr=1e-4, backbone_lr_power=0.8, momentum=0.9, weight_decay=5e-4,
        batch_size=4, nepochs=16, val_batch_size=48, val_per_iter=50, save_per_iter=200,
        tensorboard_dir='../tensorboard_log/', model_dir='../trained_model/'
):
    assert backbone in ['50', '101']
    backbone = 'se_resnext{}_32x4d'.format(backbone)
    model = STCNet(backbone=backbone, pretrained=pretrained)

    '''initialization'''
    params_list = []
    pretrained_settings = get_pretrained_settings()
    settings = pretrained_settings[backbone][pretrained]
    for pretrained_layer in [model.spatial_path, model.temporal_path]:
        initialize_pretrained_weight(pretrained_layer, settings=settings)
        params_list += group_weight(pretrained_layer, lr=backbone_lr)
    for business_layer in [model.layer_cls, model.layer_out]:
        init_business_weight(business_layer, bn_eps=bn_eps, bn_momentum=bn_momentum, mode='fan_in', nonlinearity='relu')
        params_list += group_weight(business_layer, lr=lr)

    optimizer = torch.optim.SGD(params_list, lr=lr, weight_decay=weight_decay, momentum=momentum)
    dataloader = get_DataLoader(
        mode='train', batch_size=batch_size, backbone=backbone, pretrained=pretrained,
        img_height=img_height, img_width=img_width
    )
    niters_per_epoch = len(dataloader)

    lr_policy = ExponentialLR(start_lr=lr, gamma=lr_power, niters_per_epoch=niters_per_epoch)
    backbone_lr_policy = ExponentialLR(start_lr=backbone_lr, gamma=backbone_lr_power, niters_per_epoch=niters_per_epoch)

    val_dataloader = get_DataLoader(
        mode='validation', batch_size=val_batch_size, backbone=backbone, pretrained=pretrained,
        img_height=img_height, img_width=img_width,
    )
    bar_format = '{desc}[{elapsed}<{remaining},{rate_fmt}]'
    ensure_dir(model_dir)
    ensure_dir(tensorboard_dir)
    writer = SummaryWriter(tensorboard_dir)
    log_idx = 0

    model.to(device)
    model.train()

    for epoch in range(nepochs):
        pbar = tqdm(range(niters_per_epoch), file=sys.stdout, bar_format=bar_format)
        iter_dataloader = iter(dataloader)

        for idx in pbar:
            optimizer.zero_grad()
            minibatch = iter_dataloader.next()
            frames = minibatch['frames']
            res_frames = minibatch['res_frames']
            label = minibatch['label']

            if device.type == 'cuda':
                frames = frames.cuda(non_blocking=True)
                res_frames = res_frames.cuda(non_blocking=True)
                label = label.cuda(non_blocking=True)

            preds, loss = model(rgb=frames, residual=res_frames, target=label, is_testing=False)
            current_idx = epoch * niters_per_epoch + idx

            backbone_lr = backbone_lr_policy.get_lr(current_idx)
            lr = lr_policy.get_lr(current_idx)
            for i in range(4):
                optimizer.param_groups[i]['lr'] = backbone_lr
                optimizer.param_groups[4 + i]['lr'] = lr

            loss.backward()
            optimizer.step()
            print_str = 'Epoch{}/{}'.format(epoch, nepochs) \
                        + ' Iter{}/{}:'.format(idx + 1, niters_per_epoch) \
                        + ' backbone_lr={:.4f}'.format(backbone_lr)\
                        + ' lr={:.4f}'.format(lr) \
                        + ' loss={:.4f}'.format(loss.item())
            pbar.set_description(print_str, refresh=False)

            if current_idx % val_per_iter == 0:
                model.eval()
                with torch.no_grad():
                    try:
                        val_minibatch = iter_val_dataloader.next()
                    except (StopIteration, NameError):
                        iter_val_dataloader = iter(val_dataloader)
                        val_minibatch = iter_val_dataloader.next()
                    frames_val = val_minibatch['frames']
                    res_frames_val = val_minibatch['res_frames']
                    label_val = val_minibatch['label']

                    if device.type == 'cuda':
                        frames_val = frames_val.cuda(non_blocking=True)
                        res_frames_val = res_frames_val.cuda(non_blocking=True)
                        label_val = label_val.cuda(non_blocking=True)

                    '''val'''
                    preds_val, loss_val = model(
                        rgb=frames_val, residual=res_frames_val, target=label_val, is_testing=False
                    )
                    preds_val = torch.argmax(preds_val, dim=1).cpu().detach().numpy()
                    label_val = label_val.cpu().detach().numpy()
                    accuracy_val = accuracy_score(label_val, preds_val)
                    precision_val = precision_score(label_val, preds_val)
                    recall_val = recall_score(label_val, preds_val)

                    '''train'''
                    preds = torch.argmax(preds, dim=1).cpu().detach().numpy()
                    label = label.cpu().detach().numpy()
                    accuracy = accuracy_score(label, preds)
                    precision = precision_score(label, preds)
                    recall = recall_score(label, preds)

                    writer.add_scalar('Loss/train', loss, log_idx + 1)
                    writer.add_scalar('Loss/val', loss_val, log_idx + 1)

                    writer.add_scalar('Score/accuracy_train', accuracy, log_idx + 1)
                    writer.add_scalar('Score/precision_train', precision, log_idx + 1)
                    writer.add_scalar('Score/recall_train', recall, log_idx + 1)

                    writer.add_scalar('Score/accuracy_val', accuracy_val, log_idx + 1)
                    writer.add_scalar('Score/precision_val', precision_val, log_idx + 1)
                    writer.add_scalar('Score/recall_val', recall_val, log_idx + 1)
                    log_idx += 1
                model.train()
            if current_idx % save_per_iter == 0 or (idx > len(pbar) - 2):
                torch.save(model.state_dict(), model_dir + '/state_dict.pth')
                torch.save(model, model_dir + '/model.pth')

    writer.flush()
    writer.close()