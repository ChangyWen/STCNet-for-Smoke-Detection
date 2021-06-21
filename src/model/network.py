#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
from collections import OrderedDict
from senet import se_resnext50_32x4d, se_resnext101_32x4d

class STCNet(nn.Module):
    def __init__(self, backbone='se_resnext50_32x4d', pretrained='imagenet'):
        super(STCNet, self).__init__()
        if backbone == 'se_resnext50_32x4d':
            self.spatial_path = se_resnext50_32x4d(pretrained=pretrained)
            self.temporal_path = se_resnext50_32x4d(pretrained=pretrained)
        else:
            self.spatial_path = se_resnext101_32x4d(pretrained=pretrained)
            self.temporal_path = se_resnext101_32x4d(pretrained=pretrained)
        self.layer_cls = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(in_channels=2048, out_channels=256, kernel_size=(1, 1))),
            ('conv2', nn.Conv2d(in_channels=256, out_channels=256, kernel_size=(1, 1))),
            ('avg_pool', nn.AdaptiveAvgPool2d(output_size=(1, 1)))
        ]))
        self.layer_out = nn.Sequential(OrderedDict([
            ('last_linear', nn.Linear(in_features=256, out_features=2)),
            ('log_softmax', nn.LogSoftmax(dim=1))
        ]))
        self.loss = nn.NLLLoss(reduction='mean')

    def op_mean(self, x):
        return torch.mean(x, dim=1, keepdim=True)

    def op_fuse(self, x1, x2):
        return torch.add(x1, x2)

    def forward(self, rgb, residual, targets=None, is_testing=False):
        # rgb_dim, residual_dim = rgb.ndimensions(), residual.ndimensions()
        # assert rgb_dim == residual_dim and rgb_dim == 5 # (B, T, C, H, W)
        B, T, C, H, W = rgb.size()
        rgb = rgb.view(-1, C, H, W)
        residual = residual.view(-1, C, H, W)

        s0 = self.spatial_path.layer0(rgb)
        s1 = self.spatial_path.layer1(s0)

        t0 = self.temporal_path.layer0(residual)
        t1 = self.temporal_path.layer1(t0)

        s2 = self.spatial_path.layer2(self.op_fuse(s1, t1))
        t2 = self.temporal_path.layer2(self.op_fuse(s1, t1))

        s3 = self.spatial_path.layer3(self.op_fuse(s2, t2))
        t3 = self.temporal_path.layer3(self.op_fuse(s2, t2))

        s4 = self.spatial_path.layer4(self.op_fuse(s3, t3))
        t4 = self.temporal_path.layer4(self.op_fuse(s3, t3))

        x0 = self.layer_cls(self.op_fuse(s4, t4))
        x0 = x0.unflatten(dim=0, sizes=torch.Size([B, T]))
        x0 = self.op_mean(x0)
        x0 = torch.flatten(input=x0, start_dim=1, end_dim=-1)
        preds = self.layer_out(x0)

        loss = self.loss(preds, targets) if not is_testing else None
        return preds, loss
