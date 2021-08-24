#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import argparse


def init_parser():
    parser = argparse.ArgumentParser(description='STCNet')

    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--img_width', type=int, default=224)
    parser.add_argument('--img_height', type=int, default=224)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--test', default=False, action='store_true')
    parser.add_argument('--mode', default='test')  # 'test' or 'validation'
    return parser
