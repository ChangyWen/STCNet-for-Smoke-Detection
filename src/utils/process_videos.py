#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from util import *
from optical_flow.optical_flow import OpticalFlow
from multiprocessing import Pool

thread = "1"
os.environ["MKL_NUM_THREADS"] = thread
os.environ["NUMEXPR_NUM_THREADS"] = thread
os.environ["OMP_NUM_THREADS"] = thread
os.environ["VECLIB_MAXIMUM_THREADS"] = thread
os.environ["OPENBLAS_NUM_THREADS"] = thread
import cv2 as cv
cv.setNumThreads(12)

# Process videos into rgb frame files and optical flow files
# The file format is numpy.array

def init(res):
    global resolution
    resolution = res[0]

def main(argv):
    resolution = 180
    if len(argv) > 1:
        if argv[1] not in ['320', '180']: raise AttributeError('resolution')
        resolution = int(argv[1])

    rgb_dir = "../data/rgb/"
    flow_dir = "../data/flow/"
    metadata_path = "../data/metadata.json"
    num_workers = 2

    # Check for saving directories and create if they don't exist
    check_and_create_dir(rgb_dir)
    check_and_create_dir(flow_dir)

    metadata = load_json(metadata_path)
    p = Pool(num_workers, initializer=init, initargs=(resolution,))
    p.map(compute_and_save_flow, metadata)
    print("Done process_videos.py")


def compute_and_save_flow(video_data):
    if resolution == 320:
        file_name = video_data['file_name'].replace('-180-180-', '-320-320-')
    else:
        file_name = video_data["file_name"]
    # file_name = video_data["file_name"]
    video_dir = "../data/videos/"
    rgb_dir = "../data/rgb/"
    flow_dir = "../data/flow/"
    rgb_vid_in_p = str(video_dir + file_name + ".mp4")
    rgb_4d_out_p = str(rgb_dir + file_name + ".npy")
    flow_4d_out_p = str(flow_dir + file_name + ".npy")
    if not is_file_here(rgb_vid_in_p):
        return
    if is_file_here(rgb_4d_out_p):
        rgb_4d_out_p = None
    if is_file_here(flow_4d_out_p):
        flow_4d_out_p = None
    if rgb_4d_out_p is None and flow_4d_out_p is None:
        return
    # Saves files to disk in format (time, height, width, channel) as numpy array
    # flow_type = 1 # TVL1 optical flow
    flow_type = None # will not process optical flow
    op = OpticalFlow(rgb_vid_in_p=rgb_vid_in_p, rgb_4d_out_p=rgb_4d_out_p,
            flow_4d_out_p=flow_4d_out_p, clip_flow_bound=20, flow_type=flow_type)
    op.process()


if __name__ == "__main__":
    main(sys.argv)
