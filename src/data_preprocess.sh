#!/usr/bin/env bash

resolution=$1

set -x
python utils/split_metadata.py confirm ${resolution}
python utils/download_videos.py
python utils/process_videos.py