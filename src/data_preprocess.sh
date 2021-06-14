#!/usr/bin/env bash

resolution=$1

set -x
python utils/split_metadata.py confirm
python utils/download_videos.py ${resolution}
python utils/process_videos.py ${resolution}

