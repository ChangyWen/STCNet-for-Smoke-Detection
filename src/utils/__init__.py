__all__=['download_videos', 'process_videos', 'split_metadata', 'util', 'optical_flow', 'arg_parser']

import os
import sys
import inspect
cmd_subfolder = os.path.split(inspect.getfile(inspect.currentframe()))[0]
if cmd_subfolder not in sys.path:
    sys.path.insert(0, cmd_subfolder)