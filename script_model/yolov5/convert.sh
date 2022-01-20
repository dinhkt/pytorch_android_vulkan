#!/bin/bash
# replace export.py file in this folder with export.py in cloned yolov5 folder  
# to build for CPU use, remove --vulkan
python ./yolov5/export.py --weights yolov5s.pt --include torchscript --optimize --vulkan
