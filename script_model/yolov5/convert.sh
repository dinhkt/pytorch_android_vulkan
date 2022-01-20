#!/bin/bash
# to build for CPU use, remove --vulkan
python ./yolov5/export.py --weights yolov5s.pt --include torchscript --optimize --vulkan
