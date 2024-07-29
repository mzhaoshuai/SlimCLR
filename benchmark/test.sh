#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
### Remember to change ROOT to /YOUR/PATH ###
ROOT=/home/shuzhao/Data
### Also, change {ROOT} in {configs} and {mmdet/flags.py} 
#############################################
export PYTHONPATH=${ROOT}/code/SlimCLR/benchmark:$PYTHONPATH


pretrained_model_file=${ROOT}/output/slimclr_mocov2_ep800_mask_rcnn_r50_fpn_1x_78afcc6ae3.pth
python tools/test.py configs/mask_rcnn_r50_fpn_1x.py ${pretrained_model_file} --eval "bbox" "segm" --width_mult 1.0
python tools/test.py configs/mask_rcnn_r50_fpn_1x.py ${pretrained_model_file} --eval "bbox" "segm" --width_mult 0.75
python tools/test.py configs/mask_rcnn_r50_fpn_1x.py ${pretrained_model_file} --eval "bbox" "segm" --width_mult 0.5
python tools/test.py configs/mask_rcnn_r50_fpn_1x.py ${pretrained_model_file} --eval "bbox" "segm" --width_mult 0.25
