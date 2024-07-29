#!/bin/bash
### Remember to change ROOT to /YOUR/PATH ###
ROOT=/home/shuzhao/Data
### Also, change {ROOT} in {configs} and {mmdet/flags.py} 
#############################################
export PYTHONPATH=${ROOT}/code/SlimCLR/benchmark:$PYTHONPATH


folder_id=mask_rcnn_r50_fpn_1x_num_01
config=mask_rcnn_r50_fpn_1x_8gpu_bs64.py
python -m torch.distributed.launch --nproc_per_node=$1 tools/train.py \
	configs/${config} \
	--launcher pytorch \
	--work_dir "${ROOT}/output/${folder_id}"


export CUDA_VISIBLE_DEVICES=0
pretrained_model_file=${ROOT}/output/${folder_id}/latest.pth
python tools/test.py configs/${config} ${pretrained_model_file} --eval "bbox" "segm" --width_mult 1.0