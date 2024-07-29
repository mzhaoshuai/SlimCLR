#!/bin/bash
export PYTHONUNBUFFERED="True"
# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
read gpus
export CUDA_VISIBLE_DEVICES=${gpus}

### Remember to change ROOT to /YOUR/PATH ###
ROOT=/home/shuzhao/Data
#############################################
test_only=0

# dataset
dataset_dir=${ROOT}/dataset/imagenet
lmdb_dataset=None
ssl_aug=slim

# learning strategy
lr=0.1
lr_mode=cos
num_epochs=100
weight_decay=1e-4
warmup_proportion=0.05
batch_size_per_gpu=128
precision=amp
use_bn_sync=0
optimizer=SGD
nesterov=0

# slimmable traning
slimmable_training=1
inplace_distill=0
slim_loss=sum
width_mult_list="1.0 0.75 0.5 0.25"
is_log_grad=0
num_workers=8
resume=None


runfile=${ROOT}/code/SlimCLR/main_slim.py
for num in 01
do
	case ${num} in
		01 )
			lr=0.2
			# We use 4 GPUs, and the total batch size is 4 x 128 = 512
			batch_size_per_gpu=128
			;;
		* )
			;;
	esac

log_dir=${ROOT}/output/supervised_slim_${num}
grad_log=${log_dir}/grad_${num}.txt
echo "The model dir is ${log_dir}"


python ${runfile} --dataset_dir ${dataset_dir} \
					--lmdb_dataset ${lmdb_dataset} \
					--ssl_aug ${ssl_aug} \
					--weight_decay ${weight_decay} \
					--optimizer ${optimizer} \
					--nesterov ${nesterov} \
					--lr ${lr} \
					--lr_mode ${lr_mode} \
					--num_epochs ${num_epochs} \
					--warmup_proportion ${warmup_proportion} \
					--use_bn_sync ${use_bn_sync} \
					--log_dir ${log_dir} \
					--batch_size_per_gpu ${batch_size_per_gpu} \
					--precision ${precision} \
					--test_only ${test_only} \
					--num_workers ${num_workers} \
					--slimmable_training ${slimmable_training} \
					--inplace_distill ${inplace_distill} \
					--width_mult_list ${width_mult_list} \
					--resume ${resume} \
					--grad_log ${grad_log} \
					--is_log_grad ${is_log_grad}

done

echo "Training Finished!!!"