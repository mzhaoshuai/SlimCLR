#!/bin/bash
export PYTHONUNBUFFERED="True"
### Remember to change ROOT to /YOUR/PATH ###
ROOT=/home/shuzhao/Data
#############################################
### simply set test_only=1 for inference, comment the SSL pre-trained stage or use a pre-trained checkpoint
test_only=0

# dataset
dataset=imagenet
lmdb_dataset=None
dataset_dir=${ROOT}/dataset/imagenet

# learning strategy
lr=2.4
slim_start_lr=3.2
weight_decay=1e-6
optimizer=LARS
lr_mode=cos
lr_milestones="60 80"
momentum=0.9
num_epochs=300
warmup_proportion=0.033
batch_size_per_gpu=128
precision=amp
use_bn_sync=1

# dataloader
num_workers=8

# slimmable networks
slim_fc=mocov3_slim
slimmable_training=1
width_mult_list="1.0 0.75 0.5 0.25"
inplace_distill=0
seed_loss=1
align_loss=0
mse_loss=0
aux_loss_w=0.5
slim_start_epoch=-1
slim_sub_no_clr=0
is_log_grad=0
slim_loss_weighted=5
teacher_T=1.0
student_T=1.0
overlap_weight=1

# MoCov3
ssl_arch=mocov3
moco_t=1.0
moco_m=0.99
moco_mlp_dim=4096
moco_dim=256
moco_m_cos=1
ssl_aug=mocov3

# resume and log
resume=None

# runfile
runfile_ssl=${ROOT}/code/SlimCLR/main_ssl.py
runfile_ssl_lincls=${ROOT}/code/SlimCLR/main_ssl_lincls.py

for num in 01
do
# STEP 1. FIRST pretraining
case ${num} in
	01 )
		lr=1.2
		slim_start_lr=3.2
		num_epochs=300
		slim_start_epoch=150
		warmup_proportion=0.033
		# We use a total batch size of 1024
		batch_size_per_gpu=128
		gradient_accumulation_steps=1
		slimmable_training=1
		# resume=${ROOT}/output/slimclr-mocov3-ep300-57e298e9cd.pth.tar
		;;
	* )
		;;
esac

log_dir=${ROOT}/output/${ssl_arch}_slim${slimmable_training}_${dataset}_${num}
grad_log=${log_dir}/grad_${num}.txt
echo "The model dir is ${log_dir}"

python ${runfile_ssl} --dataset_dir ${dataset_dir} \
					--lmdb_dataset ${lmdb_dataset} \
					--lr ${lr} \
					--slim_start_lr ${slim_start_lr} \
					--optimizer ${optimizer} \
					--weight_decay ${weight_decay} \
					--lr_mode ${lr_mode} \
					--lr_milestones ${lr_milestones} \
					--num_epochs ${num_epochs} \
					--momentum ${momentum} \
					--warmup_proportion ${warmup_proportion} \
					--use_bn_sync ${use_bn_sync} \
					--log_dir ${log_dir} \
					--batch_size_per_gpu ${batch_size_per_gpu} \
					--precision ${precision} \
					--test_only ${test_only} \
					--num_workers ${num_workers} \
					--slim_fc ${slim_fc} \
					--slimmable_training ${slimmable_training} \
					--inplace_distill ${inplace_distill} \
					--width_mult_list ${width_mult_list} \
					--resume ${resume} \
					--ssl_arch ${ssl_arch} \
					--ssl_aug ${ssl_aug} \
					--moco_t ${moco_t} \
					--moco_dim ${moco_dim} \
					--moco_mlp_dim ${moco_mlp_dim} \
					--moco_m ${moco_m} \
					--moco_m_cos ${moco_m_cos} \
					--seed_loss ${seed_loss} \
					--mse_loss ${mse_loss} \
					--aux_loss_w ${aux_loss_w} \
					--teacher_T ${teacher_T} \
					--student_T ${student_T} \
					--slim_start_epoch ${slim_start_epoch} \
					--slim_loss_weighted ${slim_loss_weighted} \
					--gradient_accumulation_steps ${gradient_accumulation_steps}

# STEP 2. FINETUNE
case ${num} in
   * )
		lr=0.4
		lincls_pretrained=${log_dir}/ckpt.pth.tar
		# lincls_pretrained=${ROOT}/output/slimclr-mocov3-ep300-57e298e9cd.pth.tar
		lincls_resume=None
		;;
esac

lincls_log_dir=${ROOT}/output/${ssl_arch}_slim${slimmable_training}_lincls_${dataset}_${num}
echo "The model dir is ${lincls_log_dir}"

python ${runfile_ssl_lincls} --dataset_dir ${dataset_dir} \
						--lmdb_dataset ${lmdb_dataset} \
						--lr ${lr} \
						--optimizer 'SGD' \
						--weight_decay '0.0' \
						--lr_mode 'cos' \
						--lr_milestones ${lr_milestones} \
						--num_epochs '90' \
						--momentum ${momentum} \
						--warmup_proportion '0.0' \
						--use_bn_sync '0' \
						--log_dir ${lincls_log_dir} \
						--batch_size_per_gpu ${batch_size_per_gpu} \
						--precision "fp32" \
						--test_only ${test_only} \
						--num_workers ${num_workers} \
						--slimmable_training ${slimmable_training} \
						--slim_fc 'supervised_switch' \
						--inplace_distill '1' \
						--inplace_distill_mixed '1' \
						--width_mult_list ${width_mult_list} \
						--resume ${lincls_resume} \
						--lincls_pretrained ${lincls_pretrained} \
						--ssl_aug 'normal' \
						--ssl_arch ${ssl_arch}

done

echo "Training Finished!!!"