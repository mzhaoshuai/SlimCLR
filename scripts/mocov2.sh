#!/bin/bash
export PYTHONUNBUFFERED="True"
### Remember to change ROOT to /YOUR/PATH ###
ROOT=/home/shuzhao/Data
#############################################
### simply set test_only=1 for inference, comment the SSL pre-trained stage or use a pre-trained checkpoint
test_only=1

# dataset
dataset=imagenet
lmdb_dataset=None
dataset_dir=${ROOT}/dataset/imagenet

# learning strategy
lr=0.2
lr_mode=cos
optimizer=SGD
lr_milestones="60 80"
momentum=0.9
weight_decay=1e-4
num_epochs=200
warmup_proportion=0.05
batch_size_per_gpu=128
gradient_accumulation_steps=1
precision=amp
use_bn_sync=0
# dataloader
num_workers=8
ssl_aug=mocov2

# slimmable networks
ssl_arch=mocov2
slim_fc=mocov2_slim
slimmable_training=0
width_mult_list="1.0 0.75 0.5 0.25"
inplace_distill=0
seed_loss=1
mse_loss=0
atkd_loss=0
aux_loss_w=0.5
dkd_loss=0
slim_start_epoch=-1
is_log_grad=0
slim_loss_weighted=5
teacher_T=5.0
student_T=5.0
full_distill=0
overlap_weight=1

# moco
moco_t=0.2

# resume and log
resume=None

# runfile
runfile_ssl=${ROOT}/code/SlimCLR/main_ssl.py
runfile_ssl_lincls=${ROOT}/code/SlimCLR/main_ssl_lincls.py

for num in 02
do
# STEP 1. FIRST pretraining
case ${num} in
   01 )
		num_epochs=200
		slim_start_epoch=100
		precision=amp
		# We use a total batch size 1024
		lr=0.2
		batch_size_per_gpu=256
		warmup_proportion=0.05
		slimmable_training=1
		# resume=${ROOT}/output/slimclr-mocov2-ep200-48213217.pth.tar
		;;
   02 )
		num_epochs=800
		slim_start_epoch=400
		precision=amp
		# We use a total batch size 1024
		lr=0.2
		batch_size_per_gpu=256
		warmup_proportion=0.0125
		slimmable_training=1
		# resume=${ROOT}/output/slimclr-mocov2-ep800-eda810a6a9.pth.tar
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
					--slimmable_training ${slimmable_training} \
					--overlap_weight ${overlap_weight} \
					--slim_fc ${slim_fc} \
					--inplace_distill ${inplace_distill} \
					--width_mult_list ${width_mult_list} \
					--resume ${resume} \
					--ssl_aug ${ssl_aug} \
					--moco_t ${moco_t} \
					--seed_loss ${seed_loss} \
					--mse_loss ${mse_loss} \
					--atkd_loss ${atkd_loss} \
					--dkd_loss ${dkd_loss} \
					--aux_loss_w ${aux_loss_w} \
					--slim_start_epoch ${slim_start_epoch} \
					--grad_log ${grad_log} \
					--is_log_grad ${is_log_grad} \
					--slim_loss_weighted ${slim_loss_weighted} \
					--teacher_T ${teacher_T} \
					--student_T ${student_T} \
					--full_distill ${full_distill} \
					--gradient_accumulation_steps ${gradient_accumulation_steps}

# STEP 2. FINETUNE
case ${num} in
   * )
		lr=60.0
		lincls_pretrained=${log_dir}/ckpt.pth.tar
		# lincls_pretrained=${ROOT}/output/slimclr-mocov2-ep200-48213217.pth.tar
		# lincls_pretrained=${ROOT}/output/slimclr-mocov2-ep800-eda810a6a9.pth.tar
		# lincls_resume=${ROOT}/output/slimclr-mocov2-ep200-lincls-789b1a3b17.pth.tar
		# lincls_resume=${ROOT}/output/slimclr-mocov2-ep800-lincls-35600f623f.pth.tar
		;;
esac

lincls_log_dir=${ROOT}/output/${ssl_arch}_slim${slimmable_training}_lincls_${dataset}_${num}
echo "The model dir is ${lincls_log_dir}"


python ${runfile_ssl_lincls} --dataset_dir ${dataset_dir} \
						--lmdb_dataset ${lmdb_dataset} \
						--lr ${lr} \
						--optimizer 'SGD' \
						--weight_decay "0.0" \
						--lr_mode 'step' \
						--num_epochs "100" \
						--momentum ${momentum} \
						--warmup_proportion "0.0" \
						--use_bn_sync ${use_bn_sync} \
						--log_dir ${lincls_log_dir} \
						--batch_size_per_gpu ${batch_size_per_gpu} \
						--precision "fp32" \
						--test_only ${test_only} \
						--num_workers ${num_workers} \
						--slimmable_training ${slimmable_training} \
						--overlap_weight ${overlap_weight} \
						--slim_fc 'supervised_switch' \
						--inplace_distill 1 \
						--inplace_distill_mixed 1 \
						--width_mult_list ${width_mult_list} \
						--resume ${lincls_resume} \
						--lincls_pretrained ${lincls_pretrained} \
						--ssl_aug "normal"

done

echo "Training Finished!!!"