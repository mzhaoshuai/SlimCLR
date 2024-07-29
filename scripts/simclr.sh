#!/bin/bash
export PYTHONUNBUFFERED="True"

### Remember to change ROOT to /YOUR/PATH ###
ROOT=/home/shuzhao/Data
#############################################
### simply set test_only=1 for inference, comment the SSL pre-trained stage or use a pre-trained checkpoint
test_only=0

# dataset and path
dataset=imagenet
dataset_dir=${ROOT}/dataset/imagenet
lmdb_dataset=None

# learning strategy
lr=0.1
lr_mode=cos
optimizer=LARS
weight_decay=1e-6
momentum=0.9
num_epochs=100
warmup_proportion=0.1
batch_size_per_gpu=128
gradient_accumulation_steps=1
precision=amp
use_bn_sync=1
overlap_weight=1

# dataloader
num_workers=8
ssl_arch=simclr
ssl_aug=simclr

# slimmable networks
slimmable_training=0
width_mult_list="1.0 0.5"
slim_start_epoch=-1
slim_sub_no_clr=0
slim_fc=simclr_slim
full_distill=0
seed_loss=1
mse_loss=0
inplace_distill=0
is_log_grad=0
slim_loss_weighted=5
aux_loss_w=0.5
teacher_T=5.0
student_T=5.0

# resume and log
resume=None

# run file path
runfile_ssl=${ROOT}/code/SlimCLR/main_ssl.py
runfile_ssl_lincls=${ROOT}/code/SlimCLR/main_ssl_lincls.py

for num in 02
do
	case ${num} in
	01 )
		batch_size_per_gpu=128
		gradient_accumulation_steps=1
		# square learning rate 0.075 * (sqrt(BS))
		# We use a batch size of 128 x 8 = 1024
		lr=2.4
		;;
	02 )	
		# SimCLR with slimmable networks
		slimmable_training=1
		batch_size_per_gpu=128
		gradient_accumulation_steps=1
		# square learning rate 0.075 * (sqrt(BS))
		# We use a batch size of 128 x 8 = 1024
		lr=2.4
		width_mult_list="1.0 0.75 0.5 0.25"
		slim_start_epoch=50
		seed_loss=1
		mse_loss=0
		inplace_distill=0
		slim_loss_weighted=5
		aux_loss_w=0.5
		teacher_T=5.0
		student_T=5.0
		resume=${ROOT}/output/simclr_slim1_imagenet_02/ckpt.pth.tar
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
					--num_epochs ${num_epochs} \
					--lr_mode ${lr_mode} \
					--momentum ${momentum} \
					--warmup_proportion ${warmup_proportion} \
					--use_bn_sync ${use_bn_sync} \
					--log_dir ${log_dir} \
					--batch_size_per_gpu ${batch_size_per_gpu} \
					--gradient_accumulation_steps ${gradient_accumulation_steps} \
					--precision ${precision} \
					--test_only ${test_only} \
					--num_workers ${num_workers} \
					--slimmable_training ${slimmable_training} \
					--width_mult_list ${width_mult_list} \
					--inplace_distill ${inplace_distill} \
					--slim_start_epoch ${slim_start_epoch} \
					--resume ${resume} \
					--ssl_aug ${ssl_aug} \
					--slim_fc ${slim_fc} \
					--grad_log ${grad_log} \
					--is_log_grad ${is_log_grad} \
					--seed_loss ${seed_loss} \
					--mse_loss ${mse_loss} \
					--weight_decay ${weight_decay} \
					--optimizer ${optimizer} \
					--slim_loss_weighted ${slim_loss_weighted} \
					--teacher_T ${teacher_T} \
					--student_T ${student_T} \
                    --ssl_arch ${ssl_arch}

	# STEP 2. FINETUNE
	case ${num} in
	* )
		lr=0.4
		lincls_pretrained=${log_dir}/ckpt.pth.tar
		resume=None
		;;
	esac


lincls_log_dir=${ROOT}/output/${ssl_arch}_slim${slimmable_training}_lincls_${dataset}_${num}
echo "The model dir is ${lincls_log_dir}"

python ${runfile_ssl_lincls} \
						--dataset_dir ${dataset_dir} \
						--lmdb_dataset ${lmdb_dataset} \
						--lr ${lr} \
						--optimizer 'SGD' \
						--weight_decay "1e-6" \
						--lr_mode "cos" \
						--num_epochs "90" \
						--momentum ${momentum} \
						--warmup_proportion "0.0" \
						--use_bn_sync "0" \
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
						--resume ${resume} \
						--lincls_pretrained ${lincls_pretrained} \
						--ssl_aug 'normal' \
                        --ssl_arch ${ssl_arch}

done

echo "Training Finished!!!"