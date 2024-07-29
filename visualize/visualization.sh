#!/bin/bash
export CUDA_VISIBLE_DEVICES='0'
### Remember to change ROOT to /YOUR/PATH ###
ROOT=/home/shuzhao/Data
#############################################


# Commands to plot Figure (5) in the paper


# 1. First, train a model and record the optimization information

## 1.1 Supversied
# exp 06
python -m slimssl.visualize.cifar --width_mult_list 1.0 0.5 --loss_w 1.0 1.0 --depth 20 --width 4 \
			--log_dir ${ROOT}/output/cifar_06
# INFO: [meter] Test set: Width: 1.0, Average loss: 0.2715, Accuracy: 9358.0/10000 (93.58%)
# INFO: [meter] Test set: Width: 0.5, Average loss: 0.3111, Accuracy: 9303.0/10000 (93.03%)

# exp 07
python -m slimssl.visualize.cifar --width_mult_list 1.0 0.5 --loss_w 1.0 0.0 --depth 20 --width 4 \
			--log_dir ${ROOT}/output/cifar_07
# INFO: [meter] Test set: Width: 1.0, Average loss: 0.2436, Accuracy: 9407.0/10000 (94.07%)
# INFO: [meter] Test set: Width: 0.5, Average loss: 2.4223, Accuracy: 1594.0/10000 (15.94%)


## 1.2 Self-Supervised
# exp 12 - resnet20x4
python -m slimssl.visualize.cifar_moco --width_mult_list 1.0 0.5 --loss_w 1.0 0.0 \
		--log_dir ${ROOT}/output/moco_cifar_12 --depth 20 --width 4 --epochs 100
# Test Epoch: [100/100] Width: 1.0, Acc@1:76.75%
# Test Epoch: [100/100] Width: 0.5, Acc@1:49.08%

# exp 13 - resnet20x4
python -m slimssl.visualize.cifar_moco --width_mult_list 1.0 0.5 --loss_w 1.0 1.0 \
		--log_dir ${ROOT}/output/moco_cifar_13 --depth 20 --width 4 --epochs 100
# Test Epoch: [100/100] Width: 1.0, Acc@1:75.74%
# Test Epoch: [100/100] Width: 0.5, Acc@1:74.24%


# 2. Visuliazation

## 2.1 Supversied
num=06
python -m slimssl.visualize.plot_cifar --x=-80:80:51 --y=-30:30:51 --num ${num} \
		--output ${ROOT}/output/vis \
		--dir_file ${ROOT}/output/cifar_06/pca_direction.pt \
		--model_file ${ROOT}/output/cifar_06/resnet20_cifar_final.pt \
		--traj_file ${ROOT}/output/cifar_06/cifar_traj_0.5.npy \
		--traj_file1 /none/x \
		--vlevel 5 --vmin 0 --vmax 100 \
		--use_train 0 \
		--width_mult_list 1.0 0.5 \
		--depth 20 \
		--width 4

num=07
folder=cifar_07
python -m slimssl.visualize.plot_cifar --x=-80:80:51 --y=-30:30:51 --num ${num} \
		--output ${ROOT}/output/vis \
		--dir_file ${ROOT}/output/${folder}/pca_direction.pt \
		--model_file ${ROOT}/output/${folder}/resnet20_cifar_final.pt \
		--traj_file ${ROOT}/output/${folder}/cifar_traj_0.5.npy \
		--traj_file1 /none/x \
		--vlevel 5 --vmin 0 --vmax 100 \
		--use_train 0 \
		--width_mult_list 1.0 0.5 \
		--depth 20 \
		--width 4


## 2.2 Self-Supervised
num=12
python -m slimssl.visualize.plot_cifar_moco --x=-80:80:51 --y=-20:20:51 --num ${num} \
		--output ${ROOT}/output/vis \
		--dir_file ${ROOT}/output/moco_cifar_12/pca_direction.pt \
		--model_file ${ROOT}/output/moco_cifar_12/resnet20_cifar_final.pt \
		--traj_file ${ROOT}/output/moco_cifar_12/cifar_traj_0.5.npy \
		--traj_file1 /none/x \
		--vlevel 5 --vmin 0 --vmax 100 \
		--width_mult_list 1.0 0.5 \
		--depth 20 \
		--width 4

num=13
python -m slimssl.visualize.plot_cifar_moco --x=-80:80:51 --y=-20:20:51 --num ${num} \
		--output ${ROOT}/output/vis \
		--dir_file ${ROOT}/output/moco_cifar_13/pca_direction.pt \
		--model_file ${ROOT}/output/moco_cifar_13/resnet20_cifar_final.pt \
		--traj_file ${ROOT}/output/moco_cifar_13/cifar_traj_0.5.npy \
		--traj_file1 /none/x \
		--vlevel 5 --vmin 0 --vmax 100 \
		--width_mult_list 1.0 0.5 \
		--depth 20 \
		--width 4