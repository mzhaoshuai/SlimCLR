#coding=utf-8

import os
import json
import logging
import argparse


def get_args(description='Configs of SlimCLR'):
	"""config of program"""
	parser = argparse.ArgumentParser(description=description)

	parser.add_argument("--test_only", type=int, default=0, help="Whether to run eval on the dev set.")
	parser.add_argument("--inference_speed_test", type=int, default=0, help="Only test the inference speed.")
	parser.add_argument("--debug", default=False, action="store_true", help="If true, more information is logged.")
	# model
	parser.add_argument("--custom_resnet", type=int, default=0, help="use custom resnet, which return a dict.")
	parser.add_argument("--slimmable_training", type=int, default=0, help="apply slimmable training strategy.")
	parser.add_argument("--overlap_weight", type=int, default=1, help="sharing weights of subnetworks are overlapped or not.")
	parser.add_argument("--label_smoothing", type=int, default=0, help="label smoothing.")
	parser.add_argument("--slim_loss", type=str, default='sum', choices=['sum', 'mean'],
							help="how to aggeragate the loss of slimmable networks.")
	parser.add_argument("--inplace_distill", type=int, default=0, help="apply distillationb between large and subnetworks.")
	parser.add_argument("--inplace_distill_mixed", type=int, default=0,
							help="using both distillation loss and supervision loss.")
	parser.add_argument("--model", type=str, default='models.backbones.s_resnet', help="model architecture.")
	parser.add_argument("--depth", type=int, default=50, help="model depth.")
	parser.add_argument("--width_mult", type=float, default=1.0, help="width multiplier for slimmable model.")
	parser.add_argument('--width_mult_list', type=float, default=[1.0, 0.5, 0.25], nargs='+', help="all width multiplier for slimmable model.")
	# datasets
	parser.add_argument("--dataset", default="imagenet1k", type=str, choices=['imagenet1k'], help="the training / testing dataset.")
	parser.add_argument('--dataset_dir', type=str, default='imagenet', help='where dataset located')
	parser.add_argument("--data_transforms", default="imagenet1k_basic", type=str, choices=['imagenet1k_basic'],
							help="the training / testing data tranforms.")
	parser.add_argument("--data_loader", default="imagenet1k_basic", type=str, choices=['imagenet1k_basic'],
							help="the training / testing data loader.")
	parser.add_argument('--num_workers', type=int, default=8, help='workers for pytorch dataloader')
	parser.add_argument('--lmdb_dataset', type=str, default=None, help="LMDB database for the dataset")
	parser.add_argument('--dali_data_dir', type=str, default='/not_exist/x', help="data dir for NVIDIA DALI")
	parser.add_argument('--albumentations', type=int, default=0, help="whether use albumentations for data augmentation")
	parser.add_argument('--dali_cpu', type=int, default=1, help="augmentation on cpu with dali or on gpu")
	parser.add_argument('--num_classes', default=1000, type=int, metavar='N', help='The number of classes.')

	# training settings
	parser.add_argument('--num_epochs', type=int, default=100, help='upper epoch limit')
	parser.add_argument('--batch_size_per_gpu', type=int, default=128, help='batch size per gpu')
	parser.add_argument('--batch_size_val', type=int, default=256, help='batch size eval')
	parser.add_argument('--image_size', default=224, type=int, metavar='N', help='crop size (default: 224)')	
	parser.add_argument("--log_dir", type=str, default='output01', help="The output directory where the model predictions and checkpoints will be written.")

	# learning strategies
	parser.add_argument('--optimizer', type=str, default='SGD',
							choices=['SGD', 'AdamW', 'RMSprop', 'RMSpropTF', 'LARS'], help='The optimizer.')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='optimizer momentum (default: 0.9)')
	parser.add_argument('--nesterov', default=0, type=int, help='using Nesterov accelerated gradient descent or not')
	parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
	parser.add_argument('--slim_start_lr', type=float, default=0.0001, help='learning rate for slimmable training')
	parser.add_argument('--lr_mode', type=str, choices=['cos', 'step', 'poly', 'HTD', 'exponential'], default='cos',)
	parser.add_argument('--lr_milestones', nargs='+', type=int, default=[60, 80], help='epochs at which we take a learning-rate step (default: [60, 80])')
	parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
	parser.add_argument('--coef_lr', type=float, default=1., help='coefficient for bert branch.')
	parser.add_argument("--beta1", type=float, default=0.9, help="Adam beta 1.")
	parser.add_argument("--beta2", type=float, default=0.98, help="Adam beta 2.")
	parser.add_argument("--eps", type=float, default=1e-6, help="Adam epsilon.")
	parser.add_argument("--weight_decay", type=float, default=1e-4, help="Weight decay.")
	parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')

	# https://arxiv.org/abs/2109.08203, Torch.manual_seed(3407) is all you need
	parser.add_argument('--seed', type=int, default=3407, help='random seed')
	parser.add_argument("--resume", default=None, type=str, help="path to latest checkpoint (default: none)",)
	parser.add_argument('--load_from_pretrained', type=int, default=0, help="load from pretrained model (scaler state, optimizer, epoch, etc).")
	parser.add_argument("--warmup_proportion", default=0.0, type=float,
							help="Proportion of training to perform linear learning rate warmup"
									" for. E.g., 0.1 = 10%% of training.")
	parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
							help="Number of updates steps to accumulate before performing a backward/update pass.")
	parser.add_argument("--clip_grad_norm", default=None, type=float, help="the maximum gradient norm (default None)")
	parser.add_argument("--pretrained_dir", default=os.path.expanduser("~/models/pretrained"), type=str,
							help="The pretrained directory of pretrained model")

	# arguments for distributed training
	parser.add_argument("--dist_backend", default="nccl", type=str, help="distributed backend")
	parser.add_argument('--world_size', default=1, type=int,
							help='number of nodes for distributed training')
	parser.add_argument("--local_rank", default=0, type=int, help="distribted training")
	parser.add_argument("--init_method", default="tcp://127.0.0.1:6101", type=str,
	   						help="url used to set up distributed training")
	parser.add_argument('--use_bn_sync', default=0, type=int,
							help='whether use sync bn for distributed training')

	# setting about GPUs
	parser.add_argument("--dp", default=False, action="store_true",
							help="Use DP instead of DDP.")
	parser.add_argument("--multigpu", default=None,
							type=lambda x: [int(a) for a in x.split(",")],
							help="In DP, which GPUs to use for multigpu training", )
	parser.add_argument("--gpu", type=int, default=None,
							help="Specify a single GPU to run the code on for debugging."
							 "Leave at None to use all available GPUs.", )
	parser.add_argument('--n_gpu', type=int, default=1,
							help="Changed in the execute process.")
	# precision of training weights
	parser.add_argument("--precision", choices=["amp", "fp16", "fp32"], default="fp32",
							help="Floating point precition.")
	# moco specific configs:
	parser.add_argument('--arch', metavar='ARCH', default='resnet50',
							choices=['resnet50', 'resnet101'],
							help='model architecture (default: resnet50)')
	# options for moco
	parser.add_argument('--moco_dim', default=128, type=int, help='feature dimension (default: 128)')
	parser.add_argument('--moco_k', default=65536, type=int, help='queue size; number of negative keys (default: 65536)')
	parser.add_argument('--moco_m', default=0.999, type=float, help='moco momentum of updating key encoder (default: 0.999)')
	parser.add_argument('--moco_t', default=0.07, type=float, help='softmax temperature (default: 0.07)')
	parser.add_argument('--mlp', default=1, type=int, help='use mlp head')
	# options for mocov3
	parser.add_argument('--moco_m_cos', default=1, type=int, help='gradually increase moco momentum to 1 with a half-cycle cosine schedule')
	parser.add_argument('--moco_mlp_dim', default=4096, type=int, help='hidden dimension in MLPs (default: 4096)')
	parser.add_argument('--crop-min', default=0.2, type=float, help='minimum scale for random cropping (default: 0.2)')
	# options for simclr
	parser.add_argument('--simclr_mlp_dim', default=2048, type=int, help='hidden dimension in MLPs of simclr (default: 2048)')
	# linear classification configs
	parser.add_argument('--lincls_pretrained', default='', type=str, help='path to moco pretrained checkpoint')
	parser.add_argument('--eval_freq', type=int, default=2, help='evaluation frequence')							
	# ssl specific configs:
	parser.add_argument('--ssl_arch', default='mocov2', type=str, choices=['mocov3', 'mocov2', 'simclr', 'none'],
							help='ssl model architecture (default: mocov2)')
	parser.add_argument('--slim_fc', default='supervised', type=str,
							choices=['mocov2_slim', 'mocov2_switch', 'mocov3_slim', 'supervised', 'supervised_switch', 'simclr_slim'],
							help='fc choice for slim_fc')
	parser.add_argument('--ssl_aug', default='normal', type=str,
							choices=['mocov2', 'mocov3', 'simclr', 'normal', 'val', 'slim', 'mocov2_sweet', 'mocov2_diff'],
							help='data augmentations for ssl')
	# loss for distillation
	parser.add_argument('--seed_loss', default=0, type=int,
							help='Apply SEED loss between large networks and subnetworks, a.k.a, KL Div')
	parser.add_argument('--mse_loss', default=0, type=int,
							help='use mse loss as regulariation for distillation')
	parser.add_argument('--atkd_loss', default=0, type=int,
							help='apply adaptive temperature knowledge distillation')
	parser.add_argument('--dkd_loss', default=0, type=int,
							help='apply adaptive decoupled knowledge distillation')
	parser.add_argument("--teacher_T", type=float, default=1.0,
							help="temperature of distillation teacher.")
	parser.add_argument("--student_T", type=float, default=1.0,
							help="temperature of distillation for student.")
	parser.add_argument('--aux_loss_w', default=0.5, type=float,
							help='weight of auxiliary loss in slim+moco')
	parser.add_argument('--full_distill', default=0, type=int,
							help='subnetworks will learn from all networks which are bigger than it')
	parser.add_argument('--slim_start_epoch', default=-1, type=int,
							help='start feed data to subnetworks in slimmable networks from which epoch')
	parser.add_argument('--slim_loss_weighted', default=0, type=int,
							help='weight loss by the scale of the networks')
	parser.add_argument('--is_log_grad', default=0, type=int,
							help='weight loss by the scale of the networks')
	parser.add_argument("--grad_log", type=str,
							default='grad.txt', 
							help="where to log the gradient info.")
	parser.add_argument('--width_multipier', type=float, default=1.0,
							help='width multiplier for VGG')

	args = parser.parse_args()
	args.resume = None if (args.resume is None or args.resume in ["None", "none"]) else args.resume

	# settings about training epochs and learning rate
	# if args.num_epochs in [100, 120]:
	# 	args.lr_milestones = [30, 60, 90]
	args.width_mult_list = sorted(args.width_mult_list, reverse=True)
	if args.slimmable_training:
		args.inplace_distill = args.inplace_distill and len(args.width_mult_list) > 1
		args.seed_loss = args.seed_loss and len(args.width_mult_list) > 1
		args.mse_loss = args.mse_loss if len(args.width_mult_list) > 1 else 0
		args.atkd_loss = args.atkd_loss if len(args.width_mult_list) > 1 else 0
		args.dkd_loss = args.dkd_loss if len(args.width_mult_list) > 1 else 0
		args.slim_loss_weighted = args.slim_loss_weighted if len(args.width_mult_list) > 1 else 0
		assert not (args.inplace_distill and args.seed_loss and args.atkd_loss and args.mse_loss)
		if not (args.inplace_distill or args.seed_loss or args.mse_loss or args.atkd_loss or args.dkd_loss):
			args.aux_loss_w = 0.0

	# 'mocov2_slim' produces multiple images
	if args.ssl_aug == 'mocov2_slim':
		assert len(args.width_mult_list) > 1 and args.slimmable_training

	# Check paramenters
	if args.gradient_accumulation_steps < 1:
		raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
			args.gradient_accumulation_steps))

	if not os.path.exists(args.log_dir):
		os.makedirs(args.log_dir, exist_ok=True)

	# automatically dividing with gradient_accumulation_steps
	args.batch_size_per_gpu = int(args.batch_size_per_gpu / args.gradient_accumulation_steps)

	args.tensorboard_path = os.path.join(args.log_dir, "tensorboard")
	# logging level
	args.log_level = logging.DEBUG if args.debug else logging.INFO

	print('\n', dict(sorted(vars(args).items(), key=lambda x: x[0])), )
	# save_hp_to_json(args.log_dir, args)

	return args


def save_hp_to_json(directory, args):
	"""Save hyperparameters to a json file
	"""
	filename = os.path.join(directory, 'hparams_train.json')
	hparams = vars(args)
	with open(filename, 'w') as f:
		json.dump(hparams, f, indent=4, sort_keys=True)


if __name__ == "__main__":
	args = get_args()
	# print(args)
