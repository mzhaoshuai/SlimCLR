# coding=utf-8
import os
import sys
import time
import torch
import logging
import warnings
import importlib
import numpy as np
import torch.nn.functional as F
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from models.utils import get_grad_norm, save_last_fc_grad
from utils.lr_scheduler import lr_scheduler
from params import get_args, save_hp_to_json
from utils.meters import accuracy, AverageMeter
from utils.optim import get_parameter_groups, LARS
from utils.log import setup_primary_logging, setup_worker_logging
from utils.loss_ops import label_smoothing_CE, CrossEntropyLossSoft
from utils.misc import save_checkpoint, mkdir
from utils.distributed import (init_distributed_mode, get_rank, is_dist_avail_and_initialized,
								is_master, set_random_seed)

# ignore some Pillow userwarning
warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

best_acc1 = 0


def main(args):
	"""main function"""
	set_random_seed(args.seed)
	mkdir(args.log_dir)
	# Set multiprocessing type to spawn.
	torch.multiprocessing.set_start_method('spawn')
	# Set logger
	log_queue = setup_primary_logging(os.path.join(args.log_dir, "log.txt"), logging.INFO)
	# the number of gpus
	args.ngpus_per_node = torch.cuda.device_count()
	print("INFO: [CUDA] The number of GPUs in this node is {}".format(args.ngpus_per_node))

	# Distributed training = training on more than one GPU.
	# Also easily possible to extend to multiple nodes & multiple GPUs.
	args.distributed = (args.gpu is None) and torch.cuda.is_available() and (not args.dp)
	if args.distributed:
		# Since we have ngpus_per_node processes per node, the total world_size
		# needs to be adjusted accordingly
		args.world_size = args.ngpus_per_node * args.world_size
		mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args.ngpus_per_node, log_queue, args))
	else:
		# nn.DataParallel (DP)
		if args.dp:
			args.gpu, args.world_size = args.multigpu[0], len(args.multigpu)
		else:
			args.world_size = 1
		main_worker(args.gpu, None, log_queue, args)


def main_worker(gpu, ngpus_per_node, log_queue, args):
	"""main worker"""
	from dataset.dataloader import get_dataloader
	global best_acc1
	args.gpu = gpu

	## ####################################
	# distributed training initilization
	## ####################################
	global_rank = init_distributed_mode(args, ngpus_per_node, gpu)
	setup_worker_logging(global_rank, log_queue, logging.INFO)
	# Lock the random seed of the model to ensure that the model initialization of each process is the same.
	set_random_seed(args.seed)
	# torch.backends.cudnn.benchmark = False
	# torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = True

	# save parameters
	if is_master(): save_hp_to_json(args.log_dir, args)	

	## ####################################
	# create model
	## ####################################
	model_lib = importlib.import_module(args.model)
	model = model_lib.Model(args.num_classes, input_size=args.image_size, args=args)

	if not torch.cuda.is_available():
		model.float()
		logging.warning("using CPU, this will be slow")
	else:
		model.cuda(args.gpu)
		# Previously batch size and workers were global and not per GPU.
		# args.batch_size = args.batch_size / ngpus_per_node)
		# args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
		if args.distributed and args.use_bn_sync:
			logging.info('[CUDA] Using SyncBatchNorm...')
			model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
		if args.distributed:
			model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
																find_unused_parameters=False)
		if args.dp:
			model = torch.nn.DataParallel(model, device_ids=args.multigpu)

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu", get_rank())

	## ####################################
	# dataloader loading
	## ####################################
	train_loader, val_loader, test_loader = get_dataloader(args)
	num_train_optimization_steps = (int(len(train_loader) + args.gradient_accumulation_steps - 1)
									/ args.gradient_accumulation_steps) * args.num_epochs


	## ####################################
	# optimization strategies
	## ####################################
	if getattr(args, 'label_smoothing', 0):
		criterion = label_smoothing_CE().cuda(args.gpu)
	else:
		criterion = torch.nn.CrossEntropyLoss().cuda(args.gpu)
	
	if getattr(args, 'inplace_distill', False):
		soft_criterion = CrossEntropyLossSoft(reduction='batchmean').cuda(args.gpu)
	else:
		soft_criterion = None

	grouped_parameters = get_parameter_groups(model, args.lr, args.weight_decay, norm_weight_decay=0.0)
	scaler = GradScaler() if args.precision == "amp" else None
	logging.info('[optimizer] Using {} Optimizer...'.format(args.optimizer))
	if args.optimizer == 'SGD':
		optimizer = torch.optim.SGD(grouped_parameters,
									args.lr,
									momentum=args.momentum,
									weight_decay=args.weight_decay,
									nesterov=True if args.nesterov else False
									)
	elif args.optimizer == 'LARS':
		optimizer = LARS(grouped_parameters,
							args.lr,
							momentum=args.momentum,
							weight_decay=args.weight_decay)
	else:
		raise NotImplementedError

	scheduler = lr_scheduler(mode=args.lr_mode,
								init_lr=args.lr, all_iters=num_train_optimization_steps,
								slow_start_iters=args.warmup_proportion * num_train_optimization_steps,
								weight_decay=args.weight_decay,
								lr_milestones=args.lr_milestones
							)

	## ####################################
	#  optionally resume from a checkpoint
	## ####################################
	start_epoch, global_step = 0, 0
	if args.resume is not None:
		if os.path.isfile(args.resume):
			if args.gpu is None:
				checkpoint = torch.load(args.resume)
			else:
				# Map model to be loaded to specified single gpu.
				loc = "cuda:{}".format(args.gpu)
				checkpoint = torch.load(args.resume, map_location=loc)
			
			sd = checkpoint["state_dict"]
			if not args.distributed and next(iter(sd.items()))[0].startswith('module'):
				sd = {k[len('module.'):]: v for k, v in sd.items()}
			model.load_state_dict(sd)

			if not args.load_from_pretrained:
				if "optimizer" in checkpoint and optimizer is not None:
					optimizer.load_state_dict(checkpoint["optimizer"])
				if "scaler" in checkpoint and scaler is not None:
					logging.info("[resume] => Loading state_dict of AMP loss scaler")
					scaler.load_state_dict(checkpoint['scaler'])
				start_epoch, global_step = checkpoint["epoch"], checkpoint["global_step"]

			logging.info(f"[resume] => loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})\n")
		else:
			logging.info("[resume] => no checkpoint found at '{}'\n".format(args.resume))	
	# create tensorboard logger
	tf_writer = SummaryWriter(args.tensorboard_path) if is_master() else None

	## ####################################
	# train and evalution
	## ####################################
	all_start = time.time()

	if args.test_only and is_master():
		final_acc1_all, final_acc5_all, info_tmp, infer_epoch_time = eval_epoch(test_loader, model, criterion, device, args=args)
		if torch.cuda.is_available(): torch.cuda.synchronize()
		all_time = time.time() - all_start
		logging.info('The total running time of the program is {:.2f} Seconds\n'.format(all_time))
		logging.info('The maximum GPU memory occupied by this program is {:.2f} GB\n'.format(
						torch.cuda.max_memory_allocated(0) * 1.0 / 1024 / 1024 / 1024))
		sys.exit(0)

	eval_infer_times = []
	best_e = 0
	best_info = []
	for epoch in range(start_epoch, args.num_epochs):
		if is_dist_avail_and_initialized() and not args.is_dali:
			train_loader.sampler.set_epoch(epoch)

		# with torch.autograd.set_detect_anomaly(True):
		# set_random_seed(epoch + args.seed)
		global_step = train_epoch(epoch, train_loader, model, criterion, optimizer, global_step, device,
						soft_criterion=soft_criterion, scheduler=scheduler, scaler=scaler, tf_writer=tf_writer, args=args)

		if is_master():
			logging.info("Epoch %d/%s Finished.", epoch + 1, args.num_epochs)

			# run on val dataset
			final_acc1_all, final_acc5_all, info_tmp, infer_epoch_time = \
							eval_epoch(val_loader, model, criterion, device, args=args)
			eval_infer_times.append(infer_epoch_time)
			if best_acc1 <= final_acc1_all[0]:
				best_acc1 = final_acc1_all[0]
				best_e = epoch
				best_info = info_tmp
			logging.info("The best Top-1/top-5 Acc is: {:.2f}/{:.2f}, best_e={}\n".format(
							best_acc1, final_acc5_all[0], best_e))
			# save checkpoint
			ckpt_dict = {
					'epoch': epoch + 1,
					'global_step': global_step,
					'arch': 's_resnet',
					'state_dict': model.state_dict(),
					'best_acc1': best_acc1,
					'optimizer': optimizer.state_dict(),
				}
			if scaler is not None: ckpt_dict['scaler'] = scaler.state_dict()
			save_checkpoint(ckpt_dict, best_acc1 <= final_acc1_all[0], args.log_dir, filename='ckpt.pth.tar')

			if args.is_dali: val_loader.reset()	

		# reset dali dataloader for each epoch
		if args.is_dali: train_loader.reset()		

	if is_master():
		if torch.cuda.is_available(): torch.cuda.synchronize()
		all_time = time.time() - all_start
		logging.info('The total running time of the program is {:.1f} Hour {:.1f} Minute\n'.format(all_time // 3600, 
					all_time % 3600 / 60))

		logging.info('The average inference time of {} runs is {:.2f} Seconds\n'.format(
						args.num_epochs, np.mean(eval_infer_times)))
		logging.info('The maximum GPU memory occupied by this program is {:.2f} GB\n'.format(
					torch.cuda.max_memory_allocated(0) * 1.0 / 1024 / 1024 / 1024))
		logging.info("The best R1 is: {:.4f}, best_epoch={}\n".format(best_acc1, best_e))
		logging.info(best_info)
		print("The above program id is {}\n".format(args.log_dir))

	torch.cuda.empty_cache()
	sys.exit(0)


def train_epoch(epoch, loader, model, criterion, optimizer, global_step, device,
			soft_criterion=None, scheduler=None, scaler=None, tf_writer=None, args=None):
	"""training for one epoch"""
	samples_per_epoch = len(loader.dataset) if not args.is_dali else \
							len(loader) * args.batch_size_per_gpu * args.world_size
	sorted_width_mult_list = sorted(args.width_mult_list, reverse=True)
	max_width, min_width = sorted_width_mult_list[0], sorted_width_mult_list[-1]
	num_w = len(sorted_width_mult_list)

	# record the top1 accuray of all networks
	m_top1_all, m_ce_all = [], []
	for w in sorted_width_mult_list:
		m_ce_all.append(AverageMeter('w{}_ce'.format(w), ':.4e'))
		m_top1_all.append(AverageMeter('w{}_acc@1'.format(w), ':6.2f'))

	# switch to train mode
	model.train()
	end = time.time()
	train_start_t = time.time()

	for step, data in enumerate(loader):
		image = data[0] if not args.is_dali else data[0]['data']
		target = data[1] if not args.is_dali else data[0]['label'].squeeze(-1).long()
		optimizer.zero_grad()
		if scheduler is not None: scheduler(optimizer, epoch=epoch, global_step=global_step)
		image = image.to(device=device, non_blocking=True)
		target = target.to(device=device, non_blocking=True)
		data_time = time.time() - end

		########### one forward produce all outputs, then one backward ###########
		all_losses, all_output = [], []
		with torch.cuda.amp.autocast(enabled = scaler is not None):
			if getattr(args, 'slimmable_training', False):
				dict_out = model(image)
				# NOTICE: detach() is necessary
				soft_target = F.softmax(dict_out[max_width], dim=1).detach()
				# slimmable model (s-nets)
				for width_mult in sorted_width_mult_list:
					output = dict_out[width_mult]
					if width_mult == max_width:
						loss = criterion(output, target)
					else:
						loss = soft_criterion(output, soft_target).mean() if getattr(args, 'inplace_distill', False) \
								else criterion(output, target).mean()
					all_losses.append(loss)
					all_output.append(output)
			else:
				dict_out = model(image, max_width)
				loss = criterion(dict_out[max_width], target).mean()
				all_losses.append(loss)
				all_output.append(dict_out[max_width])

		loss = torch.sum(torch.stack(all_losses)) if args.slim_loss == 'sum' \
					else torch.mean(torch.stack(all_losses))
		if scaler is not None:
			# https://pytorch.org/docs/stable/notes/amp_examples.html#working-with-multiple-models-losses-and-optimizers
			# for i, w in enumerate(sorted_width_mult_list):
				# scaler.scale(all_losses[i]).backward(retain_graph=True if i != num_w - 1 else None)
			scaler.scale(loss).backward()

			if args.clip_grad_norm is not None:
				# we should unscale the gradients of optimizer's assigned params if do gradient clipping
				scaler.unscale_(optimizer)
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
			scaler.step(optimizer)
			scaler.update()
		else:
			loss = torch.sum(torch.stack(all_losses))
			loss.backward()
			if args.clip_grad_norm is not None:
				torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
			optimizer.step()

		# measure elapsed time
		batch_time = time.time() - end
		end = time.time()

		# step 3. benckmark and print info
		batch_size_now = image.size(0)
		acc_str, loss_str = '', ''
		for i in range(len(sorted_width_mult_list)):
			acc1 = accuracy(all_output[i], target, topk=(1, ))
			m_top1_all[i].update(acc1[0].item(), batch_size_now)
			m_ce_all[i].update(all_losses[i].mean().item(), batch_size_now)

			acc_str += '{:.2f}, '.format(m_top1_all[i].avg)
			loss_str += '{:.3f}, '.format(m_ce_all[i].avg)

		global_step += 1
		if global_step % args.n_display == 0 and is_master():
			if args.is_log_grad:
				# get_grad_norm(model, sorted_width_mult_list, global_step, tf_writer=tf_writer, args=args)
				# get_grad_norm(model, [1.0, 0.5, 0.25], global_step, tf_writer=tf_writer, args=args)
				save_last_fc_grad(model, global_step, tf_writer=tf_writer, args=args)

			num_samples = (step + 1) * batch_size_now * args.world_size
			percent_complete = num_samples * 1.0 / samples_per_epoch * 100
			lr_tmp = optimizer.param_groups[0]['lr']
			info_tmp =	(f"Epoch: {epoch} [({percent_complete:.1f}%)] "
						f"Loss: [{loss_str}] Acc@1: [{acc_str}] "
						f"Data (t) {data_time:.3f} Batch (t) {batch_time:.2f} "
						f"LR: {lr_tmp:.1e}".replace(', ]', ']'))

			logging.info(info_tmp)

	if torch.cuda.is_available(): torch.cuda.synchronize()
	if is_master():
		one_epoch_time = time.time() - train_start_t
		logging.info('The total number of training samples for this device is {}'.format(m_top1_all[0].count))
		logging.info('The total model train time for one epoch is is {:.2f} Seconds\n'.format(one_epoch_time))
		logging.info('The throughout is {:.2f} images per second\n'.format(samples_per_epoch / one_epoch_time))
		
	return global_step


def eval_epoch(loader, model, criterion, device, scaler=None, args=None):
	"""validate for one epoch"""
	sorted_width_mult_list = sorted(args.width_mult_list, reverse=True)
	max_width, min_width = sorted_width_mult_list[0], sorted_width_mult_list[-1]

	# record the top1 accuray of all networks
	m_top1_all, m_top5_all, m_ce_all = [], [], []
	for w in sorted_width_mult_list:
		m_ce_all.append(AverageMeter('w{}_ce'.format(w), ':.4e'))
		m_top1_all.append(AverageMeter('w{}_acc@1'.format(w), ':6.2f'))
		m_top5_all.append(AverageMeter('w{}_acc@1'.format(w), ':6.2f'))

	# switch to evaluate mode
	model.eval()
	infer_start_t = time.time()

	with torch.no_grad():
		for step, data in enumerate(loader):
			image = data[0] if not args.is_dali else data[0]['data']
			target = data[1] if not args.is_dali else data[0]['label'].squeeze(-1).long()
			image = image.to(device=device, non_blocking=True)
			target = target.to(device=device, non_blocking=True)

			########### one forward produce all outputs, no backward ###########
			all_losses, all_output = [], []
			with torch.cuda.amp.autocast(enabled = scaler is not None):
				if getattr(args, 'slimmable_training', False):
					dict_out = model(image)
					# slimmable model (s-nets)
					for width_mult in sorted_width_mult_list:
						output = dict_out[width_mult]
						loss = criterion(output, target).mean()
						all_losses.append(loss)
						all_output.append(output)
				else:
					dict_out = model(image, max_width)
					loss = criterion(dict_out[max_width], target).mean()
					all_losses.append(loss)
					all_output.append(dict_out[max_width])
						
			# step 2. benckmark and print info
			batch_size_now = image.size(0)
			for i in range(len(sorted_width_mult_list)):
				acc1, acc5 = accuracy(all_output[i], target, topk=(1, 5))
				m_top1_all[i].update(acc1[0].item(), batch_size_now)
				m_top5_all[i].update(acc5[0].item(), batch_size_now)
				m_ce_all[i].update(all_losses[i].mean().item(), batch_size_now)

		if torch.cuda.is_available(): torch.cuda.synchronize()
		all_infer_time = time.time() - infer_start_t
		logging.info('The total model inference time of the program is {:.2f} Seconds\n'.format(all_infer_time))
		
		info_tmp = '\n'
		final_acc1_all, final_acc5_all = [], []
		for i, w in enumerate(sorted_width_mult_list):
			info_tmp += (f"Width: {w}, Top-1: {m_top1_all[i].avg:.2f}, Top-5: {m_top5_all[i].avg:.2f},"
						f"Loss: {m_ce_all[i].avg:.2f}\n")
			final_acc1_all.append(m_top1_all[i].avg)
			final_acc5_all.append(m_top5_all[i].avg)

		logging.info(info_tmp)
		logging.info('The total number of validation samples is {}'.format(m_top1_all[0].count))

	return final_acc1_all, final_acc5_all, info_tmp, all_infer_time


if __name__ == "__main__":
	args = get_args()
	if torch.cuda.is_available():
		print('The CUDA version is {}'.format(torch.version.cuda))

	# dataset
	args.is_dali = os.path.exists(args.dali_data_dir)
	main(args)
