# coding=utf-8
import os
import sys
import time
import torch
import logging
import datetime
import warnings
import torch.multiprocessing as mp
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard import SummaryWriter

from dataset import get_dataloader
from dataset.transforms import random_solarize_normalize
from models.utils import get_grad_norm
from models.model_zoo import get_ssl_model
from params import get_args, save_hp_to_json
from utils.lr_scheduler import lr_scheduler
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
	if args.is_log_grad and args.grad_log: mkdir(os.path.dirname(args.grad_log))

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
	# from dataset.dataloader import get_dataloader
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
	model = get_ssl_model(args)


	if not torch.cuda.is_available():
		model.float()
		logging.warning("using CPU, this will be slow")
		# comment out the following line for debugging
		raise NotImplementedError("Only DistributedDataParallel is supported.")
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
										find_unused_parameters=False if args.slim_start_epoch < 0 else True)
		if args.dp:
			model = torch.nn.DataParallel(model, device_ids=args.multigpu)
			# comment out the following line for debugging
			raise NotImplementedError("Only DistributedDataParallel is supported.")

	# here use local_rank
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu", get_rank() % ngpus_per_node)

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
	soft_criterion = None

	# grouped_parameters = model.parameters()
	grouped_parameters = get_parameter_groups(model, args.lr, args.weight_decay,
												norm_weight_decay=0,
												norm_bias_no_decay=True)
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
	best_e = 0
	save_epoch = 100 if args.ssl_arch == 'mocov2' else 50
	for epoch in range(start_epoch, args.num_epochs):
		if args.slim_start_epoch > 0 and epoch >= args.slim_start_epoch:
			scheduler.init_lr = getattr(args, 'slim_start_lr', args.lr)

		if is_dist_avail_and_initialized() and not args.is_dali:
			train_loader.sampler.set_epoch(epoch)
		if not args.slimmable_training:
			global_step, train_top1 = train_epoch(train_loader, model, criterion, optimizer, epoch, global_step,
													scheduler=scheduler, scaler=scaler, args=args)
		else:
			global_step, train_top1 = train_epoch_slim(train_loader, model, criterion, optimizer, epoch, global_step,
														scheduler=scheduler, scaler=scaler, args=args,
														tf_writer=tf_writer)			

		if is_master():
			logging.info("Epoch %d/%s Finished.", epoch + 1, args.num_epochs)
			if best_acc1 <= train_top1:
				best_acc1 = train_top1
				best_e = epoch
			# save checkpoint
			ckpt_dict = {
					'epoch': epoch + 1,
					'global_step': global_step,
					'arch': args.arch,
					'state_dict': model.state_dict(),
					'best_acc1': best_acc1,
					'optimizer': optimizer.state_dict(),
				}
			if scaler is not None: ckpt_dict['scaler'] = scaler.state_dict()
			save_checkpoint(ckpt_dict, False, args.log_dir, filename='ckpt.pth.tar')
			if epoch > 0 and (epoch + 1) % save_epoch == 0:
				save_checkpoint(ckpt_dict, False, args.log_dir, filename='ckpt_e{}.pth.tar'.format(epoch + 1))

		# reset dali dataloader for each epoch
		if args.is_dali: train_loader.reset()	

	if is_master():
		if torch.cuda.is_available(): torch.cuda.synchronize()
		all_time = time.time() - all_start
		logging.info('The total running time of the program is {:.1f} Hour {:.1f} Minute\n'.format(all_time // 3600, 
							all_time % 3600 / 60))
		logging.info('The maximum GPU memory occupied by this program is {:.2f} GB\n'.format(
							torch.cuda.max_memory_allocated(0) * 1.0 / 1024 / 1024 / 1024))
		logging.info("The best Top-1 Acc (Train) is: {:.2f}, best_e={}\n".format(
							best_acc1, best_e))
		print("The above program id is {}\n".format(args.log_dir))

	torch.cuda.empty_cache()


def train_epoch(loader, model, criterion, optimizer, epoch, global_step,
				scheduler=None, scaler=None, args=None):
	samples_per_epoch = len(loader.dataset) if not args.is_dali else \
							len(loader) * args.batch_size_per_gpu * args.world_size
	step_per_epoch = len(loader)
	losses = AverageMeter('Loss', ':.4e')
	top1 = AverageMeter('Acc@1', ':6.2f')
	top5 = AverageMeter('Acc@5', ':6.2f')

	# switch to train mode
	model.train()
	end = time.time()
	train_start_t = time.time()

	for step, data in enumerate(loader):
		# perform accumulated gradient udpate
		step_cond = step % args.gradient_accumulation_steps == 0

		images_q = data[0][0] if not args.is_dali else data[0]['data_q']
		images_k = data[0][1] if not args.is_dali else data[0]['data_k']	
		if args.gpu is not None:
			images_q= images_q.cuda(args.gpu, non_blocking=True)
			images_k = images_k.cuda(args.gpu, non_blocking=True)
		# for mocov3, when using dali, aug2 does not do random solarize and normalize
		if args.ssl_aug == 'mocov3' and args.is_dali:
			images_k = random_solarize_normalize(images_k, p=0.2)

		if scheduler is not None: scheduler(optimizer, epoch=epoch, global_step=global_step)
		# measure data loading time
		data_time = time.time() - end

		with torch.cuda.amp.autocast(enabled = scaler is not None):
			output_dict = model(im_q=images_q, im_k=images_k, criterion=criterion,
									epoch=step / step_per_epoch + epoch)

		# divide loss by gradient_accumulation_steps to accumulate gradient
		loss = output_dict['loss'] / args.gradient_accumulation_steps
		# compute gradient and do SGD step
		if scaler is not None:
			scaler.scale(loss).backward()
			if step_cond:
				if args.clip_grad_norm is not None:
					# we should unscale the gradients of optimizer's assigned params if do gradient clipping
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
				scaler.step(optimizer)
				scaler.update()
		else:
			loss.backward()
			if step_cond:
				if args.clip_grad_norm is not None:
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
				optimizer.step()

		# acc1/acc5 are (K+1)-way contrast classifier accuracy
		# measure accuracy and record loss
		output, target = output_dict['outputs'], output_dict['labels']
		batch_size_now = images_q.size(0)
		acc1, acc5 = accuracy(output, target, topk=(1, 5))
		losses.update(loss.item(), batch_size_now)
		top1.update(acc1[0], batch_size_now)
		top5.update(acc5[0], batch_size_now)

		# measure elapsed time
		batch_time = time.time() - end
		end = time.time()

		if step_cond:
			global_step += 1
			if global_step % args.n_display == 0 and is_master():
				percent_complete = step * 1.0 / step_per_epoch * 100
				lr_tmp = optimizer.param_groups[0]['lr']
				info_tmp =	(f"Epoch: {epoch} [({percent_complete:.1f}%)] "
							f"Loss: [{losses.avg:.3f}] Acc@1: [{top1.avg:.2f}] "
							f"Data (t) {data_time:.3f} Batch (t) {batch_time:.2f} "
							f"LR: {lr_tmp:.1e}".replace(', ]', ']'))

				logging.info(info_tmp)
			optimizer.zero_grad()

	if torch.cuda.is_available(): torch.cuda.synchronize()
	if is_master():
		one_epoch_time = time.time() - train_start_t
		logging.info('The total number of training samples for this device is {}'.format(top1.count))
		logging.info('The total model train time for one epoch is is {:.2f} Seconds\n'.format(one_epoch_time))
		logging.info('The throughout is {:.2f} images per second\n'.format(samples_per_epoch / one_epoch_time))

	return global_step, top1.avg


def train_epoch_slim(loader, model, criterion, optimizer, epoch, global_step,
						scheduler=None, scaler=None, soft_criterion=None, tf_writer=None,
						args=None):
	samples_per_epoch = len(loader.dataset) if not args.is_dali else \
							len(loader) * args.batch_size_per_gpu * args.world_size
	step_per_epoch = len(loader)
	sorted_width_mult_list = sorted(args.width_mult_list, reverse=True)

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
		# perform accumulated gradient udpate
		step_cond = step % args.gradient_accumulation_steps == 0

		images_q = data[0][0] if not args.is_dali else data[0]['data_q']
		images_k = data[0][1] if not args.is_dali else data[0]['data_k']		
		if args.gpu is not None:
			images_q= images_q.cuda(args.gpu, non_blocking=True)
			images_k = images_k.cuda(args.gpu, non_blocking=True)

			if args.ssl_aug == 'mocov2_slim':
				sweet_x = dict()
				for i in range(1, len(sorted_width_mult_list)):
					images_kx = data[i + 1] if not args.is_dali else data[0]['data_k{}'.format(i)]
					sweet_x[sorted_width_mult_list[i]] = images_kx.cuda(args.gpu, non_blocking=True)
			else:
				sweet_x = None

		if scheduler is not None: scheduler(optimizer, epoch=epoch, global_step=global_step)
		# measure data loading time
		data_time = time.time() - end

		with torch.cuda.amp.autocast(enabled = scaler is not None):
			output_dict = model(im_q=images_q, im_k=images_k, sweet_x=sweet_x,
									criterion=criterion, epoch=step / step_per_epoch + epoch)

		# divide loss by gradient_accumulation_steps to accumulate gradient
		loss = output_dict['loss'] / args.gradient_accumulation_steps
		# compute gradient and do SGD step
		if scaler is not None:
			scaler.scale(loss).backward()
			if step_cond:
				if args.clip_grad_norm is not None:
					# we should unscale the gradients of optimizer's assigned params if do gradient clipping
					scaler.unscale_(optimizer)
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
				scaler.step(optimizer)
				scaler.update()
		else:
			loss.backward()
			if step_cond:
				if args.clip_grad_norm is not None:
					torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)
				optimizer.step()

		# acc1/acc5 are (K+1)-way contrast classifier accuracy
		# measure accuracy and record loss
		all_output, target = output_dict['outputs'], output_dict['labels']
		all_losses = output_dict['all_losses']
		batch_size_now = images_q.size(0)
		acc_str, loss_str = '', ''
		for i in range(len(sorted_width_mult_list)):
			acc1, acc5 = accuracy(all_output[sorted_width_mult_list[i]], target, topk=(1, 5))
			m_top1_all[i].update(acc1[0].item(), batch_size_now)
			m_ce_all[i].update(all_losses[i].mean().item(), batch_size_now)

			acc_str += '{:.2f}, '.format(m_top1_all[i].avg)
			loss_str += '{:.3f}, '.format(m_ce_all[i].avg)		

		# measure elapsed time
		batch_time = time.time() - end
		end = time.time()

		if step_cond:
			global_step += 1
			if global_step % args.n_display == 0 and is_master():
				if args.is_log_grad:
					get_grad_norm(model, sorted_width_mult_list, global_step, tf_writer=tf_writer, args=args)

				num_samples = (step + 1) * batch_size_now * args.world_size
				percent_complete = num_samples * 1.0 / samples_per_epoch * 100
				lr_tmp = optimizer.param_groups[0]['lr']
				info_tmp =	(f"Epoch: {epoch} [({percent_complete:.1f}%)] "
							f"Loss: [{loss_str}] Acc@1: [{acc_str}] "
							f"Data (t) {data_time:.3f} Batch (t) {batch_time:.2f} "
							f"LR: {lr_tmp:.1e}".replace(', ]', ']'))

				logging.info(info_tmp)
			optimizer.zero_grad()

	if torch.cuda.is_available(): torch.cuda.synchronize()
	if is_master():
		one_epoch_time = time.time() - train_start_t
		logging.info('The total number of training samples for this device is {}'.format(m_top1_all[0].count))
		logging.info('The total model train time for one epoch is is {:.2f} Seconds\n'.format(one_epoch_time))
		logging.info('The throughout is {:.2f} images per second\n'.format(samples_per_epoch / one_epoch_time))

	return global_step, m_top1_all[0].avg


if __name__ == "__main__":
	args = get_args()
	if torch.cuda.is_available():
		print('The CUDA version is {}'.format(torch.version.cuda))

	# dataset
	args.is_dali = os.path.exists(args.dali_data_dir)
	main(args)

	sys.exit(0)