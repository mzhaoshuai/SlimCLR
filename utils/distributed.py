# coding=utf-8
# some utils for distributed training
# https://github.com/pytorch/vision/blob/main/references/classification/utils.py

import os
import torch
import random
import numpy as np
import torch.nn as nn
import torch.distributed as dist

from collections import OrderedDict
from torch._utils import (_flatten_dense_tensors, _take_tensors, _unflatten_dense_tensors)
from torch.nn.parallel.scatter_gather import scatter_kwargs


def is_dist_avail_and_initialized():
	if not dist.is_available():
		return False
	if not dist.is_initialized():
		return False
	return True


def get_world_size():
	if not is_dist_avail_and_initialized():
		return 1
	return dist.get_world_size()


def get_rank():
	if not is_dist_avail_and_initialized():
		return 0
	return dist.get_rank()


def is_master():
	"""check if current process is the master"""
	return get_rank() == 0


def setup_for_distributed(is_master):
	"""
	This function disables printing when not in master process
	"""
	import builtins as __builtin__

	builtin_print = __builtin__.print

	def print(*args, **kwargs):
		force = kwargs.pop("force", False)
		if is_master or force:
			builtin_print(*args, **kwargs)

	__builtin__.print = print


def set_random_seed(seed=None, FLAGS=None):
	"""set random seed"""
	if seed is None:
		seed = getattr(FLAGS, 'random_seed', 3407)
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def init_distributed_mode(args, ngpus_per_node, gpu):
	"""initialize for distributed training"""

	if args.distributed:
		print("INFO: [CUDA] Initialize process group for distributed training")
		global_rank = args.local_rank * ngpus_per_node + gpu
		print("INFO: [CUDA] Use [GPU: {} / Global Rank: {}] for training, "
						"init_method {}, world size {}".format(gpu, global_rank, args.init_method, args.world_size))
		# set device before init process group
		# Ref: https://github.com/pytorch/pytorch/issues/18689
		torch.cuda.set_device(args.gpu)	
		torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.init_method,
												world_size=args.world_size, rank=global_rank)
		if torch.__version__ <= "1.7.2":
			torch.distributed.barrier()
		else:
			torch.distributed.barrier(device_ids=[args.gpu])
		setup_for_distributed(global_rank == 0)

	else:
		args.local_rank = gpu
		global_rank = 0
		print("Use [GPU: {}] for training".format(gpu))

	return global_rank


# def init_distributed_mode(args):
#     if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
#         args.rank = int(os.environ["RANK"])
#         args.world_size = int(os.environ["WORLD_SIZE"])
#         args.gpu = int(os.environ["LOCAL_RANK"])
#     elif "SLURM_PROCID" in os.environ:
#         args.rank = int(os.environ["SLURM_PROCID"])
#         args.gpu = args.rank % torch.cuda.device_count()
#     elif hasattr(args, "rank"):
#         pass
#     else:
#         print("Not using distributed mode")
#         args.distributed = False
#         return

#     args.distributed = True

#     torch.cuda.set_device(args.gpu)
#     args.dist_backend = "nccl"
#     print(f"| distributed init (rank {args.rank}): {args.dist_url}", flush=True)
#     torch.distributed.init_process_group(
#         backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
#     )
#     torch.distributed.barrier()
#     setup_for_distributed(args.rank == 0)


#####################################################################
#
# code from https://github.com/JiahuiYu/slimmable_networks
#
######################################################################


def dist_reduce_tensor(tensor):
	""" Reduce to rank 0 """
	world_size = get_world_size()
	if world_size < 2:
		return tensor
	with torch.no_grad():
		dist.reduce(tensor, dst=0)
		if get_rank() == 0:
			tensor /= world_size
	return tensor


def dist_all_reduce_tensor(tensor):
	""" Reduce to all ranks """
	world_size = get_world_size()
	if world_size < 2:
		return tensor
	with torch.no_grad():
		dist.all_reduce(tensor)
		tensor.div_(world_size)
	return tensor


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
	if bucket_size_mb > 0:
		bucket_size_bytes = bucket_size_mb * 1024 * 1024
		buckets = _take_tensors(tensors, bucket_size_bytes)
	else:
		buckets = OrderedDict()
		for tensor in tensors:
			tp = tensor.type()
			if tp not in buckets:
				buckets[tp] = []
			buckets[tp].append(tensor)
		buckets = buckets.values()

	for bucket in buckets:
		flat_tensors = _flatten_dense_tensors(bucket)
		dist.all_reduce(flat_tensors)
		flat_tensors.div_(world_size)
		for tensor, synced in zip(
				bucket, _unflatten_dense_tensors(flat_tensors, bucket)):
			tensor.copy_(synced)


def allreduce_grads(model, coalesce=True, bucket_size_mb=-1):
	grads = [
		param.grad.data for param in model.parameters()
		if param.requires_grad and param.grad is not None
	]
	world_size = dist.get_world_size()
	if coalesce:
		_allreduce_coalesced(grads, world_size, bucket_size_mb)
	else:
		for tensor in grads:
			dist.all_reduce(tensor.div_(world_size))


class AllReduceDistributedDataParallel(nn.Module):

	def __init__(self, module, dim=0, broadcast_buffers=True,
				 bucket_cap_mb=25):
		super(AllReduceDistributedDataParallel, self).__init__()
		self.module = module
		self.dim = dim
		self.broadcast_buffers = broadcast_buffers

		self.broadcast_bucket_size = bucket_cap_mb * 1024 * 1024
		self._sync_params()

	def _dist_broadcast_coalesced(self, tensors, buffer_size):
		for tensors in _take_tensors(tensors, buffer_size):
			flat_tensors = _flatten_dense_tensors(tensors)
			dist.broadcast(flat_tensors, 0)
			for tensor, synced in zip(
					tensors, _unflatten_dense_tensors(flat_tensors, tensors)):
				tensor.copy_(synced)

	def _sync_params(self):
		module_states = list(self.module.state_dict().values())
		if len(module_states) > 0:
			self._dist_broadcast_coalesced(module_states,
										   self.broadcast_bucket_size)
		if self.broadcast_buffers:
			buffers = [b.data for b in self.module.buffers()]
			if len(buffers) > 0:
				self._dist_broadcast_coalesced(buffers,
											   self.broadcast_bucket_size)

	def scatter(self, inputs, kwargs, device_ids):
		return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)

	def forward(self, *inputs, **kwargs):
		inputs, kwargs = self.scatter(inputs, kwargs,
									  [torch.cuda.current_device()])
		return self.module(*inputs[0], **kwargs[0])

