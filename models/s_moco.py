# coding=utf-8
"""
slimmable networks with MoCo
"""
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from .moco import concat_all_gather
from .distiller import choose_kd_loss
from .utils import loss_weights_generate


class s_MoCo(nn.Module):
	"""
	Build a MoCo model with: a query encoder, a key encoder, and a queue
	https://arxiv.org/abs/1911.05722.
	By default, the encoder is a slimmable network.
	"""
	def __init__(self, base_encoder, dim=128, K=65536, m=0.999, T=0.07, mlp=True,
					args=None):
		"""
		dim: feature dimension (default: 128)
		K: queue size; number of negative keys (default: 65536)
		m: moco momentum of updating key encoder (default: 0.999)
		T: softmax temperature (default: 0.07)
		"""
		super(s_MoCo, self).__init__()

		self.K = K
		self.m = m
		self.T = T

		# config for slimmable networks
		self.width_mult_list = sorted(args.width_mult_list, reverse=True)
		self.max_width = max(self.width_mult_list)
		self.inplace_distill = getattr(args, 'inplace_distill', False)
		self.seed_loss = getattr(args, 'seed_loss', False)
		self.mse_loss = getattr(args, 'mse_loss', 0)
		self.atkd_loss = getattr(args, 'atkd_loss', 0)
		self.dkd_loss = getattr(args, 'dkd_loss', 0)
		self.aux_loss_w = getattr(args, 'aux_loss_w', 0.5)
		self.slim_start_epoch = getattr(args, 'slim_start_epoch', -1)
		self.tea_T = getattr(args, 'teacher_T', 1.0)
		self.stu_T = getattr(args, 'student_T', 1.0)
		self.full_distill = getattr(args, 'full_distill', False)
		if not (self.inplace_distill or self.seed_loss or self.mse_loss or self.atkd_loss or self.dkd_loss):
			self.aux_loss_w = 0.0

		assert not (self.inplace_distill and self.seed_loss)
		self.loss_weights = loss_weights_generate(getattr(args, 'slim_loss_weighted', False), self.width_mult_list)

		# create the encoders, num_classes is the output fc dimension
		self.encoder_q = base_encoder(num_classes=dim, input_size=args.image_size, args=args)
		self.encoder_k = base_encoder(num_classes=dim, input_size=args.image_size, args=args)

		# initialize & not update by gradient
		for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
			param_k.data.copy_(param_q.data)  	
			param_k.requires_grad = False

		# create the queue
		self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
		self.register_buffer("queue", torch.randn(dim, K))
		self.queue = nn.functional.normalize(self.queue, dim=0)

		if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
			logging.info("[model] s_MoCov2:\n"
							"\t feature dim / use mlp: [ {} / {} ]\n"
							"\t queue K / momentum m / temperature T: [ {} / {} / {} ]\n"
							"\t width_mult_list: {}\n"
							"\t loss_weights / slim_start_epoch / full_distill: [ {} / {} / {} ]\n"
							"\t aux_loss_w / teacher_T / student_T: [ {} / {} / {} ]\n"
							"\t KD / SEED / MSE / ATKD / DKD: [ {} / {} / {} / {} / {} ]".format(
							dim, mlp, self.K, self.m, self.T, self.width_mult_list, self.loss_weights,
							self.slim_start_epoch, self.full_distill, self.aux_loss_w, self.tea_T, self.stu_T,
							self.inplace_distill, self.seed_loss, self.mse_loss, self.atkd_loss, self.dkd_loss))

	@torch.no_grad()
	def _momentum_update_key_encoder(self):
		"""
		Momentum update of the key encoder
		"""
		for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

	@torch.no_grad()
	def _dequeue_and_enqueue(self, keys):
		# gather keys before updating queue
		keys = concat_all_gather(keys)

		batch_size = keys.shape[0]

		ptr = int(self.queue_ptr)
		assert self.K % batch_size == 0  # for simplicity

		# replace the keys at ptr (dequeue and enqueue)
		self.queue[:, ptr:ptr + batch_size] = keys.T
		ptr = (ptr + batch_size) % self.K  # move pointer

		self.queue_ptr[0] = ptr

	@torch.no_grad()
	def _batch_shuffle_ddp(self, x):
		"""
		Batch shuffle, for making use of BatchNorm.
		*** Only support DistributedDataParallel (DDP) model. ***
		"""
		# gather from all gpus
		batch_size_this = x.shape[0]
		x_gather = concat_all_gather(x)
		batch_size_all = x_gather.shape[0]

		num_gpus = batch_size_all // batch_size_this

		# random shuffle index
		idx_shuffle = torch.randperm(batch_size_all).cuda()

		# broadcast to all gpus
		torch.distributed.broadcast(idx_shuffle, src=0)

		# index for restoring
		idx_unshuffle = torch.argsort(idx_shuffle)

		# shuffled index for this gpu
		gpu_idx = torch.distributed.get_rank()
		idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

		return x_gather[idx_this], idx_unshuffle

	@torch.no_grad()
	def _batch_unshuffle_ddp(self, x, idx_unshuffle):
		"""
		Undo batch shuffle.
		*** Only support DistributedDataParallel (DDP) model. ***
		"""
		# gather from all gpus
		batch_size_this = x.shape[0]
		x_gather = concat_all_gather(x)
		batch_size_all = x_gather.shape[0]

		num_gpus = batch_size_all // batch_size_this

		# restored index for this gpu
		gpu_idx = torch.distributed.get_rank()
		idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

		return x_gather[idx_this]

	def forward(self, im_q, im_k, sweet_x=None, criterion=None, soft_criterion=None, epoch=None):
		"""
		Input:
			im_q: a batch of query images, [N, 3, H, W]
			im_k: a batch of key images, [N, 3, H, W]
		Output:
			logits, targets
		"""
		output_dict = {'outputs': None,
						'labels': None,
						'loss': None
						}

		enable_slim = epoch >= self.slim_start_epoch

		if enable_slim:
			# compute query features, a dict of queries: [N, C]
			q_dict = self.encoder_q(im_q, sweet_x=sweet_x)
			for w in self.width_mult_list:
				q_dict[w] = nn.functional.normalize(q_dict[w], dim=1)
		else:
			q_dict = self.encoder_q(im_q, specify_width=self.max_width)
			q_dict[self.max_width] = nn.functional.normalize(q_dict[self.max_width], dim=1)
			for w in self.width_mult_list[1:]:
				q_dict[w] = q_dict[self.max_width].clone().detach()

		# compute key features, no gradient to keys
		with torch.no_grad():  		
			# update the key encoder 
			self._momentum_update_key_encoder()  		

			# shuffle for making use of BN
			im_k, idx_unshuffle = self._batch_shuffle_ddp(im_k)

			# keys: [N, C]
			k = self.encoder_k(im_k, self.max_width)[self.max_width]  					
			k = nn.functional.normalize(k, dim=1)

			# undo shuffle
			k = self._batch_unshuffle_ddp(k, idx_unshuffle)

		# labels: positive key indicators, [N,]
		labels = torch.zeros(k.shape[0], dtype=torch.long).cuda()

		loss, all_losses_d = self.ssl_loss(q_dict, k, self.queue.clone().detach(), labels,
											enable_slim=enable_slim,
											criterion=criterion, soft_criterion=soft_criterion)

		output_dict['outputs'] = q_dict
		output_dict['labels'] = labels
		output_dict['loss'] = loss
		output_dict['all_losses'] = all_losses_d

		# dequeue and enqueue
		self._dequeue_and_enqueue(k)

		return output_dict

	def ssl_loss(self, q_dict, k, neg, labels, enable_slim=True,
					criterion=None, soft_criterion=None):
		losses, d_losses, all_logits = [], [], []
		
		# first calculate the loss of max_width
		max_loss, max_logits = self.InfoNCE(q_dict[self.max_width], k, neg, labels, criterion=criterion)
		max_loss = max_loss * self.loss_weights[0] if enable_slim else max_loss
		losses.append(max_loss)
		d_losses.append(max_loss.clone().detach())
		

		if enable_slim:
			# subnetworks are able to learn
			if self.inplace_distill or self.mse_loss == 1:
				all_logits.append(q_dict[self.max_width].clone().detach())

			if self.mse_loss == 2 or self.dkd_loss or self.atkd_loss or self.seed_loss:
				all_logits.append(max_logits.clone().detach())

			for index, w in enumerate(self.width_mult_list[1:]):
				# contrastive loss
				loss_tmp, logits_tmp = self.InfoNCE(q_dict[w], k, neg, labels, criterion=criterion)

				aux_loss = torch.zeros_like(loss_tmp).mean()
				denominator_ = 0.0
				for soft_target in all_logits:
					
					if self.inplace_distill or self.mse_loss == 1:
						aux_loss_tmp = choose_kd_loss(q_dict[w], soft_target, stu_T=self.stu_T, tea_T=self.tea_T,
														mse=self.mse_loss, kl_div=self.inplace_distill)

					elif self.mse_loss == 2 or self.seed_loss or self.atkd_loss or self.dkd_loss:
						aux_loss_tmp = choose_kd_loss(logits_tmp, soft_target, stu_T=self.stu_T, tea_T=self.tea_T,
														labels=labels,
														mse=self.mse_loss, kl_div=self.seed_loss,
														dkd=self.dkd_loss, atkd=self.atkd_loss)						
					else:
						aux_loss_tmp = torch.zeros_like(loss_tmp).mean()

					denominator_ += 1.0
					aux_loss = aux_loss + aux_loss_tmp

				if denominator_ > 0:
					aux_loss = aux_loss / denominator_

				# the loss of this subnetwork
				loss_tmp = self.loss_weights[index + 1] * ((1.0 - self.aux_loss_w) * loss_tmp + self.aux_loss_w * aux_loss)

				losses.append(loss_tmp)
				d_losses.append(loss_tmp.clone().detach())
				if self.full_distill:
					if self.inplace_distill or self.mse_loss == 1:
						all_logits.append(q_dict[w].clone().detach())

					if self.mse_loss == 2 or self.dkd_loss or self.atkd_loss or self.seed_loss:
						all_logits.append(logits_tmp.clone().detach())				

		else:
			# do not train subnetworks
			for w in self.width_mult_list[1:]:
				loss_tmp = torch.zeros(1, device=k.device, dtype=k.dtype).mean()
				losses.append(loss_tmp)
				d_losses.append(loss_tmp.clone().detach())

		loss = torch.sum(torch.stack(losses))
		return loss, d_losses


	def InfoNCE(self, query, pos, neg, labels, criterion=None):
		"""
		an InfoNCE loss
		return: loss, logits (without temperature)
		"""
		# positive logits: [N, 1]
		l_pos = torch.einsum('nc,nc->n', [query, pos]).unsqueeze(-1)		
		# negative logits: [N, K]
		l_neg = torch.einsum('nc,ck->nk', [query, neg])
		# logits: [N, (1+K)]
		logits = torch.cat([l_pos, l_neg], dim=1)
		# apply temperature and calculate loss
		loss_tmp = criterion(logits / self.T, labels).mean()

		return loss_tmp, logits
