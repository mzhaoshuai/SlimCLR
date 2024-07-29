# coding=utf-8
"""
A buider for moco on cifar
https://colab.research.google.com/github/facebookresearch/moco/blob/colab-notebook/colab/moco_cifar10_demo.ipynb#scrollTo=lzFyFnhbk8hj
"""
import torch
import torch.nn.modules as nn
import torch.nn.functional as F
from ..models.distiller import choose_kd_loss
from ..models.backbones.s_resnet_cifar import s_ResNet_CIFAR


class ModelMoCo(nn.Module):
	def __init__(self, dim=128, K=4096, m=0.99, T=0.1, arch='resnet18', bn_splits=8, symmetric=False,
					width_mult_list=[1.0, 0.5], args=None):
		super(ModelMoCo, self).__init__()

		self.K = K
		self.m = m
		self.T = T
		self.symmetric = symmetric
		self.width_mult_list = sorted(width_mult_list, reverse=True)
		self.slim_start_epoch = getattr(args, 'slim_start_epoch', -1)
		self.max_width, self.min_width = max(self.width_mult_list), min(self.width_mult_list)
		self.seed_loss = getattr(args, 'seed_loss', False)
		self.aux_loss_w = getattr(args, 'aux_loss_w', 0.5)
		self.tea_T = getattr(args, 'teacher_T', 1.0)
		self.stu_T = getattr(args, 'student_T', 1.0)
		self.loss_weights = getattr(args, 'loss_w', [1.0 for _ in self.width_mult_list])

		# create the encoders
		self.encoder_q = s_ResNet_CIFAR(num_classes=dim, input_size=32, width_mult_list=self.width_mult_list, args=args)
		self.encoder_k = s_ResNet_CIFAR(num_classes=dim, input_size=32, width_mult_list=self.width_mult_list, args=args)
		# initialize,  not update momentum encoder by gradient
		for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
			param_k.data.copy_(param_q.data)  
			param_k.requires_grad = False 

		# create the queue
		self.register_buffer("queue", torch.randn(dim, K))
		self.queue = F.normalize(self.queue, dim=0)
		self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
		self.register_buffer("old_queue_ptr", torch.zeros(1, dtype=torch.long))
		self.register_buffer("max_logits", torch.randn(args.batch_size, K + 1))
		self.register_buffer("old_k", torch.randn(args.batch_size, dim))

		print("[model] s_MoCo_CIFAR:\n"
						"\t feature dim: {}\n"
						"\t queue K / momentum m / temperature T: [ {} / {} / {} ]\n"
						"\t width_mult_list: {}\n"
						"\t loss_weights / slim_start_epoch: [ {} / {}]\n"
						"\t aux_loss_w / teacher_T / student_T: [ {} / {} / {} ]\n"
						"\t SEED: {}".format(
						dim, self.K, self.m, self.T, self.width_mult_list, self.loss_weights,
						self.slim_start_epoch, self.aux_loss_w, self.tea_T, self.stu_T, self.seed_loss))

	@torch.no_grad()
	def _momentum_update_key_encoder(self):
		"""
		Momentum update of the key encoder
		"""
		for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
			param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

	@torch.no_grad()
	def _dequeue_and_enqueue(self, keys):
		batch_size = keys.shape[0]

		ptr = int(self.queue_ptr)
		assert self.K % batch_size == 0                     # for simplicity
		self.old_queue_ptr[0] = ptr

		# replace the keys at ptr (dequeue and enqueue)
		self.queue[:, ptr:ptr + batch_size] = keys.t()      # transpose
		ptr = (ptr + batch_size) % self.K                   # move pointer

		self.queue_ptr[0] = ptr

	@torch.no_grad()
	def _batch_shuffle_single_gpu(self, x):
		"""
		Batch shuffle, for making use of BatchNorm.
		"""
		# random shuffle index
		idx_shuffle = torch.randperm(x.shape[0]).cuda()
		# index for restoring
		idx_unshuffle = torch.argsort(idx_shuffle)

		return x[idx_shuffle], idx_unshuffle

	@torch.no_grad()
	def _batch_unshuffle_single_gpu(self, x, idx_unshuffle):
		"""
		Undo batch shuffle.
		"""
		return x[idx_unshuffle]

	def forward_single(self, im_q, im_k, epoch=0, width=1.0):
		"""
		Input:
			im_q: a batch of query images
			im_k: a batch of key images
		Output:
			loss
		"""
		enable_slim = epoch >= self.slim_start_epoch
		if not enable_slim and width < self.max_width:
			return torch.zeros(1, device=im_q.device).mean()

		q = self.encoder_q.forward_single((im_q, width))
		q = F.normalize(q, dim=1)
		# compute key features, no gradient to keys
		with torch.no_grad():  		
			if width == self.max_width:
				# update the key encoder 
				self._momentum_update_key_encoder()
				# shuffle for making use of BN
				im_k, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
				# keys: [N, C]
				k = self.encoder_k(im_k, self.max_width)[self.max_width]  					
				k = F.normalize(k, dim=1)
				# undo shuffle
				k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)
				self.old_k = k
				# dequeue and enqueue
				# self._dequeue_and_enqueue(k)
			else:
				k = self.old_k
				# batch_size = im_q.shape[0]
				# ptr = int(self.old_queue_ptr)
				# k = self.queue[:, ptr:ptr + batch_size].clone().detach().t()

		# labels: positive key indicators, [N,]
		labels = torch.zeros(k.shape[0], dtype=torch.long).to(k.device)

		# calculate contrastive loss
		if width == self.max_width:
			loss, logits = self.InfoNCE(q, k, self.queue.clone().detach(), labels)
			loss = loss * self.loss_weights[0] if enable_slim else loss
			self.max_logits = logits.clone().detach()
	
		else:
			loss, logits = self.InfoNCE(q, k, self.queue.clone().detach(), labels)
			soft_target = self.max_logits
			if self.seed_loss:
				aux_loss = choose_kd_loss(logits, soft_target, stu_T=self.stu_T, tea_T=self.tea_T,
												labels=labels, kl_div=self.seed_loss)
				# here we assume only two networks			
				loss = ((1.0 - self.aux_loss_w) * loss + self.aux_loss_w * aux_loss)
			loss = self.loss_weights[-1] * loss

		if width == self.min_width:
			self._dequeue_and_enqueue(k)

		return loss

	def forward(self, im_q, im_k, epoch=0):
		"""
		Input:
			im_q: a batch of query images
			im_k: a batch of key images
		Output:
			loss
		"""
		output_dict = {'outputs': None,
						'labels': None,
						'loss': None
						}
		enable_slim = epoch >= self.slim_start_epoch
		if enable_slim:
			# compute query features, a dict of queries: [N, C]
			q_dict = self.encoder_q(im_q)
			for w in self.width_mult_list:
				q_dict[w] = F.normalize(q_dict[w], dim=1)
		else:
			q_dict = self.encoder_q(im_q, specify_width=self.max_width)
			q_dict[self.max_width] = F.normalize(q_dict[self.max_width], dim=1)
			for w in self.width_mult_list[1:]:
				q_dict[w] = q_dict[self.max_width].clone().detach()

		# compute key features, no gradient to keys
		with torch.no_grad():  		
			# update the key encoder 
			self._momentum_update_key_encoder()  		
			# shuffle for making use of BN
			im_k, idx_unshuffle = self._batch_shuffle_single_gpu(im_k)
			# keys: [N, C]
			k = self.encoder_k(im_k, self.max_width)[self.max_width]  					
			k = F.normalize(k, dim=1)
			# undo shuffle
			k = self._batch_unshuffle_single_gpu(k, idx_unshuffle)

		# labels: positive key indicators, [N,]
		labels = torch.zeros(k.shape[0], dtype=torch.long).to(k.device)

		loss, all_losses_d = self.ssl_loss(q_dict, k, self.queue.clone().detach(), labels,
											enable_slim=enable_slim)

		output_dict['outputs'] = q_dict
		output_dict['labels'] = labels
		output_dict['loss'] = loss
		output_dict['all_losses'] = all_losses_d

		# dequeue and enqueue
		self._dequeue_and_enqueue(k)

		return output_dict

	def ssl_loss(self, q_dict, k, neg, labels, enable_slim=True):
		losses, d_losses, all_logits = [], [], []
		
		# first calculate the loss of max_width
		max_loss, max_logits = self.InfoNCE(q_dict[self.max_width], k, neg, labels)
		max_loss = max_loss * self.loss_weights[0] if enable_slim else max_loss
		losses.append(max_loss)
		d_losses.append(max_loss.clone().detach())

		if enable_slim:
			# subnetworks are able to learn
			if self.seed_loss:
				all_logits.append(max_logits.clone().detach())

			for index, w in enumerate(self.width_mult_list[1:]):
				# contrastive loss
				loss_tmp, logits_tmp = self.InfoNCE(q_dict[w], k, neg, labels)
				aux_loss = torch.zeros_like(loss_tmp).mean()
				denominator_ = 0.0
				for soft_target in all_logits:
					if self.seed_loss:
						aux_loss_tmp = choose_kd_loss(logits_tmp, soft_target, stu_T=self.stu_T, tea_T=self.tea_T,
														labels=labels, kl_div=self.seed_loss)						
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

		else:
			# do not train subnetworks
			for w in self.width_mult_list[1:]:
				loss_tmp = torch.zeros(1, device=k.device, dtype=k.dtype).mean()
				losses.append(loss_tmp)
				d_losses.append(loss_tmp.clone().detach())

		loss = torch.sum(torch.stack(losses))
		return loss, d_losses


	def InfoNCE(self, query, pos, neg, labels):
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
		loss_tmp = F.cross_entropy(logits / self.T, labels, reduction='mean').mean()
		# loss_tmp = criterion(logits / self.T, labels).mean()

		return loss_tmp, logits
