# coding=utf-8
"""
Slimmabel networks with MoCov3
"""

import math
import torch
import logging
import torch.nn as nn
from .distiller import choose_kd_loss
from .utils import loss_weights_generate
from .backbones.slimmable_ops import SwitchableBatchNorm1d, SlimmableLinear, SlimmableReLU


class s_MoCoV3(nn.Module):
	"""
	Build a s_MoCoV3 model with a base encoder, a momentum encoder, and two MLPs.
	By default, the encoder is a slimmable network.
	"""
	def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, init_m=0.99, num_epochs=100,
					m_cos=True, args=None):
		"""
		dim: feature dimension (default: 256)
		mlp_dim: hidden dimension in MLPs (default: 4096)
		T: softmax temperature (default: 1.0)
		total_e: the number of epochs
		m_cos: whether apply cos increaing momentum
		"""
		super(s_MoCoV3, self).__init__()
		assert T > 0.5 and mlp_dim >= 4096 and dim >=256
		self.T = T
		self.init_m = init_m
		self.num_epochs = num_epochs
		self.m_cos = m_cos

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
		self.overlap_weight = getattr(args, 'overlap_weight', True)
		if not (self.inplace_distill or self.seed_loss or self.mse_loss or self.atkd_loss or self.dkd_loss):
			self.aux_loss_w = 0.0

		assert not (self.inplace_distill and self.seed_loss)
		self.loss_weights = loss_weights_generate(getattr(args, 'slim_loss_weighted', False), self.width_mult_list)

		# create the encoders, num_classes is the output fc dimension
		self.base_encoder = base_encoder(num_classes=dim, input_size=args.image_size, args=args)
		self.momentum_encoder = base_encoder(num_classes=dim, input_size=args.image_size, args=args)

		self._build_projector_and_predictor_mlps(dim, mlp_dim)

		# initialize & not update by gradient
		for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
			param_m.data.copy_(param_b.data)            # initialize
			param_m.requires_grad = False               # not update by gradient

		if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
			logging.info("[model] s_MoCov3:\n"
							"\t feature dim / mlp_dim: [ {} / {} ]\n"
							"\t num_epochs/ init momentum / temperature T / cos_m: [ {} / {} / {} / {}]\n"
							"\t width_mult_list: {}\n"
							"\t loss_weights / slim_start_epoch / full_distill: [ {} / {} / {} ]\n"
							"\t aux_loss_w / teacher_T / student_T: [ {} / {} / {} ]\n"
							"\t KD / SEED / MSE / ATKD / DKD: [ {} / {} / {} / {} / {} ]".format(
							dim, mlp_dim, self.num_epochs, self.init_m, self.T, self.m_cos,
							self.width_mult_list, self.loss_weights,
							self.slim_start_epoch, self.full_distill, self.aux_loss_w, self.tea_T, self.stu_T,
							self.inplace_distill, self.seed_loss, self.mse_loss, self.atkd_loss, self.dkd_loss))

	def _build_slimmable_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
		mlp = []
		for l in range(num_layers):
			dim1 = input_dim if l == 0 else mlp_dim
			dim2 = output_dim if l == num_layers - 1 else mlp_dim

			slim_dim1 = [dim1 if l == 0 else int(dim1 * w) for w in self.width_mult_list]
			slim_dim2 = [dim2 if l == num_layers - 1 else int(dim2 * w) for w in self.width_mult_list]

			mlp.append(SlimmableLinear(slim_dim1, slim_dim2, bias=False, width_mult_list=self.width_mult_list,
											overlap_weight=self.overlap_weight))

			if l < num_layers - 1:
				mlp.append(SwitchableBatchNorm1d(slim_dim2, width_mult_list=self.width_mult_list))
				mlp.append(SlimmableReLU(inplace=True))
			elif last_bn:
				# follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
				# for simplicity, we further removed gamma in BN
				mlp.append(SwitchableBatchNorm1d(slim_dim2, width_mult_list=self.width_mult_list, affine=False))

		return nn.Sequential(*mlp)

	def _predictor_forward(self, x_dict, specify_width=None):
		dict_out = dict()
		# predictor will return a tuple like (output, width)
		if specify_width is not None and specify_width in self.width_mult_list:
			dict_out[specify_width] = self.predictor((x_dict[specify_width], specify_width))[0]
		
		else:
			for width in self.width_mult_list:
				dict_out[width] = self.predictor((x_dict[width], width))[0]

		return dict_out

	def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
		pass

	@torch.no_grad()
	def _update_momentum_encoder(self, m):
		"""Momentum update of the momentum encoder"""
		for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
			param_m.data = param_m.data * m + param_b.data * (1. - m)

	def forward(self, im_q, im_k, sweet_x=None, criterion=None, soft_criterion=None, epoch=None):
		"""
		Input:
			im_q: first views of images
			im_k: second views of images
			epoch: the epoch now (in mocov3, it is actually step_now / all_steps)
		Output:
			loss
		"""
		output_dict = {'outputs': None,
						'labels': None,
						'loss': None
						}
		# get the momentum following the cos rule
		m = adjust_moco_momentum(epoch, self.num_epochs, self.init_m) if self.m_cos else self.init_m

		enable_slim = epoch >= self.slim_start_epoch

		if enable_slim:
			# compute query features, a dict of queries: [N, C]
			q1_dict = self._predictor_forward(self.base_encoder(im_q))
			q2_dict = self._predictor_forward(self.base_encoder(im_k))
			for w in self.width_mult_list:
				q1_dict[w] = nn.functional.normalize(q1_dict[w], dim=1)
				q2_dict[w] = nn.functional.normalize(q2_dict[w], dim=1)

		else:
			# only forward the max scale network
			q1_dict = self._predictor_forward(self.base_encoder(im_q, specify_width=self.max_width),
														specify_width=self.max_width)
			q2_dict = self._predictor_forward(self.base_encoder(im_k, specify_width=self.max_width),
														specify_width=self.max_width)
			q1_dict[self.max_width] = nn.functional.normalize(q1_dict[self.max_width], dim=1)
			q2_dict[self.max_width] = nn.functional.normalize(q2_dict[self.max_width], dim=1)
			for w in self.width_mult_list[1:]:
				q1_dict[w] = q1_dict[self.max_width].clone().detach()
				q2_dict[w] = q2_dict[self.max_width].clone().detach()

		# compute key features, no gradient to keys
		with torch.no_grad():
			# update the momentum encoder
			self._update_momentum_encoder(m)  

			# compute momentum features as targets
			k1 = self.momentum_encoder(im_q, self.max_width)[self.max_width]  
			k2 = self.momentum_encoder(im_k, self.max_width)[self.max_width]  
			# normalize
			k1 = nn.functional.normalize(k1, dim=1)
			k2 = nn.functional.normalize(k2, dim=1)

		loss1, d_loss1 = self.ssl_loss(q1_dict, k2, enable_slim=enable_slim, criterion=criterion)
		loss2, d_loss2 = self.ssl_loss(q2_dict, k1, enable_slim=enable_slim, criterion=criterion)

		output_dict['outputs'] = q1_dict
		# fake label, no valuable meaning
		output_dict['labels'] = torch.zeros(k1.shape[0], dtype=torch.long, device=k1.device)
		output_dict['loss'] = loss1 + loss2
		output_dict['all_losses'] = d_loss1

		return output_dict

	def contrastive_loss(self, q, k, labels, criterion=None):
		"""
		the input q & k are normalized,
		q from single device, k from all distributed devices.
		"""
		# Einstein sum is more intuitive
		logits = torch.einsum('nc,mc->nm', [q, k]) / self.T

		loss = criterion(logits, labels) * (2 * self.T)

		return loss, logits

	def ssl_loss(self, q_dict, k, enable_slim=True, criterion=None):
		losses, d_losses, all_logits = [], [], []
		
		# gather all targets
		all_k = concat_all_gather(k)
		# generate labels, different on different GPUs, batch size per GPU
		N = q_dict[self.max_width].shape[0]  
		labels = (torch.arange(N, dtype=torch.long, device=q_dict[self.max_width].device) + 
					N * torch.distributed.get_rank())

		# first calculate the loss of max_width
		max_loss, max_logits = self.contrastive_loss(q_dict[self.max_width], all_k, labels, criterion=criterion)
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
				loss_tmp, logits_tmp = self.contrastive_loss(q_dict[w], all_k, labels, criterion=criterion)

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

				if self.full_distill: raise NotImplementedError
		else:
			# do not train subnetworks
			for w in self.width_mult_list[1:]:
				loss_tmp = torch.zeros_like(max_loss).mean()
				losses.append(loss_tmp)
				d_losses.append(loss_tmp.clone().detach())

		loss = torch.sum(torch.stack(losses))

		return loss, d_losses


class s_MoCoV3_ResNet(s_MoCoV3):
	def _build_projector_and_predictor_mlps(self, dim, mlp_dim):

		### projectors are incorporated into the encoder for S_MoCoV3
		# projectors & remove original fc layer
		# hidden_dim = self.base_encoder.fc.weight.shape[1]
		# del self.base_encoder.fc, self.momentum_encoder.fc
		# self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
		# self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

		# only predictor
		self.predictor = self._build_slimmable_mlp(2, dim, mlp_dim, dim, False)


# class MoCo_ViT(MoCo):
# 	def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
# 		hidden_dim = self.base_encoder.head.weight.shape[1]
# 		del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

# 		# projectors
# 		self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
# 		self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

# 		# predictor
# 		self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


# utils
@torch.no_grad()
def concat_all_gather(tensor):
	"""
	Performs all_gather operation on the provided tensors.
	*** Warning ***: torch.distributed.all_gather has no gradient.
	"""
	tensors_gather = [torch.ones_like(tensor)
		for _ in range(torch.distributed.get_world_size())]
	torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

	output = torch.cat(tensors_gather, dim=0)
	return output


def adjust_moco_momentum(epoch, nums_epochs, init_m):
	"""Adjust moco momentum based on current epoch"""
	m = 1. - 0.5 * (1. + math.cos(math.pi * epoch / nums_epochs)) * (1. - init_m)
	return m
