# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# Adapt for unified interface by shuai

import math
import torch
import logging
import torch.nn as nn


class MoCo(nn.Module):
	"""
	Build a MoCo model with a base encoder, a momentum encoder, and two MLPs
	https://arxiv.org/abs/1911.05722
	"""
	def __init__(self, base_encoder, dim=256, mlp_dim=4096, T=1.0, init_m=0.99, num_epochs=100,
					m_cos=True):
		"""
		dim: feature dimension (default: 256)
		mlp_dim: hidden dimension in MLPs (default: 4096)
		T: softmax temperature (default: 1.0)
		total_e: the number of epochs
		m_cos: whether apply cos increaing momentum
		"""
		super(MoCo, self).__init__()
		assert T > 0.5 and mlp_dim >= 4096 and dim >=256
		self.T = T
		self.init_m = init_m
		self.num_epochs = num_epochs
		self.m_cos = m_cos
		if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
			logging.info('\n[model] MoCoV3, dim {}, mlp_dim {}, T {}, init_m {}, num_epochs {}, m_cos {}'.format
							(dim, mlp_dim, T, init_m, num_epochs, m_cos))

		# build encoders
		self.base_encoder = base_encoder(num_classes=mlp_dim)
		self.momentum_encoder = base_encoder(num_classes=mlp_dim)

		self._build_projector_and_predictor_mlps(dim, mlp_dim)

		for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
			param_m.data.copy_(param_b.data)  # initialize
			param_m.requires_grad = False  # not update by gradient

	def _build_mlp(self, num_layers, input_dim, mlp_dim, output_dim, last_bn=True):
		mlp = []
		for l in range(num_layers):
			dim1 = input_dim if l == 0 else mlp_dim
			dim2 = output_dim if l == num_layers - 1 else mlp_dim

			mlp.append(nn.Linear(dim1, dim2, bias=False))

			if l < num_layers - 1:
				mlp.append(nn.BatchNorm1d(dim2))
				mlp.append(nn.ReLU(inplace=True))
			elif last_bn:
				# follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
				# for simplicity, we further removed gamma in BN
				mlp.append(nn.BatchNorm1d(dim2, affine=False))

		return nn.Sequential(*mlp)

	def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
		pass

	@torch.no_grad()
	def _update_momentum_encoder(self, m):
		"""Momentum update of the momentum encoder"""
		for param_b, param_m in zip(self.base_encoder.parameters(), self.momentum_encoder.parameters()):
			param_m.data = param_m.data * m + param_b.data * (1. - m)

	def contrastive_loss(self, q, k, criterion=None):
		# normalize
		q = nn.functional.normalize(q, dim=1)
		k = nn.functional.normalize(k, dim=1)
		# gather all targets
		k = concat_all_gather(k)
		
		# Einstein sum is more intuitive
		logits = torch.einsum('nc,mc->nm', [q, k]) / self.T
		
		N = logits.shape[0]  # batch size per GPU
		labels = (torch.arange(N, dtype=torch.long, device=logits.device) + 
					N * torch.distributed.get_rank())

		loss = criterion(logits, labels) * (2 * self.T)

		return loss

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
		
		m = adjust_moco_momentum(epoch, self.num_epochs, self.init_m) if self.m_cos \
				else self.init_m

		# compute features
		q1 = self.predictor(self.base_encoder(im_q))
		q2 = self.predictor(self.base_encoder(im_k))

		with torch.no_grad():  # no gradient
			self._update_momentum_encoder(m)  # update the momentum encoder

			# compute momentum features as targets
			k1 = self.momentum_encoder(im_q)
			k2 = self.momentum_encoder(im_k)

		loss = (self.contrastive_loss(q1, k2, criterion=criterion) + 
					self.contrastive_loss(q2, k1, criterion=criterion))

		output_dict['outputs'] = (q1 + q2) / 2
		output_dict['labels'] = torch.arange(q1.shape[0], dtype=torch.long, device=q1.device) 
		output_dict['loss'] = loss

		return output_dict


class MoCo_ResNet(MoCo):
	def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
		hidden_dim = self.base_encoder.fc.weight.shape[1]
		del self.base_encoder.fc, self.momentum_encoder.fc # remove original fc layer

		# projectors
		self.base_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)
		self.momentum_encoder.fc = self._build_mlp(2, hidden_dim, mlp_dim, dim)

		# predictor
		self.predictor = self._build_mlp(2, dim, mlp_dim, dim, False)


class MoCo_ViT(MoCo):
	def _build_projector_and_predictor_mlps(self, dim, mlp_dim):
		hidden_dim = self.base_encoder.head.weight.shape[1]
		del self.base_encoder.head, self.momentum_encoder.head # remove original fc layer

		# projectors
		self.base_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)
		self.momentum_encoder.head = self._build_mlp(3, hidden_dim, mlp_dim, dim)

		# predictor
		self.predictor = self._build_mlp(2, dim, mlp_dim, dim)


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
