# coding=utf-8
"""
Slimmabel networks with SimCLR
"""
import torch
import logging
import torch.nn as nn
from .distiller import choose_kd_loss
from .utils import loss_weights_generate


class s_SimCLR(nn.Module):
	"""
	Build a SimCLR model with slimmable networks
	"""
	def __init__(self, base_encoder, dim=128, T=0.1, mlp=True, args=None):
		"""
		dim: feature dimension (default: 128)
		T: softmax temperature (default: 0.1)
		"""
		super(s_SimCLR, self).__init__()
		self.T = T
		self.num_epochs = args.num_epochs
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

		if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
			logging.info("[model] s_SimCLR:\n"
							"\t feature dim / mlp_dim: [ {} / {} ]\n"
							"\t num_epochs / temperature T: [ {} / {}]\n"
							"\t width_mult_list: {}\n"
							"\t loss_weights / slim_start_epoch / full_distill: [ {} / {} / {} ]\n"
							"\t aux_loss_w / teacher_T / student_T: [ {} / {} / {} ]\n"
							"\t KD / SEED / MSE / ATKD / DKD: [ {} / {} / {} / {} / {} ]".format(
							dim, args.simclr_mlp_dim, self.num_epochs, self.T,
							self.width_mult_list, self.loss_weights,
							self.slim_start_epoch, self.full_distill, self.aux_loss_w, self.tea_T, self.stu_T,
							self.inplace_distill, self.seed_loss, self.mse_loss, self.atkd_loss, self.dkd_loss))

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
		enable_slim = epoch >= self.slim_start_epoch

		if enable_slim:
			# forward all networks at different scales
			q1_dict = self.encoder_q(im_q)
			q2_dict = self.encoder_q(im_k)
			for w in self.width_mult_list:
				q1_dict[w] = nn.functional.normalize(q1_dict[w], dim=1)
				q2_dict[w] = nn.functional.normalize(q2_dict[w], dim=1)
		else:
			# only forward the max scale network
			q1_dict = self.encoder_q(im_q, specify_width=self.max_width)
			q2_dict = self.encoder_q(im_k, specify_width=self.max_width)			
			q1_dict[self.max_width] = nn.functional.normalize(q1_dict[self.max_width], dim=1)
			q2_dict[self.max_width] = nn.functional.normalize(q2_dict[self.max_width], dim=1)
			for w in self.width_mult_list[1:]:
				q1_dict[w] = q1_dict[self.max_width].clone().detach()
				q2_dict[w] = q2_dict[self.max_width].clone().detach()

		# gather all tensor across gpus
		for w in self.width_mult_list:
			q1_dict[w] = all_gather(q1_dict[w])
			q2_dict[w] = all_gather(q2_dict[w])
		# torch.distributed.barrier()

		# labels
		N = q1_dict[self.max_width].shape[0]
		label1 = torch.arange(N, dtype=torch.long, device=q1_dict[self.max_width].device) + N - 1
		label2 = torch.arange(N, dtype=torch.long, device=q1_dict[self.max_width].device)
		labels = torch.cat([label1, label2], dim=0)

		loss, d_losses, output_logits = self.ssl_loss(q1_dict, q2_dict, labels, enable_slim=enable_slim, criterion=criterion)

		output_dict['outputs'] = output_logits
		output_dict['labels'] = labels
		output_dict['loss'] = loss
		output_dict['all_losses'] = d_losses

		return output_dict

	def contrastive_loss(self, q1, q2, labels, criterion=None):
		# total batch size
		N = q1.shape[0]
		q_all = torch.cat([q1, q2], dim=0)

		# [2N, 2N]
		sim_matrix = torch.matmul(q_all, q_all.T)
		mask = torch.eye(2 * N, dtype=torch.bool, device=q1.device)
		# [2N, 2N - 1], discard the main diagonal
		sim_matrix = sim_matrix[~mask].view(2 * N, -1)

		loss = criterion(sim_matrix / self.T, labels)

		return sim_matrix, loss

	def ssl_loss(self, q1_dict, q2_dict, labels, enable_slim=True, criterion=None):
		losses, d_losses, all_logits = [], [], []
		output_logits = dict()

		# first calculate the loss of max_width
		max_logits, max_loss = self.contrastive_loss(q1_dict[self.max_width], q2_dict[self.max_width], labels, criterion=criterion)
		max_loss = max_loss * self.loss_weights[0] if enable_slim else max_loss
		losses.append(max_loss)
		d_losses.append(max_loss.clone().detach())
		output_logits[self.max_width] = max_logits

		if enable_slim:
			# subnetworks are able to learn
			if self.mse_loss == 2 or self.dkd_loss or self.atkd_loss or self.seed_loss:
				all_logits.append(max_logits.clone().detach())

			for index, w in enumerate(self.width_mult_list[1:]):
				# contrastive loss
				logits_tmp, loss_tmp = self.contrastive_loss(q1_dict[w], q2_dict[w], labels, criterion=criterion)

				aux_loss = torch.zeros_like(loss_tmp).mean()
				denominator_ = 0.0
				for soft_target in all_logits:

					if self.inplace_distill or self.mse_loss == 1:
						raise NotImplementedError("do not support operations on the output of backbones")

					elif self.mse_loss == 2 or self.seed_loss or self.atkd_loss or self.dkd_loss:
						aux_loss_tmp = choose_kd_loss(logits_tmp, soft_target, stu_T=self.stu_T, tea_T=self.tea_T,
														labels=labels,
														mse=self.mse_loss, kl_div=self.seed_loss,
														dkd=self.dkd_loss, atkd=self.atkd_loss)						
					else:
						aux_loss_tmp = torch.zeros_like(loss_tmp).mean()

					denominator_ += 1.0
					aux_loss = aux_loss + aux_loss_tmp

				if denominator_ > 0: aux_loss = aux_loss / denominator_

				# the loss of this subnetwork
				loss_tmp = self.loss_weights[index + 1] * ((1.0 - self.aux_loss_w) * loss_tmp + self.aux_loss_w * aux_loss)

				losses.append(loss_tmp)
				d_losses.append(loss_tmp.clone().detach())
				output_logits[w] = logits_tmp

		else:
			# do not train subnetworks
			for w in self.width_mult_list[1:]:
				loss_tmp = torch.zeros_like(max_loss).mean()
				losses.append(loss_tmp)
				d_losses.append(loss_tmp.clone().detach())		
				output_logits[w] = max_logits.clone().detach()	

		loss = torch.sum(torch.stack(losses))

		return loss, d_losses, output_logits


def all_gather(tensor):
	"""gather a tensor from all devices in distributed training"""
	if torch.distributed.is_available() and torch.distributed.is_initialized():
		world_size = torch.distributed.get_world_size()
		rank = torch.distributed.get_rank()

		# We gather tensors from all gpus to get more negatives to contrast with.
		gathered_tensor = [torch.zeros_like(tensor) for _ in range(world_size)]
		torch.distributed.all_gather(gathered_tensor, tensor)
		gathered_tensor[rank] = tensor

		return torch.cat(gathered_tensor, dim=0)

	else:
		return tensor


if __name__ == "__main__":
	from torchvision.models import resnet50
	model = s_SimCLR(resnet50)
	x1 = torch.rand(2, 3, 224, 224)
	x2 = torch.rand(2, 3, 224, 224)

	od = model(x1, x2, criterion=torch.nn.CrossEntropyLoss())
	print(od['loss'])
	print(od['loss1'])
	"""
	# 1
	tensor(1.0808, grad_fn=<NllLossBackward>)
	tensor(1.0808, grad_fn=<NllLossBackward>)

	# 2
	tensor(1.0275, grad_fn=<NllLossBackward>)
	tensor(1.0275, grad_fn=<NllLossBackward>)
	"""
