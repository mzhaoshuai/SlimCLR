# coding=utf-8
import torch
import torch.nn as nn


class SimCLR(nn.Module):
	"""
	Build a SimCLR model
	"""
	def __init__(self, base_encoder, dim=128, T=0.1, mlp=True):
		"""
		dim: feature dimension (default: 128)
		T: softmax temperature (default: 0.1)
		"""
		super(SimCLR, self).__init__()
		self.T = T
		# create the encoders, num_classes is the output fc dimension
		self.encoder_q = base_encoder(num_classes=dim)

		if mlp:
			# https://github.com/google-research/simclr/blob/2fc637bdd6a723130db91b377ac15151e01e4fc2/model_util.py#L141
			dim_mlp = self.encoder_q.fc.weight.shape[1]
			self.encoder_q.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp),
												nn.BatchNorm1d(dim_mlp),
												nn.ReLU(),
												# simclr do not use bias in the second fc layer
												nn.Linear(dim_mlp, dim, bias=False),
												nn.BatchNorm1d(dim))

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
		q1 = self.encoder_q(im_q)
		q2 = self.encoder_q(im_k)
		q1 = nn.functional.normalize(q1, dim=1)
		q2 = nn.functional.normalize(q2, dim=1)

		# gather all tensor
		q1 = all_gather(q1)
		q2 = all_gather(q2)
		# torch.distributed.barrier()

		logits, loss, labels = self.contrastive_loss(q1, q2, criterion=criterion)
		# logit1s, loss1, labels1 = self.contrastive_loss_v2(q1, q2, criterion=criterion)

		output_dict['outputs'] = logits
		output_dict['labels'] = labels
		output_dict['loss'] = loss

		return output_dict

	def contrastive_loss(self, q1, q2, criterion=None):
		# total batch size
		N = q1.shape[0]
		q_all = torch.cat([q1, q2], dim=0)

		# [2N, 2N]
		sim_matrix = torch.matmul(q_all, q_all.T)
		mask = torch.eye(2 * N, dtype=torch.bool, device=q1.device)
		# [2N, 2N - 1], discard the main diagonal
		sim_matrix = sim_matrix[~mask].view(2 * N, -1)

		# labels
		label1 = torch.arange(N, dtype=torch.long, device=q1.device) + N - 1
		label2 = torch.arange(N, dtype=torch.long, device=q1.device)
		labels = torch.cat([label1, label2], dim=0)

		loss = criterion(sim_matrix / self.T, labels)

		return sim_matrix, loss, labels

	def contrastive_loss_v2(self, q1, q2, criterion=None):
		# github.com/sthalles/SimCLR/blob/master/simclr.py
		N = q1.shape[0]
		q_all = torch.cat([q1, q2], dim=0)

		labels = torch.cat([torch.arange(N) for i in range(2)], dim=0)
		labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
		labels = labels.to(q1.device)

		# [2N, 2N]
		sim_matrix = torch.matmul(q_all, q_all.T)
		# discard the main diagonal from both: labels and similarities matrix
		mask = torch.eye(labels.shape[0], dtype=torch.bool, device=q1.device)
		labels = labels[~mask].view(labels.shape[0], -1)
		sim_matrix = sim_matrix[~mask].view(sim_matrix.shape[0], -1)

		# select and combine multiple positives
		positives = sim_matrix[labels.bool()].view(labels.shape[0], -1)

		# select only the negatives the negatives
		negatives = sim_matrix[~labels.bool()].view(sim_matrix.shape[0], -1)

		logits = torch.cat([positives, negatives], dim=1)
		labels = torch.zeros(logits.shape[0], dtype=torch.long, device=q1.device)

		loss = criterion(logits / self.T, labels)

		return logits, loss, labels


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
	model = SimCLR(resnet50)
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
