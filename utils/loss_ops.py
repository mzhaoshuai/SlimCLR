# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CrossEntropyLossSoft(torch.nn.modules.loss._Loss):
	# https://pytorch.org/docs/stable/_modules/torch/nn/modules/loss.html#KLDivLoss
	"""
	The Kullback-Leibler divergence loss.
	inplace distillation for image classification
	"""
	def __init__(self, reduction: str = 'batchmean') -> None:
		super().__init__()
		self.log_target = False
		self.reduction = reduction

	# def forward(self, input: Tensor, target: Tensor) -> Tensor:
	# 	input = F.log_softmax(input, dim=1)
	# 	return F.kl_div(input, target, reduction=self.reduction, log_target=self.log_target)

	def forward(self, output, target):
		output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
		kldiv = - torch.sum(output_log_prob * target, dim=1).mean()

		return kldiv


class CrossEntropyLossSmooth(torch.nn.modules.loss._Loss):
	""" label smooth """
	def forward(self, output, target, label_smoothing=0.1):
		eps = label_smoothing
		n_class = output.size(1)
		one_hot = torch.zeros_like(output).scatter(1, target.view(-1, 1), 1)
		target = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
		output_log_prob = torch.nn.functional.log_softmax(output, dim=1)
		target = target.unsqueeze(1)
		output_log_prob = output_log_prob.unsqueeze(2)
		cross_entropy_loss = -torch.bmm(target, output_log_prob)

		return cross_entropy_loss


class label_smoothing_CE(nn.Module):
	""" label smooth CE loss"""
	def __init__(self, e=0.1, reduction='mean'):
		super(label_smoothing_CE, self).__init__()

		self.log_softmax = nn.LogSoftmax(dim=-1)
		self.e = e
		self.reduction = reduction

	def forward(self, x, target):

		if x.size(0) != target.size(0):
			raise ValueError('Expected input batchsize ({}) to match target batch_size({})'
					.format(x.size(0), target.size(0)))

		if x.dim() < 2:
			raise ValueError('Expected input tensor to have least 2 dimensions(got {})'
					.format(x.size(0)))

		num_classes = x.size(-1)
		one_hot = F.one_hot(target, num_classes=num_classes).type_as(x)
		smoothed_target = one_hot * (1.0 - self.e) + (1.0 - one_hot) * self.e / (num_classes - 1)

		# negative log likelihood
		log_probs = self.log_softmax(x)
		loss = torch.sum(- log_probs * smoothed_target, dim=-1)

		if self.reduction == 'none':
			return loss
		elif self.reduction == 'sum':
			return torch.sum(loss)
		elif self.reduction == 'mean':
			return torch.mean(loss)
		else:
			raise ValueError('unrecognized option, expect reduction to be one of none, mean, sum')


if __name__ == "__main__":
	output = torch.rand(128, 100)
	output.requires_grad = True

	target = F.softmax(torch.rand(128, 100), dim=1)

	loss1 = F.kl_div(F.log_softmax(output, dim=1), target, reduction='batchmean', log_target=False)
	loss1.backward()
	print(output.grad.sum())

	output.grad.zero_()
	cr = CrossEntropyLossSoft()
	loss2 = cr(output, target).mean()
	loss2.backward()
	print(output.grad.sum())

	print(loss1)
	print(loss2)
	print('difference {}'.format(loss2 - loss1))
	