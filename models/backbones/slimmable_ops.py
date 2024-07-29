# coding=utf-8
import torch.nn as nn
FLAGS = None

## ####################################
# Ops for slimmable neural networks
## ####################################


class SwitchableBatchNorm1d(nn.Module):
	def __init__(self, num_features_list, width_mult_list=[1.0,], affine=True):
		"""
		switchable BN for slimmable networks
		"""
		super(SwitchableBatchNorm1d, self).__init__()
		self.num_features_list = num_features_list
		self.num_features = max(num_features_list)
		bns = []
		for i in num_features_list:
			bns.append(nn.BatchNorm1d(i, affine=affine))
		self.bn = nn.ModuleList(bns)
		self.width_mult = max(width_mult_list)
		self.ignore_model_profiling = True
		self.width_mult_list = width_mult_list

	def forward(self, x_tuple: tuple):
		x, width_mult = x_tuple
		idx = self.width_mult_list.index(width_mult)
		y = self.bn[idx](x)

		return (y, width_mult)


class SwitchableBatchNorm2d(nn.Module):
	def __init__(self, num_features_list, width_mult_list=[1.0,], affine=True):
		"""
		switchable BN for slimmable networks
		"""
		super(SwitchableBatchNorm2d, self).__init__()
		self.num_features_list = num_features_list
		self.num_features = max(num_features_list)
		bns = []
		for i in num_features_list:
			bns.append(nn.BatchNorm2d(i, affine=affine))
		self.bn = nn.ModuleList(bns)
		self.width_mult = max(width_mult_list)
		self.ignore_model_profiling = True
		self.width_mult_list = width_mult_list

	def forward(self, x_tuple: tuple):
		x, width_mult = x_tuple
		idx = self.width_mult_list.index(width_mult)
		y = self.bn[idx](x)

		return (y, width_mult)


class SwitchableLinear(nn.Module):
	def __init__(self, in_features_list, out_features_list, bias=True,
						width_mult_list=[1.0,]):
		"""
		switchable Linear for slimmable networks, only Linear, no bn
		"""
		super(SwitchableLinear, self).__init__()
		lins = []
		for in_f, out_f in zip(in_features_list, out_features_list):
			lins.append(nn.Linear(in_f, out_f, bias=bias))
		self.linears = nn.ModuleList(lins)
		self.width_mult_list = width_mult_list

	def forward(self, x_tuple: tuple):
		x, width_mult = x_tuple
		idx = self.width_mult_list.index(width_mult)
		y = self.linears[idx](x)

		return (y, width_mult)


class SlimmableConv2d(nn.Conv2d):
	def __init__(self, in_channels_list, out_channels_list,
				kernel_size=3, stride=1, padding=0, dilation=1,
				groups_list=[1], bias=True, width_mult_list=[1.0,],
				overlap_weight=True):
		"""
		Slimmable Conv2d Layer
		Args:
			overlap_weight: If False, weights of subnetworks are not overlapped
		"""
		super(SlimmableConv2d, self).__init__(
			max(in_channels_list), max(out_channels_list),
			kernel_size, stride=stride, padding=padding, dilation=dilation,
			groups=max(groups_list), bias=bias)
		
		self.in_channels_list = in_channels_list
		self.out_channels_list = out_channels_list
		self.groups_list = groups_list
		if self.groups_list == [1]:
			self.groups_list = [1 for _ in range(len(in_channels_list))]
		self.width_mult = max(width_mult_list)
		self.width_mult_list = width_mult_list
		self.overlap_weight = overlap_weight

		if not self.overlap_weight:
			self.in_c = get_nonoverlap_channels(in_channels_list)
			self.out_c = get_nonoverlap_channels(out_channels_list)

	def forward(self, x_tuple: tuple):
		x, width_mult = x_tuple
		idx = self.width_mult_list.index(width_mult)
		self.groups = self.groups_list[idx]

		if self.overlap_weight:
			self.in_channels = self.in_channels_list[idx]
			self.out_channels = self.out_channels_list[idx]
			weight = self.weight[:self.out_channels, :self.in_channels, :, :]
			if self.bias is not None:
				bias = self.bias[:self.out_channels]
			else:
				bias = self.bias
		else:
			out_start, out_end = self.out_c[idx]
			in_start, in_end = self.in_c[idx]
			weight = self.weight[out_start:out_end, in_start:in_end, :, :]
			if self.bias is not None:
				bias = self.bias[out_start:out_end]
			else:
				bias = self.bias

		y = nn.functional.conv2d(x, weight, bias, self.stride,
									self.padding, self.dilation, self.groups)

		return (y, width_mult)


class SlimmableLinear(nn.Linear):
	def __init__(self, in_features_list, out_features_list, bias=True,
						width_mult_list=[1.0,], overlap_weight=True):
		"""
		Slimmable Linear Layer
		Args:
			overlap_weight: If False, weights of subnetworks are not overlapped
		"""
		super(SlimmableLinear, self).__init__(max(in_features_list), max(out_features_list), bias=bias)
		self.in_features_list = in_features_list
		self.out_features_list = out_features_list
		self.width_mult = max(width_mult_list)
		self.width_mult_list = width_mult_list
		
		self.overlap_weight = overlap_weight
		if not self.overlap_weight:
			self.in_c = get_nonoverlap_channels(in_features_list)
			self.out_c = get_nonoverlap_channels(out_features_list)

	def forward(self, x_tuple: tuple):
		x, width_mult = x_tuple
		idx = self.width_mult_list.index(width_mult)

		if self.overlap_weight:
			self.in_features = self.in_features_list[idx]
			self.out_features = self.out_features_list[idx]
			weight = self.weight[:self.out_features, :self.in_features]
			
			if self.bias is not None:
				bias = self.bias[:self.out_features]
			else:
				bias = self.bias
		else:
			out_start, out_end = self.out_c[idx]
			in_start, in_end = self.in_c[idx]
			weight = self.weight[out_start:out_end, in_start:in_end]

			if self.bias is not None:
				bias = self.bias[out_start:out_end]
			else:
				bias = self.bias

		y = nn.functional.linear(x, weight, bias)

		return (y, width_mult)


class SlimmableReLU(nn.Module):
	def __init__(self, inplace: bool = False):
		super(SlimmableReLU, self).__init__()
		self.inplace = inplace

	def forward(self, x_tuple: tuple):
		x, width_mult = x_tuple
		y = nn.functional.relu(x, inplace=self.inplace)

		return (y, width_mult)


class SlimmableIdentity(nn.Module):
	def __init__(self):
		super(SlimmableIdentity, self).__init__()

	def forward(self, x_tuple: tuple):

		return x_tuple


def get_nonoverlap_channels(channels):
	"""
	Args:
		channels: in descending order, e.g., [128, 64, 32]
	Returns:
		index of channels, e.g., [[0, 128], [32, 96], [0, 32]]
	"""
	nums_c = len(channels)
	ascending_c = sorted(channels, reverse=False)
	if len(set(ascending_c)) == 1:
		# the fisrt conv has the same in channel
		index_c = [[0, c] for c in ascending_c]
	else:
		index_c = []
		accumulated_c = 0
		for i in range(nums_c):
			if i == nums_c - 1:
				index_c.append([0, ascending_c[-1]])
			else:
				index_c.append([accumulated_c, accumulated_c + ascending_c[i]])
				
				accumulated_c += ascending_c[i]

	return list(reversed(index_c))


## ####################################
# Ops for universal slimmable neural networks
## ####################################

def make_divisible(v, divisor=8, min_value=1):
	"""
	forked from slim:
	https://github.com/tensorflow/models/blob/\
	0344c5503ee55e24f0de7f37336a6e08f10976fd/\
	research/slim/nets/mobilenet/mobilenet.py#L62-L69
	"""
	if min_value is None:
		min_value = divisor
	new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
	# Make sure that round down does not go down by more than 10%.
	if new_v < 0.9 * v:
		new_v += divisor
	return new_v


class USConv2d(nn.Conv2d):
	def __init__(self, in_channels, out_channels, kernel_size, stride=1,
				 padding=0, dilation=1, groups=1, depthwise=False, bias=True,
				 us=[True, True], ratio=[1, 1]):
		super(USConv2d, self).__init__(
			in_channels, out_channels,
			kernel_size, stride=stride, padding=padding, dilation=dilation,
			groups=groups, bias=bias)
		self.depthwise = depthwise
		self.in_channels_max = in_channels
		self.out_channels_max = out_channels
		self.width_mult = None
		self.us = us
		self.ratio = ratio

	def forward(self, input):
		if self.us[0]:
			self.in_channels = make_divisible(
				self.in_channels_max
				* self.width_mult
				/ self.ratio[0]) * self.ratio[0]
		if self.us[1]:
			self.out_channels = make_divisible(
				self.out_channels_max
				* self.width_mult
				/ self.ratio[1]) * self.ratio[1]
		self.groups = self.in_channels if self.depthwise else 1
		weight = self.weight[:self.out_channels, :self.in_channels, :, :]
		if self.bias is not None:
			bias = self.bias[:self.out_channels]
		else:
			bias = self.bias
		y = nn.functional.conv2d(
			input, weight, bias, self.stride, self.padding,
			self.dilation, self.groups)
		if getattr(FLAGS, 'conv_averaged', False):
			y = y * (max(self.in_channels_list) / self.in_channels)
		return y


class USLinear(nn.Linear):
	def __init__(self, in_features, out_features, bias=True, us=[True, True]):
		super(USLinear, self).__init__(
			in_features, out_features, bias=bias)
		self.in_features_max = in_features
		self.out_features_max = out_features
		self.width_mult = None
		self.us = us

	def forward(self, input):
		if self.us[0]:
			self.in_features = make_divisible(
				self.in_features_max * self.width_mult)
		if self.us[1]:
			self.out_features = make_divisible(
				self.out_features_max * self.width_mult)
		weight = self.weight[:self.out_features, :self.in_features]
		if self.bias is not None:
			bias = self.bias[:self.out_features]
		else:
			bias = self.bias
		return nn.functional.linear(input, weight, bias)


class USBatchNorm2d(nn.BatchNorm2d):
	def __init__(self, num_features, ratio=1):
		super(USBatchNorm2d, self).__init__(
			num_features, affine=True, track_running_stats=False)
		self.num_features_max = num_features
		# for tracking performance during training
		self.bn = nn.ModuleList([
			nn.BatchNorm2d(i, affine=False) for i in [
				make_divisible(
					self.num_features_max * width_mult / ratio) * ratio
				for width_mult in FLAGS.width_mult_list]])
		self.ratio = ratio
		self.width_mult = None
		self.ignore_model_profiling = True

	def forward(self, input):
		weight = self.weight
		bias = self.bias
		c = make_divisible(
			self.num_features_max * self.width_mult / self.ratio) * self.ratio
		if self.width_mult in FLAGS.width_mult_list:
			idx = FLAGS.width_mult_list.index(self.width_mult)
			y = nn.functional.batch_norm(
				input,
				self.bn[idx].running_mean[:c],
				self.bn[idx].running_var[:c],
				weight[:c],
				bias[:c],
				self.training,
				self.momentum,
				self.eps)
		else:
			y = nn.functional.batch_norm(
				input,
				self.running_mean,
				self.running_var,
				weight[:c],
				bias[:c],
				self.training,
				self.momentum,
				self.eps)
		return y


def pop_channels(autoslim_channels):
	return [i.pop(0) for i in autoslim_channels]


def bn_calibration_init(m):
	""" calculating post-statistics of batch normalization """
	if getattr(m, 'track_running_stats', False):
		# reset all values for post-statistics
		m.reset_running_stats()
		# set bn in training mode to update post-statistics
		m.training = True
		# if use cumulative moving average
		if getattr(FLAGS, 'cumulative_bn_stats', False):
			m.momentum = None
