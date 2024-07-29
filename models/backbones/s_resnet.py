# coding=utf-8
"""
A slimmable resnet model
"""
import torch
import logging
import torch.nn as nn

from .slimmable_ops import SlimmableConv2d, SlimmableLinear, SlimmableReLU
from .slimmable_ops import SwitchableBatchNorm1d, SwitchableBatchNorm2d, SwitchableLinear


act_inplace = True


class Block(nn.Module):
	# Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
	# while original implementation places the stride at the first 1x1 convolution(self.conv1)
	# according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
	# This variant is also known as ResNet V1.5 and improves accuracy according to
	# https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.
	# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py

	def __init__(self, inp, outp, stride, width_mult_list=[1.0,], overlap_weight=True):
		super(Block, self).__init__()
		assert stride in [1, 2]
		midp = [i // 4 for i in outp]

		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = SlimmableConv2d(inp, midp, 1, 1, 0, bias=False, width_mult_list=width_mult_list,
										overlap_weight=overlap_weight)
		self.bn1 = SwitchableBatchNorm2d(midp, width_mult_list=width_mult_list)
		self.conv2 = SlimmableConv2d(midp, midp, 3, stride, 1, bias=False, width_mult_list=width_mult_list,
										overlap_weight=overlap_weight)
		self.bn2 = SwitchableBatchNorm2d(midp, width_mult_list=width_mult_list)
		self.conv3 = SlimmableConv2d(midp, outp, 1, 1, 0, bias=False, width_mult_list=width_mult_list,
										overlap_weight=overlap_weight)
		self.bn3 = SwitchableBatchNorm2d(outp, width_mult_list=width_mult_list)
		self.relu = SlimmableReLU(inplace=act_inplace)

		self.residual_connection = (stride != 1 or inp != outp)
		if self.residual_connection:
			self.shortcut = nn.Sequential(
				SlimmableConv2d(inp, outp, 1, stride=stride, bias=False, width_mult_list=width_mult_list,
										overlap_weight=overlap_weight),
				SwitchableBatchNorm2d(outp, width_mult_list=width_mult_list),
			)

	def forward(self, x_tuple: tuple):
		"""
		Args:
			x_tuple: (x, width_mult)
		Return:
			a tuple: (x, width_mult)
		"""
		x, width_mult = x_tuple
		identity = x

		out = self.conv1(x_tuple)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		out = self.conv3(out)
		out = self.bn3(out)

		if self.residual_connection:
			identity = self.shortcut(x_tuple)[0]
		
		res = (identity + out[0], width_mult)
		res = self.relu(res)
		
		return res


class Model(nn.Module):
	def __init__(self, num_classes=1000, input_size=224, args=None):
		"""
		A resnet model
		"""
		super(Model, self).__init__()
		assert input_size % 32 == 0
		self.slim_fc = getattr(args, 'slim_fc', 'supervised')
		self.overlap_weight = getattr(args, 'overlap_weight', True)
		self.moco_mlp_dim = getattr(args, 'moco_mlp_dim', 4096)
		self.simclr_mlp_dim = getattr(args, 'simclr_mlp_dim', 2048)

		# setting of inverted residual blocks
		self.block_setting_dict = {
			# : [stage1, stage2, stage3, stage4]
			50: [3, 4, 6, 3],
			101: [3, 4, 23, 3],
			152: [3, 8, 36, 3],
		}
		feats = [64, 128, 256, 512]
		self.block_setting = self.block_setting_dict[args.depth]
		self.width_mult_list = sorted(args.width_mult_list, reverse=True)
		self.max_width, self.min_width = self.width_mult_list[0], self.width_mult_list[-1]
		if not self.overlap_weight: assert sum(self.width_mult_list) <= 2 * self.max_width

		if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
			logging.info("[model] s_ResNet:\n"
							"\t width_mult_list: {}\n"
							"\t overlap_weight: {}\n"
							"\t slim_fc / moco_mlp_dim / simclr_mlp_dim: [{} / {} / {}]\n"
							"\t distill / distill_mixed: [{} / {}]".format(
								self.width_mult_list, self.overlap_weight,
								self.slim_fc, self.moco_mlp_dim, self.simclr_mlp_dim,
								args.inplace_distill, args.inplace_distill_mixed))

		# 1. stem, a 7x7 conv
		channels = [int(64 * width_mult) for width_mult in self.width_mult_list]
		self.conv1 = SlimmableConv2d([3 for _ in range(len(channels))], channels,
										kernel_size=7, stride=2, padding=3, bias=False,
										width_mult_list=self.width_mult_list, overlap_weight=self.overlap_weight)
		self.bn1 = SwitchableBatchNorm2d(channels, width_mult_list=self.width_mult_list)
		self.relu = SlimmableReLU(inplace=act_inplace)
		self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

		# 2. backbone, 4 resnet bottleneck blocks
		features = []
		expansion = 4
		for stage_id, n in enumerate(self.block_setting):
			# output channles for 4 blocks
			outp = [int(feats[stage_id] * width_mult * expansion) for width_mult in self.width_mult_list]

			for i in range(n):
				features.append(Block(channels, outp, 2 if (i == 0 and stage_id != 0) else 1,
										width_mult_list=self.width_mult_list, overlap_weight=self.overlap_weight))
				channels = outp
		self.features = nn.Sequential(*features)

		# 3. classifier, a fc layer
		self.outp = channels
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		if self.slim_fc == 'mocov2_slim':
			self.fc = nn.Sequential(SlimmableLinear(self.outp, self.outp,
										width_mult_list=self.width_mult_list, overlap_weight=self.overlap_weight),
									SlimmableReLU(inplace=act_inplace),
									SlimmableLinear(self.outp, [num_classes for _ in range(len(self.outp))],
										width_mult_list=self.width_mult_list, overlap_weight=self.overlap_weight)
									)

		elif self.slim_fc == 'mocov2_switch':
			self.fc = nn.Sequential(SwitchableLinear(self.outp, self.outp,
										width_mult_list=self.width_mult_list),
									SlimmableReLU(inplace=act_inplace),
									SwitchableLinear(self.outp, [num_classes for _ in range(len(self.outp))],
										width_mult_list=self.width_mult_list)
									)

		elif self.slim_fc == 'mocov3_slim':
			mlp_channels = [int(self.moco_mlp_dim * width_mult) for width_mult in self.width_mult_list]
			self.fc = nn.Sequential(SlimmableLinear(self.outp, mlp_channels, bias=False,
										width_mult_list=self.width_mult_list, overlap_weight=self.overlap_weight),
									SwitchableBatchNorm1d(mlp_channels, self.width_mult_list),
									SlimmableReLU(inplace=act_inplace),
									SlimmableLinear(mlp_channels, [num_classes for _ in range(len(self.outp))],
										bias=False,
										width_mult_list=self.width_mult_list, overlap_weight=self.overlap_weight),
									# follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
									# for simplicity, we further removed gamma in BN
									SwitchableBatchNorm1d([num_classes for _ in range(len(self.outp))],
										self.width_mult_list, affine=False)
									)

		elif self.slim_fc == 'simclr_slim':
			mlp_channels = [int(self.simclr_mlp_dim * width_mult) for width_mult in self.width_mult_list]
			self.fc = nn.Sequential(SlimmableLinear(self.outp, mlp_channels,
										width_mult_list=self.width_mult_list, overlap_weight=self.overlap_weight),
									SwitchableBatchNorm1d(mlp_channels, self.width_mult_list),
									SlimmableReLU(inplace=act_inplace),
									SlimmableLinear(mlp_channels, [num_classes for _ in range(len(self.outp))],
										bias=False,
										width_mult_list=self.width_mult_list, overlap_weight=self.overlap_weight),
									SwitchableBatchNorm1d([num_classes for _ in range(len(self.outp))],
										self.width_mult_list)
									)

		elif self.slim_fc == 'supervised':
			self.fc = SlimmableLinear(self.outp, [num_classes for _ in range(len(self.outp))],
										width_mult_list=self.width_mult_list, overlap_weight=self.overlap_weight)

		elif self.slim_fc == 'supervised_switch':
			self.fc = SwitchableLinear(self.outp, [num_classes for _ in range(len(self.outp))],
										width_mult_list=self.width_mult_list)

		else:
			raise NotImplementedError("fc arch {} is not supported.".format(self.slim_fc))

		# 4. parameter initialization
		self.reset_parameters()

	def forward_single(self, x_tuple: tuple):
		"""
		Args:
			x_tuple: (x, width_mult)
		Return:
			output tensor
		"""
		width_mult = x_tuple[1]
		out = self.conv1(x_tuple)
		out = self.bn1(out)
		out = self.relu(out)
		o_x = self.maxpool(out[0])

		out = self.features((o_x, width_mult))

		o_x = self.avgpool(out[0])
		o_x = torch.flatten(o_x, 1)

		out = self.fc((o_x, width_mult))

		return out[0]

	def forward(self, x, specify_width=None, sweet_x=None):
		"""
		sweet_x: a dict {width -> input sample}
		"""
		dict_out = dict()

		if specify_width is not None and specify_width in self.width_mult_list:
			dict_out[specify_width] = self.forward_single((x, specify_width))
		else:
			for width in self.width_mult_list:
				if width == self.max_width:
					dict_out[width] = self.forward_single((x, width))
				else:
					if sweet_x is not None:
						dict_out[width] = self.forward_single((sweet_x[width], width))
					else:
						dict_out[width] = self.forward_single((x, width))

		return dict_out

	@torch.no_grad()
	def reset_parameters(self, zero_init_residual: bool = True):
		# initialization in the below link
		# https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
				if m.bias is not None:
					m.bias.data.zero_()
			elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
				if m.affine:
					nn.init.constant_(m.weight, 1.0)
					nn.init.constant_(m.bias, 0.0)
			elif isinstance(m, nn.Linear):
				m.weight.data.normal_(mean=0.0, std=0.01)
				if m.bias is not None:
					m.bias.data.zero_()
		# Zero-initialize the last BN in each residual branch,
		# so that the residual branch starts with zeros, and each residual block behaves like an identity.
		# This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
		if zero_init_residual:
			for n, m in self.named_modules():
				if 'bn3.bn.' in n:
					nn.init.constant_(m.weight, 0) 
					# print(n)
