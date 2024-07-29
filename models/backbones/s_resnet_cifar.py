# coding=utf-8
"""
A slimmable resnet model for CIFAR dataset
"""
import torch
import logging
import torch.nn as nn
from .slimmable_ops import SwitchableBatchNorm2d
from .slimmable_ops import SlimmableConv2d, SlimmableLinear, SlimmableReLU

act_inplace = True


class BasicBlock(nn.Module):
	def __init__(self, inp, outp, stride, width_mult_list=[1.0,], overlap_weight=True):
		super(BasicBlock, self).__init__()
		assert stride in [1, 2]

		# Both self.conv2 and self.downsample layers downsample the input when stride != 1
		self.conv1 = SlimmableConv2d(inp, outp, 3, stride, 1, bias=False, width_mult_list=width_mult_list)
		self.bn1 = SwitchableBatchNorm2d(outp, width_mult_list=width_mult_list)
		self.conv2 = SlimmableConv2d(outp, outp, 3, 1, 1, bias=False, width_mult_list=width_mult_list)
		self.bn2 = SwitchableBatchNorm2d(outp, width_mult_list=width_mult_list)
		self.relu = SlimmableReLU(inplace=act_inplace)

		self.residual_connection = (stride != 1 or inp != outp)
		if self.residual_connection:
			self.shortcut = nn.Sequential(
				SlimmableConv2d(inp, outp, 1, stride=stride, bias=False, width_mult_list=width_mult_list),
				SwitchableBatchNorm2d(outp, width_mult_list=width_mult_list),
			)

	def forward(self, x_tuple: tuple):
		"""
		Args: x_tuple: (x, width_mult)
		Return: a tuple: (x, width_mult)
		"""
		x, width_mult = x_tuple
		identity = x

		out = self.conv1(x_tuple)
		out = self.bn1(out)
		out = self.relu(out)

		out = self.conv2(out)
		out = self.bn2(out)
		out = self.relu(out)

		if self.residual_connection:
			identity = self.shortcut(x_tuple)[0]
		
		res = (identity + out[0], width_mult)
		res = self.relu(res)
		
		return res


class s_ResNet_CIFAR(nn.Module):
	def __init__(self, num_classes=10, input_size=32, width_mult_list=[1.0, 0.5], args=None, depth=None, width=None):
		"""
		A resnet model
		"""
		super(s_ResNet_CIFAR, self).__init__()
		assert input_size % 16 == 0
		self.slim_fc = getattr(args, 'slim_fc', 'supervised')
		self.overlap_weight = True
		self.depth = getattr(args, 'depth', 20) if depth is None else depth
		self.width = getattr(args, 'width', 1) if width is None else width
		self.width_mult_list = sorted(width_mult_list, reverse=True)

		# setting of inverted residual blocks
		self.block_setting_dict = {
			# : [stage1, stage2, stage3]
			20: [3, 3, 3],
			32: [5, 5, 5],
			44: [7, 7, 7],
			56: [9, 9, 9],
			110: [18, 18, 18],
		}
		self.feats =[int(self.width  * x) for x in [16, 32, 64]]
		self.block_setting = self.block_setting_dict[self.depth]
		self.max_width, self.min_width = self.width_mult_list[0], self.width_mult_list[-1]

		print("[model] s_ResNet_CIFAR:\n"
						"\t depth \ width \ feats: [{} \ {} \ {}]\n"
						"\t width_mult_list: {}\n"
						"\t overlap_weight: {}\n"
						"\t slim_fc : {}\n".format(self.depth, self.width, self.feats,
													self.width_mult_list, self.overlap_weight, self.slim_fc))

		# 1. stem, a 3x3 conv
		channels = [int(self.feats[0] * width_mult) for width_mult in self.width_mult_list]
		self.conv1 = SlimmableConv2d([3 for _ in range(len(channels))], channels,
										kernel_size=3, stride=1, padding=1, bias=False,
										width_mult_list=self.width_mult_list, overlap_weight=self.overlap_weight)
		self.bn1 = SwitchableBatchNorm2d(channels, width_mult_list=self.width_mult_list)
		self.relu = SlimmableReLU(inplace=act_inplace)

		# 2. backbone, 3 resnet basic blocks
		features = []
		expansion = 1
		for stage_id, n in enumerate(self.block_setting):
			# output channles for 4 blocks
			outp = [int(self.feats[stage_id] * width_mult * expansion) for width_mult in self.width_mult_list]

			for i in range(n):
				features.append(BasicBlock(channels, outp, 2 if (i == 0 and stage_id != 0) else 1,
										width_mult_list=self.width_mult_list, overlap_weight=self.overlap_weight))
				channels = outp
		self.features = nn.Sequential(*features)

		# 3. classifier, a fc layer
		self.outp = channels
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
		self.fc = SlimmableLinear(self.outp, [num_classes for _ in range(len(self.outp))],
										width_mult_list=self.width_mult_list, overlap_weight=self.overlap_weight)

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
		out = self.features(out)

		o_x = self.avgpool(out[0])
		o_x = torch.flatten(o_x, 1)

		out = self.fc((o_x, width_mult))

		return out[0]

	def forward(self, x, specify_width=None):
		"""
		sweet_x: a dict {width -> input sample}
		"""
		dict_out = dict()

		if specify_width is not None and specify_width in self.width_mult_list:
			dict_out[specify_width] = self.forward_single((x, specify_width))
		else:
			for width in self.width_mult_list:
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


if __name__ == "__main__":
	x = torch.rand(2, 3, 32, 32)
	model = s_ResNet_CIFAR(num_classes=10, width_mult_list=[1.0, 0.5])
	print(model)
	y = model(x)
	print(y[1].shape)
	print(y[0.5].shape)

