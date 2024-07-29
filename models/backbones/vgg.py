# coding=utf-8
# an modified VGG for detection tasks -- backbone for SSD, not the same as the original
import torch
import torch.nn as nn
import torch.nn.functional as F


class VGGBasev2(nn.Module):
	"""
	VGG base convolutions to produce lower-level feature maps.
	"""
	def __init__(self, width_multipier=1.0, num_classes=1000):
		super(VGGBasev2, self).__init__()
		print('INFO: [model] Creating VGG with width_multipier {}'.format(width_multipier))
		default_channels = [128, 256, 512, 512, 1024, 1024]
		self.chs = [64,] + [int(width_multipier * x) for x in default_channels]

		# standard convolutional layers in VGG16
		self.conv1_1 = nn.Conv2d(3, self.chs[0], kernel_size=3, padding=1)
		self.conv1_2 = nn.Conv2d(self.chs[0], self.chs[0], kernel_size=3, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv2_1 = nn.Conv2d(self.chs[0], self.chs[1], kernel_size=3, padding=1)
		self.conv2_2 = nn.Conv2d(self.chs[1], self.chs[1], kernel_size=3, padding=1)
		self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv3_1 = nn.Conv2d(self.chs[1], self.chs[2], kernel_size=3, padding=1)
		self.conv3_2 = nn.Conv2d(self.chs[2], self.chs[2], kernel_size=3, padding=1)
		self.conv3_3 = nn.Conv2d(self.chs[2], self.chs[2], kernel_size=3, padding=1)
		# ceiling (not floor) here for even dims
		self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

		self.conv4_1 = nn.Conv2d(self.chs[2], self.chs[3], kernel_size=3, padding=1)
		self.conv4_2 = nn.Conv2d(self.chs[3], self.chs[3], kernel_size=3, padding=1)
		self.conv4_3 = nn.Conv2d(self.chs[3], self.chs[3], kernel_size=3, padding=1)
		self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

		self.conv5_1 = nn.Conv2d(self.chs[3], self.chs[4], kernel_size=3, padding=1)
		self.conv5_2 = nn.Conv2d(self.chs[4], self.chs[4], kernel_size=3, padding=1)
		self.conv5_3 = nn.Conv2d(self.chs[4], self.chs[4], kernel_size=3, padding=1)
		# retains size because stride is 1 (and padding)
		self.pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)

		# replacements for FC6 and FC7 in VGG16, atrous convolution
		self.conv6 = nn.Conv2d(self.chs[4], self.chs[5], kernel_size=3, padding=6, dilation=6)
		self.conv7 = nn.Conv2d(self.chs[5], self.chs[6], kernel_size=1)


		# not used in SSD300
		self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
		self.conv8 = nn.Conv2d(self.chs[6], self.chs[6] * 2, kernel_size=1)
		self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

		self.relu = nn.ReLU(inplace=True)
		self.fc = nn.Linear(self.chs[6] * 2, num_classes)

		self.init_weights()

	def forward(self, image):
		"""
		Forward propagation.
		:param image: images, a tensor of dimensions (N, 3, H, W)
		:return: lower-level feature maps conv4_3 and conv7
		"""
		out = self.relu(self.conv1_1(image)) 			# (N, 64, H, W)
		out = self.relu(self.conv1_2(out))  			# (N, 64, H, W)
		out = self.pool1(out)  							# (N, 64, H // 2, W // 2)

		out = self.relu(self.conv2_1(out))  			# (N, 128, 150, 150)
		out = self.relu(self.conv2_2(out))  			# (N, 128, 150, 150)
		out = self.pool2(out)  							# (N, 128, 75, 75)

		out = self.relu(self.conv3_1(out))  			# (N, 256, 75, 75)
		out = self.relu(self.conv3_2(out))  			# (N, 256, 75, 75)
		out = self.relu(self.conv3_3(out))  			# (N, 256, 75, 75)
		out = self.pool3(out)  							# (N, 256, 38, 38), it would have been 37 if not for ceil_mode = True

		out = self.relu(self.conv4_1(out))  			# (N, 512, 38, 38)
		out = self.relu(self.conv4_2(out))  			# (N, 512, 38, 38)
		out = self.relu(self.conv4_3(out))  			# (N, 512, 38, 38)
		# conv4_3_feats = out  							# (N, 512, 38, 38)
		out = self.pool4(out)  							# (N, 512, 19, 19)

		out = self.relu(self.conv5_1(out))  			# (N, 512, 19, 19)
		out = self.relu(self.conv5_2(out))  			# (N, 512, 19, 19)
		out = self.relu(self.conv5_3(out))  			# (N, 512, 19, 19)
		out = self.pool5(out)  							# (N, 512, 19, 19), pool5 does not reduce dimensions

		out = self.relu(self.conv6(out))  				# (N, 1024, 19, 19)

		out = self.relu(self.conv7(out))
		# conv7_feats = self.relu(self.conv7(out)) 		# (N, 1024, 19, 19)

		out = self.pool6(out)
		out = self.relu(self.conv8(out))
		out = self.avgpool(out)

		out = self.fc(torch.flatten(out, 1))
		# Lower-level feature maps
		return out

	def init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				nn.init.normal_(m.weight, 0, 0.01)
				nn.init.constant_(m.bias, 0)


if __name__ == "__main__":
	model = VGGBasev2()
	x = torch.rand(1, 3, 224, 224)
	y = model(x)
	print(y.shape)
