# coding=utf-8
import os
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from torchvision import datasets, transforms
from ..utils.distributed import set_random_seed
from ..utils.misc import get_the_number_of_params
from ..models.backbones.s_resnet_cifar import s_ResNet_CIFAR


def train(args, model, device, train_loader, optimizer, epoch, global_step=-1):
	sorted_width_mult_list = sorted(args.width_mult_list, reverse=True)
	loss_weight = args.loss_w
	max_width, min_width = sorted_width_mult_list[0], sorted_width_mult_list[-1]
	step_per_epoch = len(train_loader)
	model.train()
	progress_bar = tqdm(train_loader)

	for batch_idx, (data, target) in enumerate(progress_bar):
		data, target = data.to(device), target.to(device)
		optimizer.zero_grad()
		# calculate loss for multiple networks
		# all_losses = []
		for i, width_mult in enumerate(sorted_width_mult_list):
			output = model.forward_single((data, width_mult))
			if width_mult == max_width:
				loss = F.cross_entropy(output, target, reduction='mean')
				soft_target = F.softmax(output.clone(), dim=1).detach()
			else:
				if args.inplace_distill:
					loss = (F.cross_entropy(output, target, reduction='mean') +
							F.kl_div(F.log_softmax(output, dim=1), soft_target, reduction='batchmean')) / 2.0
				else:
					loss = F.cross_entropy(output, target, reduction='mean')
			loss = loss_weight[i] * loss
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			if (global_step + 1) % step_per_epoch == 0:
				torch.save(model.state_dict(), os.path.join(args.log_dir, "{}_{}_{:0>6d}.pt".format(
								args.model_prefix, width_mult, global_step)))

		progress_bar.set_description("Epoch: {}. Iter [{}/{}]. Loss: {:.5f}".format(epoch,
										batch_idx * len(data), len(train_loader.dataset), loss.item()))
		global_step += 1

	return global_step


def test(model, device, test_loader, print_out=True):
	model.eval()
	num_sample = len(test_loader.dataset)
	width_mult_list = model.width_mult_list
	test_loss = [0.0 for _ in width_mult_list]
	correct = [0.0 for _ in width_mult_list]
	acc = [0.0 for _ in width_mult_list]
	with torch.no_grad():
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			dict_out = model(data)
			for i, w in enumerate(width_mult_list):
				output = dict_out[w]
				# sum up batch loss
				test_loss[i] = test_loss[i] + F.cross_entropy(output, target, reduction='sum').item()
				# get the index of the max log-probability
				pred = output.argmax(dim=1, keepdim=True)
				correct[i] = correct[i] + pred.eq(target.view_as(pred)).sum().item()		

	for i, w in enumerate(width_mult_list):
		test_loss[i] = test_loss[i] / num_sample
		acc[i] = 100. * correct[i] / num_sample

		if print_out:
			print('INFO: [meter] Test set: Width: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
					w, test_loss[i], correct[i], num_sample, acc[i]))

	return test_loss[0], acc[0]


def main():
	# Training settings
	parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
	parser.add_argument('--dataset_dir', type=str,
							default='/home/shuai/dataset/cifar',
							help='where dataset located')
	parser.add_argument("--log_dir", type=str,
							default='/home/shuai/output/resnet22_01', 
							help="The output directory where the model predictions and checkpoints will be written.")
	parser.add_argument("--resume", type=str,
							default='/none/', 
							help="intilial weights.")
	parser.add_argument("--model_prefix", type=str,
							default='resnet22_step', 
							help="intilial weights.")
	parser.add_argument('--workers', default=4, type=int, metavar='N',
							help='number of data loading workers (default: 4)')
	parser.add_argument('--batch_size', type=int, default=256, metavar='N',
							help='input batch size for training (default: 64)')
	parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
							help='input batch size for testing (default: 1000)')
	parser.add_argument('--epochs', type=int, default=100, metavar='N',
							help='number of epochs to train (default: 10)')
	parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
							help='learning rate (default: 1.0)')
	parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
							help='momentum')
	parser.add_argument('--weight_decay', default=1e-4, type=float,
							metavar='W', help='weight decay (default: 1e-4)')
	parser.add_argument('--no-cuda', action='store_true', default=False,
							help='disables CUDA training')
	parser.add_argument('--seed', type=int, default=3047, metavar='S',
							help='random seed (default: 1)')
	parser.add_argument('--width_mult_list', default=[1.0, 0.25], nargs='+', type=float,
							help="all width multiplier for slimmable model.")
	parser.add_argument('--loss_w', default=[1.0, 1.0], nargs='+', type=float,
							help="loss weight for different subnetworks.")
	parser.add_argument("--inplace_distill", type=int, default=0,
							help="apply distillationb between large and subnetworks.")
	parser.add_argument('--depth', type=int, default=20,
							help='the depth of networks (default: 20)')
	parser.add_argument('--width', type=int, default=1,
							help='the width of networks (default: 1)')
	args = parser.parse_args()
	assert len(args.width_mult_list) == len(args.loss_w)
	print(args)
	print('INFO: [log] The output dir is {}'.format(args.log_dir))
	if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	set_random_seed(args.seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True
	device = torch.device("cuda" if use_cuda else "cpu")

	# create data loader
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),
											transforms.RandomCrop(32, padding=4),
											transforms.ToTensor(),
											normalize,
										])
	val_transform = transforms.Compose([transforms.ToTensor(), normalize])
	train_dataset = datasets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=train_transform)
	val_dataset = datasets.CIFAR10(root=args.dataset_dir, train=False, transform=val_transform)
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
												num_workers=args.workers, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False)

	# create model
	model = s_ResNet_CIFAR(num_classes=10, input_size=32, width_mult_list=args.width_mult_list, args=args).to(device)
	if os.path.exists(args.resume):
		print('INFO: [resume] load ckpt from {}'.format(args.resume))
		old_sd = torch.load(args.resume, map_location=device)
		model.load_state_dict(old_sd)

	# define loss function (criterion) and optimizer
	optimizer = torch.optim.SGD(model.parameters(), args.lr,
								momentum=args.momentum,
								weight_decay=args.weight_decay)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs + 1)

	# training loop
	global_step = 0

	print('INFO: [model] The number of parameters are {}'.format(get_the_number_of_params(model)))
	for width in args.width_mult_list:
		torch.save(model.state_dict(), os.path.join(args.log_dir, "{}_{}_{:0>6d}.pt".format(
					args.model_prefix, width, global_step)))

	for epoch in range(1, args.epochs + 1):
		# with torch.autograd.set_detect_anomaly(True):
		g_step = train(args, model, device, train_loader, optimizer, epoch, global_step=global_step)
		global_step = g_step
		test(model, device, test_loader)
		scheduler.step()
		print('INFO: [lr] The current lr is {}'.format(scheduler.get_last_lr()))

	torch.save(model.state_dict(), os.path.join(args.log_dir, "resnet20_cifar_final.pt"))


if __name__ == '__main__':
	main()
