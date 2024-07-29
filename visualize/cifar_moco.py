# coding=utf-8
"""
A visualization demo for moco on minist
"""
import os
import torch
import argparse
import torch.nn.functional as F

from tqdm.autonotebook import tqdm
from torchvision import datasets, transforms
from .moco_builder import ModelMoCo
from ..dataset import TwoCropsTransform
from ..utils.distributed import set_random_seed
from ..utils.misc import get_the_number_of_params


def train(net, data_loader, optimizer, epoch, args):
	"""train for one epoch"""
	net.train()
	width_mult_list = net.width_mult_list
	total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
	step, step_per_epoch = 0, len(data_loader)
	for (im_1, im_2), target in train_bar:
		step += 1
		optimizer.zero_grad()
		im_1, im_2 = im_1.cuda(non_blocking=True), im_2.cuda(non_blocking=True)

		for i, width in enumerate(width_mult_list):
			loss = net.forward_single(im_1, im_2, epoch=epoch, width=width)
			
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()

			if step % step_per_epoch == 0:
				torch.save(net.encoder_q.state_dict(), os.path.join(args.log_dir, "{}_{}_{:0>6d}.pt".format(
				args.model_prefix, width, epoch)))

		total_num += im_1.shape[0]
		total_loss += loss.item() * data_loader.batch_size
		train_bar.set_description('Train Epoch: [{}/{}], lr: {:.4f}, Loss: {:.4f}'.format(
									epoch, args.epochs, optimizer.param_groups[0]['lr'], total_loss / total_num))

	return total_loss / total_num


def test(net, memory_data_loader, test_data_loader, epoch, args, print_out=True):
	"""test for all subnetworks"""
	width_mult_list = net.width_mult_list
	top1 = []
	for width in width_mult_list:
		avg_top1 = test_single_width(net, memory_data_loader, test_data_loader, epoch, width, args, print_out=print_out)
		top1.append(avg_top1)

	return top1[0]


def test_single_width(net, memory_data_loader, test_data_loader, epoch, width, args, print_out=True):
	"""test using a knn monitor"""
	net.eval()
	classes = len(memory_data_loader.dataset.classes)
	total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
	with torch.no_grad():
		# generate feature bank
		for data, target in memory_data_loader:
			feature = net(data.cuda(non_blocking=True), width)[width]
			feature = F.normalize(feature, dim=1)
			feature_bank.append(feature)
		# feature [D, N], labels [N]
		feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
		feature_labels = torch.tensor(memory_data_loader.dataset.targets, device=feature_bank.device)

		# loop test data to predict the label by weighted knn search
		for data, target in test_data_loader:
			data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)
			feature = net(data, width)[width]
			feature = F.normalize(feature, dim=1)

			pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, args.knn_k, args.knn_t)

			total_num += data.size(0)
			total_top1 += (pred_labels[:, 0] == target).float().sum().item()
		avg_top1 = total_top1 / total_num * 100
		if print_out:
			print('Test Epoch: [{}/{}] Width: {}, Acc@1:{:.2f}%'.format(epoch, args.epochs, width, avg_top1))

	return avg_top1


def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
	"""
	knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
	implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
	"""
	# compute cos similarity between each feature vector and feature bank ---> [B, N]
	sim_matrix = torch.mm(feature, feature_bank)
	# [B, K]
	sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
	# [B, K]
	sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
	sim_weight = (sim_weight / knn_t).exp()

	# counts for each class
	one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
	# [B*K, C]
	one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
	# weighted score ---> [B, C]
	pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

	pred_labels = pred_scores.argsort(dim=-1, descending=True)
	return pred_labels


def main():
	parser = argparse.ArgumentParser(description='Train MoCo on CIFAR-10')
	parser.add_argument('--dataset_dir', type=str,
							default='/home/shuai/dataset/cifar',
							help='where dataset located')
	parser.add_argument('--workers', default=16, type=int, metavar='N',
							help='number of data loading workers (default: 16)')
	parser.add_argument('-a', '--arch', default='resnet18')
	parser.add_argument('--seed', type=int, default=3047, metavar='S',
							help='random seed (default: 1)')
	parser.add_argument("--log_dir", type=str,
							default='/home/shuai/output/resnet22_01', 
							help="The output directory where the model predictions and checkpoints will be written.")
	parser.add_argument("--model_prefix", type=str,
							default='resnet22_step', 
							help="intilial weights.")
	parser.add_argument('--width_mult_list', default=[1.0, 0.5], nargs='+', type=float,
								help="all width multiplier for slimmable model.")
	parser.add_argument('--loss_w', default=[1.0, 1.0], nargs='+', type=float,
							help="loss weight for different subnetworks.")
	parser.add_argument('--depth', type=int, default=20,
							help='the depth of networks (default: 20)')
	parser.add_argument('--width', type=int, default=1,
							help='the width of networks (default: 1)')
	# lr: 0.06 for batch 512 (or 0.03 for batch 256)
	parser.add_argument('--lr', default=0.06, type=float, metavar='LR', help='initial learning rate', dest='lr')
	parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
	parser.add_argument('--schedule', default=[120, 160], nargs='*', type=int,
							help='learning rate schedule (when to drop lr by 10x); does not take effect if --cos is on')
	parser.add_argument('--cos', action='store_true', help='use cosine lr schedule')

	parser.add_argument('--batch_size', default=512, type=int, metavar='N', help='mini-batch size')
	parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
							help='input batch size for testing (default: 1000)')
	parser.add_argument('--wd', default=5e-4, type=float, metavar='W', help='weight decay')

	# moco specific configs:
	parser.add_argument('--moco_dim', default=128, type=int, help='feature dimension')
	parser.add_argument('--moco_k', default=4096, type=int, help='queue size; number of negative keys')
	parser.add_argument('--moco_m', default=0.99, type=float, help='moco momentum of updating key encoder')
	parser.add_argument('--moco_t', default=0.1, type=float, help='softmax temperature')

	parser.add_argument('--bn_splits', default=8, type=int,
							help='simulate multi-gpu behavior of BatchNorm in one gpu; 1 is SyncBatchNorm in multi-gpu')
	parser.add_argument('--symmetric', action='store_true', help='use a symmetric loss function that backprops to both crops')
	# knn monitor
	parser.add_argument('--knn_k', default=200, type=int, help='k in kNN monitor')
	parser.add_argument('--knn_t', default=0.1, type=float, help='softmax temperature in kNN monitor; could be different with moco-t')
	# utils
	parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
	parser.add_argument('--results_dir', default='', type=str, metavar='PATH', help='path to cache (default: none)')
	args = parser.parse_args()
	print(args)
	print('INFO: [log] The output dir is {}'.format(args.log_dir))
	if not os.path.exists(args.log_dir): os.makedirs(args.log_dir)

	set_random_seed(args.seed)
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.deterministic = True		
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

	# create data loader
	train_aug = transforms.Compose([
		transforms.RandomResizedCrop(32),
		transforms.RandomHorizontalFlip(p=0.5),
		transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
		transforms.RandomGrayscale(p=0.2),
		transforms.ToTensor(),
		transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
	train_transform = TwoCropsTransform(train_aug)

	test_transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])

	train_dataset = datasets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=train_transform)
	memory_dataset = datasets.CIFAR10(root=args.dataset_dir, train=True, download=True, transform=test_transform)
	val_dataset = datasets.CIFAR10(root=args.dataset_dir, train=False, transform=test_transform)
	
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
												num_workers=args.workers, pin_memory=True, drop_last=True)
	memory_loader = torch.utils.data.DataLoader(memory_dataset, batch_size=args.test_batch_size, shuffle=False,
												num_workers=args.workers, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False,
												num_workers=args.workers, pin_memory=True)

	# define optimizer
	model = ModelMoCo(dim=args.moco_dim, K=args.moco_k, m=args.moco_m, T=args.moco_t,
						width_mult_list=args.width_mult_list, args=args).to(device)
	if os.path.exists(args.resume):
		print('INFO: [resume] load ckpt from {}'.format(args.resume))
		old_sd = torch.load(args.resume, map_location=device)
		model.load_state_dict(old_sd)
	optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
	scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs + 1)

	# training loop
	# global_step = 0
	print('INFO: [model] The number of parameters are {}'.format(get_the_number_of_params(model.encoder_q)))
	for width in args.width_mult_list:
		torch.save(model.encoder_q.state_dict(), os.path.join(args.log_dir, "{}_{}_{:0>6d}.pt".format(
					args.model_prefix, width, 0)))

	for epoch in range(1, args.epochs + 1):
		train(model, train_loader, optimizer, epoch, args)
		test(model.encoder_q, memory_loader, test_loader, epoch, args)
		scheduler.step()
		# print('INFO: [lr] The current lr is {}'.format(scheduler.get_last_lr()))
		# torch.save(model.encoder_q.state_dict(), os.path.join(args.log_dir, "{}{:0>6d}.pt".format(prefix, epoch)))
	torch.save(model.encoder_q.state_dict(), os.path.join(args.log_dir, "resnet20_cifar_final.pt"))


if __name__ == '__main__':
	main()
