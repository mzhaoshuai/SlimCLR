# coding=utf-8
import os
import copy
import torch
import argparse
import torchvision
import numpy as np
from tqdm.autonotebook import tqdm
from torchvision import datasets, transforms

from .cifar import test
from ..utils.distributed import set_random_seed
from ..models.backbones.s_resnet_cifar import s_ResNet_CIFAR
from .plotter import plot_2d_contour, plot_contour_trajectory
from .plot_utils import create_random_direction, get_unplotted_indices, get_weights, set_weights, list_cal_angle


if __name__ == '__main__':	

	parser = argparse.ArgumentParser(description='plotting loss surface')
	parser.add_argument('--no_cuda', action='store_true', default=False,
							help='disables CUDA training')
	parser.add_argument('--dataset_dir', type=str,
							default='/home/shuai/dataset/cifar',
							help='where dataset located')
	parser.add_argument('--output', type=str,
							default='/home/shuai/output/vis',
							help='output path')
	parser.add_argument('--num', default=1, type=int, help='the id of experiment')

	parser.add_argument('--use_train', default=0, type=int, help='plot using training data or not')

	# model parameters
	parser.add_argument('--model_file',
							default='/home/shuai/output/visualize/mnist_cnn_e10.pt',
							help='path to the trained model file')
	parser.add_argument('--batch_size', type=int, default=64, metavar='N',
							help='input batch size for training (default: 64)')
	parser.add_argument('--test_batch_size', type=int, default=1000, metavar='N',
							help='input batch size for testing (default: 1000)')

	# direction parameters
	parser.add_argument('--dir_file', default='/home/shuai/output/vis/direction.pt',
							help='specify the name of direction file, or the path to an eisting direction file')
	parser.add_argument('--loss_file', default='',
							help='specify the name of direction file, or the path to an eisting loss file')
	parser.add_argument('--error_file', default='',
							help='specify the name of direction file, or the path to an eisting loss file')
	parser.add_argument('--traj_file', default='/home/shuai/output/vis/mnist_02/mnist_traj.npy',
							help='path to an eisting trajectory file')
	parser.add_argument('--traj_file1', default=None, type=str,
							help='another traj file')

	parser.add_argument('--x', default='-1:1:11', help='A string with format xmin:x_max:xnum')
	parser.add_argument('--y', default='-1:1:11', help='A string with format ymin:ymax:ynum')
	parser.add_argument('--norm', default='filter', help='direction normalization: filter | layer | weight')
	parser.add_argument('--ignore', default='biasbn', help='ignore bias and BN parameters: biasbn')

	 # plot parameters
	parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
	parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
	parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')

	parser.add_argument('--width_mult_list', default=[1.0, 0.25], nargs='+', type=float,
							help="all width multiplier for slimmable model.")

	parser.add_argument('--depth', type=int, default=20,
							help='the depth of networks (default: 20)')
	parser.add_argument('--width', type=int, default=1,
							help='the width of networks (default: 1)')

	args = parser.parse_args()
	args.output = os.path.join(args.output, 'vis_cifar_' + '{:0>3d}'.format(args.num))
	print('INFO: [files] The output dir is {}'.format(args.output))
	print(args)
	if not os.path.exists(args.output): os.makedirs(args.output)
	use_cuda = not args.no_cuda and torch.cuda.is_available()
	device = torch.device("cuda" if use_cuda else "cpu")
	set_random_seed(seed=3407)

	#--------------------------------------------------------------------------
	# Check plotting resolution
	#--------------------------------------------------------------------------
	try:
		args.xmin, args.xmax, args.xnum = [float(a) for a in args.x.split(':')]
		args.ymin, args.ymax, args.ynum = [float(a) for a in args.y.split(':')]
	except:
		raise Exception('Improper format for x- or y-coordinates. Try something like -1:1:51')
	# Create the coordinates(resolutions) at which the function is evaluated
	xcoordinates = np.linspace(args.xmin, args.xmax, num=int(args.xnum))
	ycoordinates = np.linspace(args.ymin, args.ymax, num=int(args.ynum))
	coords = get_unplotted_indices(xcoordinates, ycoordinates)
	print('INFO: [coordinates] The x-axis min {}, max {}, num {}'.format(args.xmin, args.xmax, args.xnum))
	print('INFO: [coordinates] The number of points are {}'.format(len(coords)))

	#--------------------------------------------------------------------------
	# create model
	#--------------------------------------------------------------------------
	model = s_ResNet_CIFAR(num_classes=10, input_size=32, width_mult_list=args.width_mult_list, args=args).to(device)
	print('INFO: [model] load model from {}'.format(args.model_file))
	old_sd = torch.load(args.model_file, map_location=device)
	model.load_state_dict(old_sd)
	final_w = get_weights(model)

	#--------------------------------------------------------------------------
	# generate directions
	#--------------------------------------------------------------------------
	if os.path.exists(args.dir_file):
		print('INFO: [direction] load directions from {}'.format(args.dir_file))
		xy_dir = torch.load(args.dir_file, map_location=device)
	else:
		x_dir = create_random_direction(model, ignore=args.ignore, norm=args.norm)
		y_dir = create_random_direction(model, ignore=args.ignore, norm=args.norm)
		xy_dir = [x_dir, y_dir]
		torch.save(xy_dir, os.path.join(args.output, 'direction.pt'))
	sim = list_cal_angle(xy_dir[0], xy_dir[1])
	print('INFO: [direction] The simlarity of two directions are {}'.format(sim.item()))

	#--------------------------------------------------------------------------
	# dataloader
	#--------------------------------------------------------------------------
	smaples = 4000
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
	val_transform = transforms.Compose([transforms.ToTensor(), normalize])
	train_dataset = datasets.CIFAR10(root=args.dataset_dir, train=True, download=False, transform=val_transform)
	val_dataset = datasets.CIFAR10(root=args.dataset_dir, train=False, transform=val_transform)
	train_dataset = torch.utils.data.Subset(train_dataset, list(range(smaples)))
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False,
												num_workers=8, pin_memory=True)
	test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.test_batch_size, shuffle=False,
												num_workers=8, pin_memory=True)

	print('INFO: [dataset] Use train set' if args.use_train else 'INFO: [dataset] Use val set')
	loader = train_loader if args.use_train else test_loader

	# get loss for plotting
	if os.path.exists(args.error_file) or os.path.exists(os.path.join(args.output, 'errors.npy')):
		if os.path.exists(args.error_file):
			errors = np.load(args.error_file)
		else:
			errors = np.load(os.path.join(args.output, 'errors.npy'))
	else:
		losses, errors = [], []
		progress_bar = tqdm(coords)
		with torch.no_grad():
			for cnt, coord in enumerate(progress_bar):
				set_weights(model, final_w, xy_dir, coord)
				loss, acc = test(model, device, loader, print_out=False)
				losses.append(loss)
				errors.append(100.0 - acc)
				progress_bar.set_description("Idx: {}. Loss: {:.3f}. Acc: {:.2f}. Coord: ({:.2f},{:.2f})".format(
												cnt, loss, acc, coord[0], coord[1]))

		losses = np.array(losses)
		errors = np.array(errors)
		np.save(os.path.join(args.output, 'losses.npy'), losses)
		np.save(os.path.join(args.output, 'errors.npy'), errors)

	surf_name = 'test_error'
	# losses = losses.reshape((int(args.xnum), int(args.ynum)))
	errors = errors.reshape((int(args.xnum), int(args.ynum)))
	# print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(losses), surf_name, np.min(losses)))
	print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(errors), surf_name, np.min(errors)))
	# plot loss surface or contour
	plot_2d_contour(xcoordinates, ycoordinates, errors, surf_name,
	 				args.vmin, args.vmax, args.vlevel, False, args.output)

	# plot trajectory
	print('INFO: [trajectory] Load trajectories from {}'.format(args.traj_file))
	traj_xy = np.load(args.traj_file)
	proj_x, proj_y = traj_xy[0], traj_xy[1]
	if os.path.exists(args.traj_file1):
		print('INFO: [trajectory] Load another trajectories from {}'.format(args.traj_file1))
		traj_xy1 = np.load(args.traj_file1)
		proj_x1, proj_y1 = traj_xy1[0], traj_xy1[1]
	else:
		proj_x1, proj_y1 = None, None

	plot_contour_trajectory(xcoordinates, ycoordinates, errors, proj_x, proj_y, surf_name,
								args.vmin, args.vmax, args.vlevel, path=args.output,
								proj_xcoord_1=proj_x1, proj_ycoord_1=proj_y1)
