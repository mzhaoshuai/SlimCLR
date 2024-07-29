# coding=utf-8
"""
https://github.com/tomgoldstein/loss-landscape
"""
import os
import copy
import torch
import random
import numpy as np
from sklearn.decomposition import PCA


@torch.no_grad()
def get_weights(net):
	"""Extract parameters from net, and return a list of tensors"""
	return [p.data.clone() for p in net.parameters()]


@torch.no_grad()
def get_diff_weights(weights, weights2):
	"""Produce a direction from 'weights' to 'weights2'."""
	return [w2 - w if w.dim() > 1 else torch.zeros_like(w) for (w, w2) in zip(weights, weights2)]


@torch.no_grad()
def set_weights(net, weights, directions=None, step=None):
	"""
	Overwrite the network's weights with a specified list of tensors
	or change weights along directions with a step size.
	"""
	assert step is not None, 'If a direction is specified then step must be specified as well'

	# dx = directions[0]
	# dy = directions[1]

	# for i, p in enumerate(net.parameters()):
	# 	p.data.copy_(weights[i] + step[0] * dx[i] + step[1] * dy[i])

	dx = directions[0]
	dy = directions[1]
	changes = [d0 * step[0] + d1 * step[1] for (d0, d1) in zip(dx, dy)]

	for (p, w, d) in zip(net.parameters(), weights, changes):
		p.data = w + d


@torch.no_grad()
def get_random_weights(weights):
	"""
	Produce a random direction that is a list of random Gaussian tensors
	with the same shape as the network's weights, so one direction entry per weight.
	"""
	return [torch.randn(w.size(), device=w.device) for w in weights]


################################################################################
#                        Normalization Functions
################################################################################
@torch.no_grad()
def normalize_direction(direction, weights, norm='filter'):
	"""
	Rescale the direction so that it has similar norm as their corresponding
	model in different levels.

	Args:
		direction: a variables of the random direction for one layer
		weights: a variable of the original model for one layer
		norm: normalization method, 'filter' | 'layer' | 'weight'
	"""
	if norm == 'filter':
		# Rescale the filters (weights in group) in 'direction' so that each
		# filter has the same norm as its corresponding filter in 'weights'.
		for d, w in zip(direction, weights):
			d.mul_(w.norm()/(d.norm() + 1e-10))
	elif norm == 'layer':
		# Rescale the layer variables in the direction so that each layer has
		# the same norm as the layer variables in weights.
		direction.mul_(weights.norm()/direction.norm())
	elif norm == 'weight':
		# Rescale the entries in the direction so that each entry has the same
		# scale as the corresponding weight.
		direction.mul_(weights)
	elif norm == 'dfilter':
		# Rescale the entries in the direction so that each filter direction
		# has the unit norm.
		for d in direction:
			d.div_(d.norm() + 1e-10)
	elif norm == 'dlayer':
		# Rescale the entries in the direction so that each layer direction has
		# the unit norm.
		direction.div_(direction.norm())


def ignore_biasbn(direction, weights):
	"""ignore bias and bn for a direction"""
	assert(len(direction) == len(weights))
	for d, w in zip(direction, weights):
		if d.dim() <= 1:
			# ignore directions for weights with 1 dimension
			d.fill_(0)
		else:
			# keep directions for weights/bias that are only 1 per node
			d.copy_(w)


def normalize_directions_for_weights(direction, weights, norm='filter', ignore='biasbn'):
	"""
	The normalization scales the direction entries according to the entries of weights.
	"""
	assert(len(direction) == len(weights))
	for d, w in zip(direction, weights):
		if d.dim() <= 1:
			if ignore == 'biasbn':
				d.fill_(0) # ignore directions for weights with 1 dimension
			else:
				d.copy_(w) # keep directions for weights/bias that are only 1 per node
		else:
			normalize_direction(d, w, norm)


def create_random_direction(net, ignore='biasbn', norm='filter'):
	"""
	Setup a random (normalized) direction with the same dimension as
	the weights or states.

	Args:
		net: the given trained model
		ignore: 'biasbn', ignore biases and BN parameters.
		norm: direction normalization method, including
			'filter" | 'layer' | 'weight' | 'dlayer' | 'dfilter'

	Returns:
		direction: a random direction with the same dimension as weights or states.
	"""
	# random direction
	weights = get_weights(net) # a list of parameters.
	direction = get_random_weights(weights)
	normalize_directions_for_weights(direction, weights, norm, ignore)

	return direction


def get_unplotted_indices(xcoordinates, ycoordinates):
	"""
	Args:
	  xcoordinates: x locations, i.e.,[-1, -0.5, 0, 0.5, 1]
	  ycoordinates: y locations, i.e.,[-1, -0.5, 0, 0.5, 1]

	Returns:
	  - a list of indices into vals for points that have not yet been calculated.
	  - a list of corresponding coordinates, with one x/y coordinate per row.
	"""
	# If the plot is 2D, then use meshgrid to enumerate all coordinates in the 2D mesh
	xcoord_mesh, ycoord_mesh = np.meshgrid(xcoordinates, ycoordinates)
	s1 = xcoord_mesh.ravel()
	s2 = ycoord_mesh.ravel()
	return np.c_[s1, s2]


def list_to_tensor(tensor_list):
	"""flatten tensor into 1D shape and concat them"""
	v = []
	for t in tensor_list:
		v.append(torch.flatten(t))

	return torch.cat(v)


def list_cal_angle(list1, list2):
	"""calculate the angle of two tensor list with the same shape"""
	return torch.nn.functional.cosine_similarity(list_to_tensor(list1),
													list_to_tensor(list2), dim=0)


################################################################################
#                        Functions for PCA direction
################################################################################
def load_traj_diff_files(net, model_dir, device, final_w, prefix='mnist_cnn_step'):
	"""
	load the intermediate files during training
	Args:
		model_dir: path has files like
						mnist_cnn_step000000.pt
						mnist_cnn_step000001.pt
						mnist_cnn_step000002.pt
						mnist_cnn_step000003.pt
						mnist_cnn_step000004.pt
						mnist_cnn_step000005.pt
						....
	Return:
		return a list contains the list of weights
	"""
	files = sorted([x for x in os.listdir(model_dir) if x.startswith(prefix)])
	all_diff_weights = []
	# import pdb; pdb.set_trace()
	for file in files:
		old_sd = torch.load(os.path.join(model_dir, file), map_location=device)
		net.load_state_dict(old_sd)
		tensor_list = get_diff_weights(final_w, get_weights(net))
		# ignore bias and bn
		# tensor_list = [x if x.dim() > 1 else torch.zeros_like(x) for x in tensor_list]
		all_diff_weights.append(list_to_tensor(tensor_list))

	return torch.stack(all_diff_weights, dim=0)


def npvec_to_tensorlist(direction, params):
	""" Convert a numpy vector to a list of tensors with the same shape as "params".

		Args:
			direction: a list of numpy vectors, e.g., a direction loaded from h5 file.
			base: a list of parameter tensors from net

		Returns:
			a list of tensors with the same shape as base
	"""
	if isinstance(params, list):
		w2 = copy.deepcopy(params)
		idx = 0
		for w in w2:
			w.copy_(torch.tensor(direction[idx:idx + w.numel()]).view(w.size()))
			idx += w.numel()
		assert(idx == len(direction))
		return w2
	else:
		s2 = []
		idx = 0
		for (k, w) in params.items():
			s2.append(torch.Tensor(direction[idx:idx + w.numel()]).view(w.size()))
			idx += w.numel()
		assert(idx == len(direction))
		return s2


def PCA_direction(net, model_dir, device, final_w, prefix='mnist_cnn_step',
					output_file='/home/zhaoshuai/models/vis'):
	"""get PCA direction from optimization process"""
	matrix = load_traj_diff_files(net, model_dir, device, final_w, prefix)
	matrix = matrix.numpy()
	# import pdb; pdb.set_trace()
	# Perform PCA on the optimization path matrix
	print ("INFO: [direction] Perform PCA on the models, the shape of the input is {}".format(matrix.shape))
	pca = PCA(n_components=2)
	pca.fit(matrix)
	pc1 = torch.from_numpy(pca.components_[0]).float()
	pc2 = torch.from_numpy(pca.components_[1]).float()

	sim = torch.nn.functional.cosine_similarity(pc1, pc2, dim=0)

	print("INFO: [direction] cos sim between pc1 and pc2: {}".format(sim.item()))
	print("INFO: [direction] pca.explained_variance_ratio_: %s" % str(pca.explained_variance_ratio_))

	x_dir = npvec_to_tensorlist(pca.components_[0], final_w)
	y_dir = npvec_to_tensorlist(pca.components_[1], final_w)
	# ignore_biasbn(x_dir, final_w)
	# ignore_biasbn(y_dir, final_w)
	xy_dir = [x_dir, y_dir]
	torch.save(xy_dir, output_file)


def project_1D(w, d):
	"""
	Project vector w to vector d and get the length of the projection.
	Args:
		w: vectorized weights
		d: vectorized direction

	Returns:
		the projection scalar
	"""
	assert len(w) == len(d), 'dimension does not match for w and '
	scale = torch.dot(w, d) / d.norm()

	return scale.item()


def project_2D(d, dx, dy, proj_method):
	"""
	Project vector d to the plane spanned by dx and dy.
	Args:
		d: vectorized weights
		dx: vectorized direction
		dy: vectorized direction
		proj_method: projection method
	Returns:
		x, y: the projection coordinates
	"""

	if proj_method == 'cos':
		# when dx and dy are orthorgonal
		x = project_1D(d, dx)
		y = project_1D(d, dy)
	elif proj_method == 'lstsq':
		# solve the least squre problem: Ax = d
		A = np.vstack([dx.numpy(), dy.numpy()]).T
		[x, y] = np.linalg.lstsq(A, d.numpy())[0]

	return x, y


def project_trajectory(net, dir_file, final_w, model_dir, device, prefix='mnist_cnn_step',
			   			proj_method='cos', ouput_file='.'):
	"""
	Project the optimization trajectory onto the given two directions.

	Args:
		net: the torch model
		dir_file: the file that contains the directions
		w: weights of the final model
		model_dir: path to the checkpoint files
		proj_method: cosine projection

	Returns:
		proj_file: the projection filename
	"""
	print('INFO: [direction] load directions from {}'.format(dir_file))
	xy_dir = torch.load(dir_file, map_location=device)
	dx = list_to_tensor(xy_dir[0])
	dy = list_to_tensor(xy_dir[1])
	# all saved model files
	model_files = sorted([x for x in os.listdir(model_dir) if x.startswith(prefix)])

	xcoord, ycoord = [], []
	for model_file in model_files:

		old_sd = torch.load(os.path.join(model_dir, model_file), map_location=device)
		net.load_state_dict(old_sd)
		w2 = get_weights(net)
		d = get_diff_weights(final_w, w2)
		d = list_to_tensor(d)

		x, y = project_2D(d, dx, dy, proj_method)
		print("%s  (%.4f, %.4f)" % (model_file, x, y))

		xcoord.append(x)
		ycoord.append(y)

	print('INFO: [direction] save trajectory into {}'.format(ouput_file))
	np.save(ouput_file, [np.array(xcoord), np.array(ycoord)], allow_pickle=True)


def mnist_pca_direction():
	from .mnist import Net
	device = torch.device("cpu")
	folder_id = 'mnist_04'
	# final_model_file = '/home/zhaoshuai/models/vis/{}/mnist_cnn_final.pt'.format(folder_id)
	final_model_file = '/home/zhaoshuai/models/vis/mnist_03/mnist_cnn_final.pt'
	model_files = '/home/zhaoshuai/models/vis/{}'.format(folder_id)
	output_file = os.path.join(model_files, 'pca_direction.pt')
	prefix='mnist_cnn_step'

	model = Net()
	print('INFO: [model] load model from {}'.format(final_model_file))
	old_sd = torch.load(final_model_file, map_location=device)
	model.load_state_dict(old_sd)
	final_w = get_weights(model)

	# calculate PCA direction
	PCA_direction(model, model_files, device, final_w, prefix=prefix,
					output_file=output_file)

	# calculate trajectory
	output_traj_file = '/home/zhaoshuai/models/vis/{}/mnist_traj.npy'.format(folder_id)
	dir_file = '/home/zhaoshuai/models/vis/mnist_03/pca_direction.pt'
	project_trajectory(model, dir_file, final_w, model_files, device, prefix=prefix,
							proj_method='cos', ouput_file=output_traj_file)


if __name__ == "__main__":
	from ..models.backbones.s_resnet_cifar import s_ResNet_CIFAR
	device = torch.device("cpu")
	folder_id = 'cifar_06'
	depth = 20
	width = 4
	num_classes = 10
	s_width = 1.0

	final_model_file = '/home/shuai/output/{}/resnet20_cifar_final.pt'.format(folder_id)
	model_files = '/home/shuai/output/{}'.format(folder_id)
	output_file = os.path.join(model_files, 'pca_direction.pt')
	prefix = 'resnet22_step_{}'.format(s_width)

	model = s_ResNet_CIFAR(num_classes=num_classes, input_size=32, width_mult_list=[1.0, 0.5], depth=depth, width=width)
	print('INFO: [model] load model from {}'.format(final_model_file))
	old_sd = torch.load(final_model_file, map_location=device)
	model.load_state_dict(old_sd)
	final_w = get_weights(model)

	# calculate PCA direction
	PCA_direction(model, model_files, device, final_w, prefix=prefix, output_file=output_file)

	# calculate trajectory
	output_traj_file = '/home/shuai/output/{}/cifar_traj_{}.npy'.format(folder_id, s_width)
	dir_file = '/home/shuai/output/{}/pca_direction.pt'.format(folder_id)
	prefix = 'resnet22_step_{}'.format(s_width)
	project_trajectory(model, dir_file, final_w, model_files, device, prefix=prefix,
							proj_method='cos', ouput_file=output_traj_file)
