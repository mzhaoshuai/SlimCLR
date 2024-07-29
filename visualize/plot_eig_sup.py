# coding=utf-8
import os
import sys
import json
import torch
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import AutoMinorLocator
from scipy.interpolate import make_interp_spline


# import seaborn as sns
# sns.set_theme(style="whitegrid")
plt.style.use('seaborn-paper')
plt.rcParams["figure.autolayout"] = True
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in' 
plt.rcParams['ytick.right'] = plt.rcParams['ytick.labelright'] = True
plt.rcParams['ytick.left'] = True
plt.rcParams['ytick.labelright'] = False
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']

largest_color = '#66c2a5'
second_color = '#8da0cb'


linestyle_tuple = [
	('loosely dotted',        (0, (1, 10))),
	('dotted',                (0, (1, 1))),
	('densely dotted',        (0, (1, 1))),

	('loosely dashed',        (0, (5, 10))),
	('dashed',                (0, (5, 5))),
	('densely dashed',        (0, (5, 1))),

	('loosely dashdotted',    (0, (3, 10, 1, 10))),
	('dashdotted',            (0, (3, 5, 1, 5))),
	('densely dashdotted',    (0, (3, 1, 1, 1))),

	('dashdotdotted',         (0, (3, 5, 1, 5, 1, 5))),
	('loosely dashdotdotted', (0, (3, 10, 1, 10, 1, 10))),
	('densely dashdotdotted', (0, (3, 1, 1, 1, 1, 1)))
	]


def spline(x, y, new_ticks=100):
	X_Y_Spline = make_interp_spline(x, y)
	# returns evenly spaced numbers over a specified interval.
	X_ = np.linspace(x.min(), x.max(), new_ticks)
	Y_ = X_Y_Spline(X_)

	return X_, Y_


def load_json(json_file):
	with open(json_file, mode='r') as f:
		content = json.load(f)
	times, values = [], []
	for c in content:
		times.append(c[1])
		values.append(c[2])
	x, y = np.array(times), np.array(values)
	return x * 4, y


def plot_lines(json_file=None, extra_json_file=None, filename=None, title=None, title_fontsize=16,
				show_y=False, min_v=0.0, max_v=1.0, ticks=100):
	x1, y1 = load_json(json_file)
	x2, y2 = load_json(extra_json_file)

	# smooth the curve
	x1, y1 = spline(x1, y1, new_ticks=ticks)
	x2, y2 = spline(x2, y2, new_ticks=ticks)

	# plot
	fig, ax = plt.subplots()  # Create a figure and an axes.

	ax.plot(x1, y1, label=r'supervised, largest singular value', color=largest_color, linewidth=2.5)
	ax.plot(x2, y2, label=r'supervised, second largest singular value', color=second_color, linewidth=2.5)

	ax.set_xlabel('Training step of ResNet-50$_{[1.0, 0.5, 0.25]}$', fontsize=title_fontsize)
	if show_y:
		ax.set_ylabel(r'Eigenvalues', fontsize=title_fontsize)
	ax.set_title(title, fontsize=title_fontsize, color='blue')
	ax.grid(axis='y', which='major', linestyle='-.', linewidth=0.5)
	ax.grid(axis='x', which='major', linestyle='-.', linewidth=0.5)
	ax.set_xlim([0, 500400])
	ax.set_ylim(min_v, max_v)
	ax.legend(loc='upper left', prop={'size': title_fontsize})

	ax.tick_params(axis='x', labelsize=14)
	ax.tick_params(axis='y', labelsize=14)
	ax.ticklabel_format(style='scientific', scilimits=(0, 0), axis='x')
	ax.xaxis.get_offset_text().set_fontsize(14)

	print(json_file, extra_json_file, '====>>>>', filename)
	fig.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')


def load_weight_files(model_dir, device='cpu', prefix='fc_grad_layer2'):
	"""
	load the intermediate files during training
	Args:
		model_dir: path has files like
		fc_grad_layer2_step99300.pth
		fc_grad_layer2_step99400.pth
		fc_grad_layer2_step99500.pth
		fc_grad_layer2_step99600.pth
		fc_grad_layer2_step99700.pth
		fc_grad_layer2_step99800.pth
		fc_grad_layer2_step99900.pth
						....
	Return:
		return a list contains the list of weights
	"""
	print("loading files from {}".format(model_dir))
	files = sorted([x for x in os.listdir(model_dir) if x.startswith(prefix)])
	# print(files)
	all_weights, time_stamp = [], []
	for file in files:
		old_sd = torch.load(os.path.join(model_dir, file), map_location=device)
		all_weights.append(old_sd.flatten())
		time_stamp.append(int(file.split('step')[-1].split('.')[0]))

	return torch.stack(all_weights, dim=0), time_stamp


def PCA_projection(model_dir, device="cpu", prefix='fc_grad_layer2'):
	"""get PCA direction from optimization process and projection points"""
	matrix_th, time_stamp = load_weight_files(model_dir, device, prefix)
	matrix = matrix_th.numpy()
	# import pdb; pdb.set_trace()
	# Perform PCA on the optimization path matrix
	print ("INFO: [direction] Perform PCA on the model weights, the shape of the input is {}".format(matrix.shape))
	pca = PCA(n_components=2)
	pca.fit(matrix)
	pc1 = torch.from_numpy(pca.components_[0]).float()
	pc2 = torch.from_numpy(pca.components_[1]).float()

	sim = torch.nn.functional.cosine_similarity(pc1, pc2, dim=0)

	print("INFO: [direction] cos sim between pc1 and pc2: {}".format(sim.item()))
	print("INFO: [direction] pca.explained_variance_ratio_: %s" % str(pca.explained_variance_ratio_))

	# projection
	y = torch.matmul(matrix_th, pc1) # / pc1.norm()
	z = torch.matmul(matrix_th, pc2) # / pc2.norm()
	x, y, z = np.array(time_stamp), y.numpy(), z.numpy()
	inds = np.argsort(x)
	x, y, z = x[inds], y[inds], z[inds]

	xyz = [x, y, z]
	torch.save(xyz, os.path.join(model_dir, prefix + '_pca.pth'))

	return xyz


def plot_pca_directions(model_dir, filename, prefix='fc_grad_layer2', ticks=100, overwrite=False):

	color = largest_color
	title = "Supervised"
	xlims = [0, 500400]
	ylims = [-0.35, 0.35]
	zlims = [-0.35, 0.35]
	tick_font = 8
	label_font = 12
	title_font = 16
	mark_size = 12

	possible_file = os.path.join(model_dir, prefix + '_pca.pth')
	if os.path.exists(possible_file) and not overwrite:
		print("INFO: [direction] load {}".format(possible_file))
		x, y, z = torch.load(possible_file, map_location='cpu')
	else:
		x, y, z = PCA_projection(model_dir, prefix=prefix)
	# legacy bug
	x = x * 4
	print("shapes, x {}, y {}, z {}".format(x.shape, y.shape, z.shape))
	print("min, x {}, y {}, z {}".format(x.min(), y.min(), z.min()))
	print("max, x {}, y {}, z {}".format(x.max(), y.max(), z.max()))
	y = np.clip(y, ylims[0], ylims[1])
	z = np.clip(z, zlims[0], zlims[1])

	# --------------------------------------------------------------------
	# Plot 3D scatters
	# --------------------------------------------------------------------
	fig = plt.figure()
	ax = Axes3D(fig)

	x_mean = np.mean(x)
	# yy = np.linspace(ylims[0], ylims[1], 9)
	# zz = np.linspace(zlims[0], zlims[1], 9)
	# yy_, zz_ = np.meshgrid(yy, zz)
	# ax.plot_surface(X=yy * 0 + x_mean, Y=yy_, Z=zz_,  color='r', alpha=0.3) 
	# # ax.plot3D(x_mean + yy * 0, yy, ylims[0] + yy * 0, color='r', alpha=0.5)
	# ax.text(x_mean + 0.05e5, ylims[0], zlims[0], "slow start", color='red')
	ax.scatter3D(x, y, z, color=color) # cmap='Blues', 
	# ax.plot3D(x, y, z, 'gray')

	ax.set_xlim(xlims)
	ax.set_ylim(ylims)
	ax.set_zlim(zlims)
	ax.tick_params(axis='x', labelsize=tick_font)
	ax.tick_params(axis='y', labelsize=tick_font)
	ax.tick_params(axis='z', labelsize=tick_font)

	ax.ticklabel_format(style='scientific', scilimits=(0, 0), axis='x')
	ax.set_xlabel('Training step', fontsize=label_font)
	ax.set_ylabel('1st PCA component', fontsize=label_font)
	ax.set_zlabel('2nd PCA component', fontsize=label_font)
	ax.set_title(title, fontsize=title_font, color='blue')

	# ax1.ticklabel_format(style='scientific', scilimits=(0, 0), axis='x')
	fn1 = filename.replace(".pdf", "_3D.pdf")
	print(model_dir, '====>>>>', fn1)
	fig.savefig(fn1, dpi=300, bbox_inches='tight', format='pdf')


	# --------------------------------------------------------------------
	# Plot 2D scatters
	# --------------------------------------------------------------------
	fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)
	# ax1.vlines(x_mean, ylims[0], ylims[1], color='r', alpha=0.5)
	# ax1.text(x_mean - 0.4e5, ylims[0] + 0.05, "slow start", color='red')
	ax1.scatter(x, y, color=color, s=mark_size)
	ax1.set_xlim(xlims)
	ax1.set_ylim(ylims)
	ax1.tick_params(axis='x', labelsize=tick_font)
	ax1.tick_params(axis='y', labelsize=tick_font)
	ax1.set_ylabel('1st PCA component', fontsize=label_font)
	ax1.ticklabel_format(style='scientific', scilimits=(0, 0), axis='x')
	ax1.set_title(title, fontsize=title_font, color='blue')
	
	# ax2.vlines(x_mean, ylims[0], ylims[1], color='r', alpha=0.5)
	# ax2.text(x_mean - 0.4e5, ylims[0] + 0.05, "slow start", color='red')
	ax2.scatter(x, z, color=color, s=mark_size)
	ax2.set_xlim(xlims)
	ax2.set_ylim(ylims)
	ax2.tick_params(axis='x', labelsize=tick_font)
	ax2.tick_params(axis='y', labelsize=tick_font)
	ax2.set_ylabel('2nd PCA component', fontsize=label_font)
	ax2.set_xlabel('Training step of ResNet-50$_{[1.0, 0.5, 0.25]}$', fontsize=label_font)
	
	# save files
	fn2 = filename.replace(".pdf", "_2D.pdf")
	print(model_dir, '====>>>>', fn2)
	fig.savefig(fn2, dpi=300, bbox_inches='tight', format='pdf')


if __name__ == "__main__":
	abs_path = '/path/to/output'

	# code for Figure.(4a) in the Paper
	# plot directions graph
	overwrite = int(sys.argv[1]) if len(sys.argv) > 1 else False
	model_dir = os.path.join(abs_path, "slim_ImageNet_39")
	filename = "eigenvector_slim_39.pdf"
	plot_pca_directions(model_dir, filename, prefix='fc_grad_layer1', overwrite=overwrite)

	# model_dir = os.path.join(abs_path, "moco_ImageNet_301_1")
	# filename = "eigenvector_301.pdf"
	# plot_pca_directions(model_dir, filename, prefix='fc_grad_layer2', overwrite=overwrite)