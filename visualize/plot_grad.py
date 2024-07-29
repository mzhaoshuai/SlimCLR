# coding=utf-8
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
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


def plot_lines(txt_file=None, filename=None, title=None, title_fontsize=16,
				show_y=False, acc1=r'77.6', acc2=r'77.6', min_v=0.0, max_v=4.0):
	f = open(txt_file, mode='r')
	content = [x.strip().split(' ') for x in f.readlines()]

	means = []
	for x in content:
		if len(x) == 4:
			means.append([float(i) for i in x])
	means_np = np.array(means)

	# fix legacy bug
	means_np[:, 3] = np.power(np.power(means_np[:, 1], 2.0) - np.power(means_np[:, 2], 2.0), 0.5)
	# smooth the curve
	ticks = 100
	x1, y1 = spline(means_np[:, 0], means_np[:, 1], new_ticks=ticks)
	x2, y2 = spline(means_np[:, 0], means_np[:, 2], new_ticks=ticks)
	x3, y3 = spline(means_np[:, 0], means_np[:, 3], new_ticks=ticks)
	# plot
	fig, ax = plt.subplots()  # Create a figure and an axes.
	# color plan 1
	# ax.plot(x1, y1, label=r'$ResNet_{1.0}$', color='#29abe3') 
	# ax.plot(x1, y1, label=r'$ResNet_{1.0}$', color='#f27970', linestyle='dashed', linewidth=2)
	# ax.plot(x2, y2, label=r'$ResNet_{0.25}$', color='#29abe3', linewidth=1.5)
	# ax.plot(x3, y3, label=r'$ResNet_{1.0} - ResNet_{0.25}$', color='#f7801f', linewidth=1.5)
	# color plan 2
	# ax.plot(x1, y1, label=r'$ResNet_{1.0}$', color='#8ecfc9', linestyle='dashed', linewidth=2)
	# ax.plot(x2, y2, label=r'$ResNet_{0.25}$', color='#ffbe7a', linewidth=1.5)
	# ax.plot(x3, y3, label=r'$ResNet_{1.0} - ResNet_{0.25}$', color='#82b0d2', linewidth=1.5)
	# color plan 3
	# https://colorbrewer2.org/#type=qualitative&scheme=Pastel1&n=3
	# https://colorsupplyyy.com/app
	ax.plot(x1, y1, label=r'$\theta_{1.0}, ~' + acc1 + r'\%$', color='#fc8d62', linestyle=linestyle_tuple[2][1], linewidth=2.5)
	ax.plot(x2, y2, label=r'$\theta_{0.25}, ' + acc2 + r'\%$', color='#66c2a5', linewidth=2)
	# ax.plot(x3, y3, label=r'$\theta_{1.0} \backslash \theta_{0.25}$', color='#8da0cb', linewidth=2)
	ax.plot(x3, y3, label=r'$\theta_{1.0 \backslash 0.25}$', color='#8da0cb', linewidth=2)
	# color plan 4
	# https://colorbrewer2.org/#type=qualitative&scheme=Pastel1&n=3
	# ax.plot(x1, y1, label=r'$ResNet_{1.0}$', color='#66c2a5', linestyle='dotted', linewidth=2.2)
	# ax.plot(x2, y2, label=r'$ResNet_{0.25}$', color='#fc8d62', linewidth=1.5)
	# ax.plot(x3, y3, label=r'$ResNet_{1.0} - ResNet_{0.25}$', color='#8da0cb', linewidth=1.5)

	ax.set_xlabel('Training step of ResNet-50$_{[1.0, 0.5, 0.25]}$', fontsize=title_fontsize)
	if show_y:
		ax.set_ylabel(r'$Average~\ell_2$ norm of gradients', fontsize=title_fontsize)
	ax.set_title(title, fontsize=title_fontsize, color='blue')
	ax.yaxis.set_minor_locator(AutoMinorLocator(3))
	ax.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax.grid(axis='y', which='major', linestyle='-.', linewidth=0.5)
	ax.grid(axis='x', which='major', linestyle='-.', linewidth=0.5)
	ax.set_xlim([0, 252000])
	ax.set_ylim(min_v, max_v)
	ax.legend(loc='upper left', prop={'size': title_fontsize})

	ax.tick_params(axis='x', labelsize=14)
	ax.tick_params(axis='y', labelsize=14)
	ax.ticklabel_format(style='scientific', scilimits=(0, 0), axis='x')
	ax.xaxis.get_offset_text().set_fontsize(14)

	print(txt_file, '====>>>>', filename)
	fig.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')


def plot_lines_v2_sub2(txt_file=None, filename=None, title='Supervised', title_fontsize=16, show_y=True,
						min_v=0.0, max_v=1.0):
	"""plot grad for supervised training"""
	# plt.rc('font', size=14)
	f = open(txt_file, mode='r')
	content = [x.strip().split(' ') for x in f.readlines()]

	means = []
	for x in content:
		if len(x) == 4:
			means.append([float(i) for i in x])
	means_np = np.array(means)
	# calculate the grad norm of ResNet1.0 - ResNet0.25
	means_np[:, 2] = np.power(np.power(means_np[:, 1], 2.0) - np.power(means_np[:, 3], 2.0), 0.5)
	# smooth the curve
	ticks = 100
	x1, y1 = spline(means_np[:, 0], means_np[:, 1], new_ticks=ticks)
	x2, y2 = spline(means_np[:, 0], means_np[:, 3], new_ticks=ticks)
	x3, y3 = spline(means_np[:, 0], means_np[:, 2], new_ticks=ticks)
	# plot
	fig, ax = plt.subplots()  # Create a figure and an axes.
	# color plan 3
	# https://colorbrewer2.org/#type=qualitative&scheme=Pastel1&n=3
	# https://colorsupplyyy.com/app
	# ax.plot(x1, y1, label=r'$\theta_{1.0}, 76.0\%$', color='#fc8d62', linestyle=linestyle_tuple[2][1], linewidth=2.5)
	# ax.plot(x2, y2, label=r'$\theta_{0.25}, 64.4\%$', color='#66c2a5', linewidth=2)
	# ax.plot(x3, y3, label=r'$\theta_{1.0 \backslash 0.25}$', color='#8da0cb', linewidth=2)

	ax.plot(x1, y1, label=r'$\theta_{1.0}, 76.6\%$', color='#fc8d62', linestyle=linestyle_tuple[2][1], linewidth=2.5)
	ax.plot(x2, y2, label=r'$\theta_{0.25}, N/A\%$', color='#66c2a5', linewidth=2)
	ax.plot(x3, y3, label=r'$\theta_{1.0 \backslash 0.25}$', color='#8da0cb', linewidth=2)

	ax.set_xlabel('Training step of ResNet-50', fontsize=title_fontsize)
	# ax.set_xlabel('Training step of ResNet-50$_{[1.0, 0.5, 0.25]}$', fontsize=title_fontsize)
	if show_y:
		ax.set_ylabel(r'$Average~\ell_2$ norm of gradients', fontsize=title_fontsize)
	ax.set_title(title, fontsize=title_fontsize)

	ax.yaxis.set_minor_locator(AutoMinorLocator(3))
	ax.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax.grid(axis='y', which='major', linestyle='-.', linewidth=0.5)
	ax.grid(axis='x', which='major', linestyle='-.', linewidth=0.5)
	ax.set_xlim([0, 500400])
	ax.set_ylim(min_v, max_v)
	ax.legend(loc='upper left', prop={'size': title_fontsize})

	ax.tick_params(axis='x', labelsize=14)
	ax.tick_params(axis='y', labelsize=14)
	ax.ticklabel_format(style='scientific', scilimits=(0, 0), axis='x')
	ax.xaxis.get_offset_text().set_fontsize(14)
	ax.tick_params(axis='x', labelsize=14)

	print(txt_file, '====>>>>', filename)
	fig.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')


def plot_lines_v2_sub2_ratio(txt_file=None, filename=None, title='Supervised', title_fontsize=16, show_y=True,
								min_v=0.0, max_v=1.5):
	"""plot grad for supervised training"""
	# plt.rc('font', size=14)
	f = open(txt_file, mode='r')
	content = [x.strip().split(' ') for x in f.readlines()]

	means = []
	for x in content:
		if len(x) == 4:
			means.append([float(i) for i in x])
	means_np = np.array(means)
	# calculate the grad norm of ResNet1.0 - ResNet0.25
	means_np[:, 2] = np.power(np.power(means_np[:, 1], 2.0) - np.power(means_np[:, 3], 2.0), 0.5)
	# smooth the curve
	ticks = 100
	x1, y1_raw = spline(means_np[:, 0], means_np[:, 1], new_ticks=ticks)
	x2, y2_raw = spline(means_np[:, 0], means_np[:, 3], new_ticks=ticks)
	x3, y3_raw = spline(means_np[:, 0], means_np[:, 2], new_ticks=ticks)
	y2 = y3_raw / y2_raw
	y3 = y1_raw / y2_raw
	# plot
	fig, ax = plt.subplots()  # Create a figure and an axes.
	# ax.set_xlabel('Training step of ResNet-50$_{[1.0, 0.5, 0.25]}$', fontsize=title_fontsize)
	ax.set_xlabel('Training step of ResNet-50', fontsize=title_fontsize)
	if show_y:
		ax.set_ylabel(r'Ratios of gradient norms', fontsize=title_fontsize)
	ax.set_title(title, fontsize=title_fontsize)

	ax.yaxis.set_minor_locator(AutoMinorLocator(2))
	ax.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax.grid(axis='y', which='major', linestyle='-.', linewidth=0.5)
	ax.grid(axis='x', which='major', linestyle='-.', linewidth=0.5)
	ax.axhline(y=1.0, linestyle='-.', linewidth=1.5, color='red')

	# color plan 3
	# ax.plot(x1, y1, label=r'$\Theta_{1.0},~76.0\%$', color='#fc8d62', linestyle=linestyle_tuple[2][1], linewidth=2.5)
	ax.plot(x3, y3, label=r'$|| \nabla_{\theta_{1.0}} \mathcal{L} ||_2 ~/~ || \nabla_{\theta_{0.25}} \mathcal{L}||_2$',
				color='#8da0cb', linewidth=2)	
	ax.plot(x2, y2, label=r'$|| \nabla_{\theta_{1.0 \backslash 0.25}} \mathcal{L} ||_2 ~/~ || \nabla_{\theta_{0.25}} \mathcal{L}||_2$',
				color='#66c2a5', linewidth=2)

	ax.set_xlim([0, 500400])
	ax.set_ylim(min_v, max(0.1, max_v))
	ax.legend(loc='lower left', prop={'size': 16})
	# ax.legend(loc='lower left', prop={'size': 12})

	ax.tick_params(axis='x', labelsize=14)
	ax.tick_params(axis='y', labelsize=14)
	ax.ticklabel_format(style='scientific', scilimits=(0, 0), axis='x')
	ax.xaxis.get_offset_text().set_fontsize(14)
	ax.tick_params(axis='x', labelsize=14)

	print(txt_file, '====>>>>', filename)
	fig.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')


def plot_lines_ratio(txt_file=None, filename=None, title=None, title_fontsize=16,
				show_y=False, acc1=r'77.6', acc2=r'77.6', min_v=0.0, max_v=2.0):
	f = open(txt_file, mode='r')
	content = [x.strip().split(' ') for x in f.readlines()]

	means = []
	for x in content:
		if len(x) == 4:
			means.append([float(i) for i in x])
	means_np = np.array(means)

	# fix legacy bug
	means_np[:, 3] = np.power(np.power(means_np[:, 1], 2.0) - np.power(means_np[:, 2], 2.0), 0.5)
	# smooth the curve
	ticks = 100
	x1, y1_raw = spline(means_np[:, 0], means_np[:, 1], new_ticks=ticks)
	x2, y2_raw = spline(means_np[:, 0], means_np[:, 2], new_ticks=ticks)
	x3, y3_raw = spline(means_np[:, 0], means_np[:, 3], new_ticks=ticks)
	y2 = y3_raw / y2_raw
	y3 = y1_raw / y2_raw
	# plot
	fig, ax = plt.subplots()  # Create a figure and an axes.
	ax.set_xlabel('Training step of ResNet-50$_{[1.0, 0.5, 0.25]}$', fontsize=title_fontsize)
	if show_y:
		ax.set_ylabel(r'Ratios of gradient norms', fontsize=title_fontsize)
	ax.set_title(title, fontsize=title_fontsize, color='blue')
	ax.yaxis.set_minor_locator(AutoMinorLocator(2))
	ax.xaxis.set_minor_locator(AutoMinorLocator(2))
	ax.grid(axis='y', which='major', linestyle='-.', linewidth=0.5)
	ax.grid(axis='x', which='major', linestyle='-.', linewidth=0.5)
	ax.axhline(y=1.0, linestyle='-.', linewidth=1.5, color='red')

	# color plan 3
	# https://colorbrewer2.org/#type=qualitative&scheme=Pastel1&n=3
	# https://colorsupplyyy.com/app
	ax.plot(x3, y3, label=r'$|| \nabla_{\theta_{1.0}} \mathcal{L} ||_2 ~/~ || \nabla_{\theta_{0.25}} \mathcal{L}||_2$',
				color='#8da0cb', linewidth=2)
	ax.plot(x2, y2, label=r'$|| \nabla_{\theta_{1.0 \backslash 0.25}} \mathcal{L} ||_2 ~/~ || \nabla_{\theta_{0.25}} \mathcal{L}||_2$',
				color='#66c2a5', linewidth=2)

	ax.set_xlim([0, 252000])
	ax.set_ylim(min_v, max(0.35, max_v))
	# ax.legend(loc='lower left', prop={'size': 12})
	ax.legend(loc='lower left', prop={'size': 16})
	ax.tick_params(axis='x', labelsize=14)
	ax.tick_params(axis='y', labelsize=14)
	ax.ticklabel_format(style='scientific', scilimits=(0, 0), axis='x')
	ax.xaxis.get_offset_text().set_fontsize(14)

	print(txt_file, '====>>>>', filename)
	fig.savefig(filename, dpi=300, bbox_inches='tight', format='pdf')


if __name__ == "__main__":
	# code for Figure (3) in the paper

	# ratio, sub1
	# txt_file = '/home/shuai/models/ssl/slimssl_38/grad_38.txt'
	# filename = txt_file.replace('.txt', '_ratio.pdf')
	# plot_lines_v2_sub2_ratio(txt_file, filename, title='Supvervised, normal',
	# 							show_y=True, min_v=0.0, max_v=4.0)

	# ratio, sub 2
	# txt_file = '/home/shuai/models/ssl/slimssl_37/grad_37.txt'
	# filename = txt_file.replace('.txt', '_ratio.pdf')
	# plot_lines_v2_sub2_ratio(txt_file, filename, title='Supvervised, slimmable',
	# 							show_y=False, min_v=0.4, max_v=1.6)

	# # ratio, sub3
	txt_file = '/home/shuai/models/slim_grad/grad_49_new.txt'
	filename = txt_file.replace('.txt', '_ratio.pdf')
	plot_lines_ratio(txt_file, filename, title='SlimCLR-MoCov2, vanilla baseline',
						show_y=False, acc1=r'66.3', acc2=r'55.1', min_v=0.4, max_v=1.6)

	# # ratio, sub4
	# txt_file = '/home/shuai/models/slim_grad/grad_65_new.txt'
	# filename = txt_file.replace('.txt', '_ratio.pdf')
	# plot_lines_ratio(txt_file, filename, title='SlimCLR-MoCov2, slow start',
	# 					show_y=True, acc1=r'66.3', acc2=r'55.1', min_v=0.0, max_v=4.0)

	# # ratio, sub5
	# txt_file = '/home/shuai/models/slim_grad/grad_100_new.txt'
	# filename = txt_file.replace('.txt', '_ratio.pdf')
	# plot_lines_ratio(txt_file, filename, title='SlimCLR-MoCov2, slow start + distill',
	# 					show_y=False, acc1=r'66.3', acc2=r'55.1', min_v=0.0, max_v=4.0)

	# # ratio, sub6
	# txt_file = '/home/shuai/models/slim_grad/grad_91_new.txt'
	# filename = txt_file.replace('.txt', '_ratio.pdf')
	# plot_lines_ratio(txt_file, filename, title='SlimCLR-MoCov2, slow start + distill + reweight',
	# 					show_y=False, acc1=r'66.3', acc2=r'55.1', min_v=0.0, max_v=4.0)