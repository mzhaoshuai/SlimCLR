# coding=utf-8
import os
import numpy as np
import seaborn as sns
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def plot_2d_contour(xcoordinates, ycoordinates, Z, surf_name='train_loss',
					vmin=0.1, vmax=10, vlevel=0.5, show=False, path='.'):
	"""
	Plot 2D contour map and 3D surface.
	Args:
		xcoordinates: array of x-axis
		ycoordinates: array of y-axis
	"""
	X, Y = np.meshgrid(xcoordinates, ycoordinates)

	print('------------------------------------------------------------------')
	print('plot_2d_contour')
	print('------------------------------------------------------------------')
	print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(xcoordinates), len(ycoordinates)))
	print('max(%s) = %f \t min(%s) = %f' % (surf_name, np.max(Z), surf_name, np.min(Z)))
	if (len(xcoordinates) <= 1 or len(ycoordinates) <= 1):
		print('The length of coordinates is not enough for plotting contours')
		return

	# --------------------------------------------------------------------
	# Plot 2D contours
	# --------------------------------------------------------------------
	fig = plt.figure()
	CS = plt.contour(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
	plt.clabel(CS, inline=1, fontsize=8)
	fig.savefig(os.path.join(path, surf_name + '_2dcontour' + '.pdf'),
				dpi=300, bbox_inches='tight', format='pdf')

	fig = plt.figure()
	CS = plt.contourf(X, Y, Z, cmap='summer', levels=np.arange(vmin, vmax, vlevel))
	fig.savefig(os.path.join(path, surf_name + '_2dcontourf' + '.pdf'),
				dpi=300, bbox_inches='tight', format='pdf')

	# --------------------------------------------------------------------
	# Plot 2D heatmaps
	# --------------------------------------------------------------------
	fig = plt.figure()
	sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
							xticklabels=False, yticklabels=False)
	sns_plot.invert_yaxis()
	sns_plot.get_figure().savefig(os.path.join(path, surf_name + '_2dheat.pdf'),
									dpi=300, bbox_inches='tight', format='pdf')

	# --------------------------------------------------------------------
	# Plot 3D surface
	# --------------------------------------------------------------------
	fig = plt.figure()
	ax = Axes3D(fig)
	surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)
	fig.colorbar(surf, shrink=0.5, aspect=5)
	fig.savefig(os.path.join(path, surf_name + '_3dsurface.pdf'), dpi=300, bbox_inches='tight', format='pdf')

	if show: plt.show()


def plot_contour_trajectory(xcoordinates, ycoordinates, Z, proj_xcoord, proj_ycoord,
							surf_name='loss_vals', vmin=0.1, vmax=10, vlevel=0.5,\
							show=False,  path='.',
							proj_xcoord_1=None, proj_ycoord_1=None):
	"""2D contour + trajectory"""
	# plot contours
	X, Y = np.meshgrid(xcoordinates, ycoordinates)

	fig = plt.figure()
	CS1 = plt.contour(X, Y, Z, levels=np.arange(vmin, vmax, vlevel))
	# CS2 = plt.contour(X, Y, Z, levels=np.logspace(1, 8, num=8))

	# plot trajectories
	plt.plot(proj_xcoord, proj_ycoord, marker='.', color='#4B709A', markersize=10)
	# plot red points when learning rate decays
	plt.plot([proj_xcoord[-1]], [proj_ycoord[-1]], marker='x', color='r')
	
	if proj_xcoord_1 is not None and proj_ycoord_1 is not None:
		plt.plot(proj_xcoord_1, proj_ycoord_1, marker='*', color='#EB6500', alpha=0.9)
		plt.plot([proj_xcoord_1[-1]], [proj_ycoord_1[-1]], marker='x', color='r')

	# add PCA notes
	plt.xlabel('1st PCA component', fontsize='xx-large')
	plt.ylabel('2nd PCA component', fontsize='xx-large')
	plt.clabel(CS1, inline=1, fontsize=6)
	# plt.clabel(CS2, inline=1, fontsize=6)
	fig.savefig(os.path.join(path, surf_name + '_2dcontour_proj.pdf'), dpi=300,
				bbox_inches='tight', format='pdf')

	if show: plt.show()
