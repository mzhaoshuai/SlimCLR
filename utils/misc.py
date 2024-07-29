# coding=utf-8
import os
import errno
import torch
import shutil


def save_checkpoint(state, is_best, model_dir, filename='checkpoint.pth.tar'):
	filename = os.path.join(model_dir, filename)
	torch.save(state, filename)
	if is_best:
		shutil.copyfile(filename, filename.replace('pth.tar', 'best.pth.tar'))


def get_the_number_of_params(model, is_trainable=False):
	"""get the number of the model"""
	if is_trainable:
		return sum(p.numel() for p in model.parameters() if p.requires_grad)
	return sum(p.numel() for p in model.parameters())


def mkdir(path):
	try:
		os.makedirs(path)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise
