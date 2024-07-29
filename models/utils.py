# coding=utf-8
import os
import sys
import torch
import logging
# import numpy as np


def sanity_check(state_dict, pretrained_weights, ssl_arch):
	"""
	Linear classifier should not change any weights other than the linear layer.
	This sanity check asserts nothing wrong happens (e.g., BN stats updated).
	"""
	print("=> loading '{}' for sanity check".format(pretrained_weights))
	checkpoint = torch.load(pretrained_weights, map_location="cpu")
	state_dict_pre = checkpoint['state_dict']

	for k in list(state_dict.keys()):
		# only ignore fc layer
		if 'fc.weight' in k or 'fc.bias' in k or 'fc.linear' in k:
			continue

		# name in pretrained model
		if ssl_arch == 'mocov2':
			k_pre = 'module.encoder_q.' + k[len('module.'):] \
				if k.startswith('module.') else 'module.encoder_q.' + k
		elif ssl_arch == 'mocov3':
			k_pre = 'module.base_encoder.' + k[len('module.'):] \
				if k.startswith('module.') else 'module.base_encoder.' + k
		elif ssl_arch == 'simclr':
			k_pre = 'module.encoder_q.' + k[len('module.'):] \
				if k.startswith('module.') else 'module.encoder_q.' + k       
		else:
			raise NotImplementedError

		assert ((state_dict[k].cpu() == state_dict_pre[k_pre]).all()), \
			'{} is changed in linear classifier training.'.format(k)

	print("=> sanity check passed.")


def load_ssl_pretrained(model, args):
	if args.lincls_pretrained:
		if os.path.isfile(args.lincls_pretrained):
			logging.info("[model] loading checkpoint '{}'".format(args.lincls_pretrained))
			checkpoint = torch.load(args.lincls_pretrained, map_location="cpu")
			# rename moco pre-trained keys
			state_dict = checkpoint['state_dict']
			for k in list(state_dict.keys()):

				# remove prefix
				if args.ssl_arch == 'mocov2':
					if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
						state_dict[k[len("module.encoder_q."):]] = state_dict[k]
				
				elif args.ssl_arch == 'mocov3':
					if k.startswith('module.base_encoder') and not k.startswith('module.base_encoder.fc'):
						state_dict[k[len("module.base_encoder."):]] = state_dict[k]
				
				elif args.ssl_arch == 'simclr':
					if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
						state_dict[k[len("module.encoder_q."):]] = state_dict[k]

				else:
					raise NotImplementedError('The ssl arch {} is not implemented.'.format(args.ssl_arch))

				# delete renamed or unused k
				del state_dict[k]

			msg = model.load_state_dict(state_dict, strict=False)
			assert set(msg.missing_keys) == {"fc.weight", "fc.bias"} or \
				all(['fc.linear' in k for k in msg.missing_keys]), "wrong missing keys {}".format(msg.missing_keys)
			logging.info("[model] loaded pre-trained model '{}'".format(args.lincls_pretrained))
		else:
			logging.info("[model] no checkpoint found at '{}'".format(args.lincls_pretrained))
			raise IOError
	else:
		logging.info("[model] Do not load pretrained model from {}".format(args.lincls_pretrained))
		raise IOError


@torch.no_grad()
def get_grad_norm(model, sorted_width_mult_list, global_step,
					tf_writer=None, args=None):
	"""calculate the gradient norm"""
	uppper = 1000.0
	max_width, min_width = max(sorted_width_mult_list), min(sorted_width_mult_list)

	if hasattr(model, 'module'):
		model_p = model.module
		if hasattr(model_p, 'encoder_q'):
			model_p = model.module.encoder_q
	else:
		model_p = model

	norms_all = [list() for _ in sorted_width_mult_list]
	norms_sum = [0.0 for _ in sorted_width_mult_list]
	cnt = 0.0
	for n, p in model_p.named_parameters():
		if p.requires_grad and 'weight' in n and 'bn' not in n:
			for i, w in enumerate(sorted_width_mult_list):
				dim1, dim2 = int(w * p.shape[0]), int(w * p.shape[1])
				# linear layer
				if 'fc' in n: dim1 = p.shape[0]
				# first conv
				if p.shape[1] == 3: dim2 = p.shape[1]

				norms_all[i].append(p.grad.data[:dim1, :dim2, ...].norm(2).clamp_(max=uppper).item())

			# no NaN gradient for any gradient
			if all([l[-1] < uppper for l in norms_all]):
				for i, l in enumerate(norms_all):
					norms_sum[i] = norms_sum[i] + l[-1]	
				cnt += 1.0

	if cnt > 0.5:
		norms_mean = [x / cnt for x in norms_sum]
	else:
		norms_mean = [1000.0 for _ in sorted_width_mult_list]

	with open(args.grad_log, mode='a+') as f:
		f.writelines([str(global_step), ' ', ' '.join([str(x) for x in norms_mean]), '\n'])
		for norm_list in norms_all:
			f.writelines(["{:.5f}".format(x) + ' ' for x in norm_list] + ['\n'])

	if tf_writer is not None:
		for i, w in enumerate(sorted_width_mult_list):
			tf_writer.add_scalar("train/mean_norm_{}".format(w), norms_mean[i], global_step=global_step)
		tf_writer.add_scalar("train/fc_all_norm", norms_all[0][-2], global_step=global_step)
		tf_writer.add_scalar("train/fc_min_norm", norms_all[-1][-2], global_step=global_step)


@torch.no_grad()
def save_last_fc_grad(model,  global_step, tf_writer=None, args=None, sorted_width_mult_list=None):
	"""save the gradient of fc layers"""
	if hasattr(model, 'module'):
		model_p = model.module
		if hasattr(model_p, 'encoder_q'):
			model_p = model.module.encoder_q
	else:
		model_p = model

	name_prefix = "fc"
	counter = 1
	for n, p in model_p.named_parameters():
		if p.requires_grad and 'fc' in n and 'weight' in n :
			filename = os.path.join(args.log_dir, "{}_grad_layer{}_step{}.pth".format(name_prefix, counter, global_step))
			torch.save(p.grad.data, filename)

			U, S, Vh = torch.linalg.svd(p.grad.data, full_matrices=False)		
			if tf_writer is not None:
				tf_writer.add_scalar("train/{}_grad_layer{}_1st_eig".format(name_prefix, counter), S[0].item(), global_step=global_step)
				tf_writer.add_scalar("train/{}_grad_layer{}_2nd_eig".format(name_prefix, counter), S[1].item(), global_step=global_step)				

			counter += 1


def loss_weights_generate(slim_loss_weighted, width_mult_list=[1.0]):
	"""generate loss weights for multiple networks
	choice for weighting manner:
	0. all weights are 1.0
	1. width / sum(width_mult_list)
	2. width
	3. width / sum(width_mult_list) X numbers_of_networks
	5. w for max_width is {1.0 + sum(width_mult_list) - max_width}, else 1.0
	6. w for width is (sum_of_rest_width + 1.0) then normalize to have sum sum(width_mult_list)
	7. w for max_width is {1.0 + max(width_mult_list - max_width), else 1.0
	"""
	max_width = max(width_mult_list)

	if slim_loss_weighted == 1:
		loss_weights = [round(w / sum(width_mult_list), 3) for w in width_mult_list]

	elif slim_loss_weighted == 2:
		loss_weights = [float(w) for w in width_mult_list]

	elif slim_loss_weighted == 3:
		num = len(width_mult_list) * 1.0
		loss_weights = [round(w / sum(width_mult_list) * num, 3) for w in width_mult_list]

	elif slim_loss_weighted == 5:
		max_w = 1.0 + sum(width_mult_list) - max_width
		loss_weights = [max_w if w == max_width else 1.0 for w in width_mult_list]

	elif slim_loss_weighted == 6:
		all_w = 1.0 * len(width_mult_list)
		w_l = []
		for i, w in enumerate(width_mult_list):
			# if i != len(self.width_mult_list) - 1:
			# 	w_l.append(sum(self.width_mult_list[i+1:]) / self.width_mult_list[i] + 1.0)
			# else:
			# 	w_l.append(1.0)
			w_l.append(sum(width_mult_list) - sum(width_mult_list[:i+1]) + 1.0)

		loss_weights = [round(w / sum(w_l) * all_w, 3) for w in w_l]

	elif slim_loss_weighted == 7:
		cp_width_mult_list = width_mult_list.copy()
		cp_width_mult_list.remove(max_width)
		max_w = 1.0 + max(cp_width_mult_list)
		loss_weights = [max_w if w == max_width else 1.0 for w in width_mult_list]

	else:
		loss_weights = [1.0 for w in width_mult_list]

	return loss_weights
