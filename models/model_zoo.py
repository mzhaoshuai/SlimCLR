# coding=utf-8

import logging
import torchvision.models as tv_models
from functools import partial

from . import moco, mocov3, s_moco, s_mocov3, simclr, s_simclr
from .backbones import s_resnet


def get_ssl_model(args):
	"""get ssl models"""

	if args.ssl_arch == 'mocov2':
		if not args.slimmable_training:
			logging.info("[model] creating model for MoCov2--'{}'".format(args.arch))
			model = moco.MoCo(tv_models.__dict__[args.arch],
										args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp)
		else:
			logging.info("[model] creating model for slimmable MoCov2")
			model = s_moco.s_MoCo(s_resnet.Model,
										args.moco_dim, args.moco_k, args.moco_m, args.moco_t, args.mlp,
										args=args
									)

	elif args.ssl_arch == 'mocov3':
		if not args.slimmable_training:
			logging.info("[model] creating model for MoCov3--'{}'".format(args.arch))
			model = mocov3.MoCo_ResNet(partial(tv_models.__dict__[args.arch], zero_init_residual=True),
										args.moco_dim, args.moco_mlp_dim, args.moco_t,
										init_m=args.moco_m, num_epochs=args.num_epochs, m_cos=args.moco_m_cos)
		else:
			logging.info("[model] creating model for slimmable MoCov3")
			model = s_mocov3.s_MoCoV3_ResNet(s_resnet.Model,
										args.moco_dim, args.moco_mlp_dim, args.moco_t,
										init_m=args.moco_m, num_epochs=args.num_epochs, m_cos=args.moco_m_cos,
										args=args)

	elif args.ssl_arch == 'simclr':
		if not args.slimmable_training:
			logging.info("[model] creating model for SimCLR--'{}'".format(args.arch))
			model = simclr.SimCLR(tv_models.__dict__[args.arch])
		else:
			logging.info("[model] creating model for slimmable SimCLR")
			model = s_simclr.s_SimCLR(s_resnet.Model, args=args)

	else:
		raise NotImplementedError("Model {} is not implemented".format(args.ssl_arch))

	return model
