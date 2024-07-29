# coding=utf-8
"""
Reference:
[1] https://raw.githubusercontent.com/open-mmlab/mmselfsup/master/mmselfsup/core/optimizer/optimizers.py
[2] https://github.com/NUS-HPC-AI-Lab/LARS-ImageNet-PyTorch/blob/main/lars.py
"""
import torch


class LARS(torch.optim.Optimizer):
    """
    LARS optimizer, no rate scaling or weight decay for parameters <= 1D.
	https://github.com/facebookresearch/moco-v3/blob/main/moco/optimizer.py
    """
    def __init__(self, params, lr=0, weight_decay=0, momentum=0.9, trust_coefficient=0.001):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum, trust_coefficient=trust_coefficient)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if p.ndim > 1: # if not normalization gamma/beta or bias
                    dp = dp.add(p, alpha=g['weight_decay'])
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                    (g['trust_coefficient'] * param_norm / update_norm), one),
                                    one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)
                p.add_(mu, alpha=-g['lr'])


def get_parameter_groups(model, lr, weight_decay, norm_weight_decay=0,
							norm_bias_no_decay=True):
	"""
	Separate model parameters from scale and bias parameters following norm if
	training imagenet
	"""
	if norm_bias_no_decay:
		wd_params = []
		no_wd_params = []

		for name, p in model.named_parameters():
			if p.requires_grad:
				# if 'fc' not in name and ('norm' in name or 'bias' in name):
				if 'norm' in name or 'Norm' in name or 'bias' in name:
					no_wd_params += [p]
				else:
					wd_params += [p]

		return [
				{'params': wd_params, 'lr': lr, 'lr_mult': 1.0, 'weight_decay': weight_decay, 'decay_mult': 1.0,},
				{'params': no_wd_params, 'lr': lr, 'lr_mult': 1.0, 'weight_decay': norm_weight_decay, 'decay_mult': 1.0,},
			]

	else:
		wd_params = []
		for name, p in model.named_parameters():
			if p.requires_grad:
				wd_params.append(p)
		return [
				{'params': wd_params, 'lr': lr, 'lr_mult': 1.0, 'weight_decay': weight_decay, 'decay_mult': 1.0,},
		]
