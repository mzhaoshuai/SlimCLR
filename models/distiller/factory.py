# coding=utf-8
import torch
import torch.nn.functional as F
from .DKD import dkd_distill_loss
from .ATKD import atkd_distill_loss
from .KD import kd_distill_loss_v2


def choose_kd_loss(logits_student, logits_teacher, stu_T=1.0, tea_T=1.0, labels=None,
						mse=0, kl_div=0, dkd=0, atkd=0):
	"""choose one type of kd losses"""

	if mse:
		d_loss = F.mse_loss(logits_student, logits_teacher, reduction='mean').mean()

	elif kl_div:
		d_loss = kd_distill_loss_v2(logits_student, logits_teacher, T_stu=stu_T, T_tea=tea_T)

	elif dkd:
		# default setting of dkd on ImageNet is beta=0.5, alpha=1.0, T=1.0
		d_loss = dkd_distill_loss(logits_student, logits_teacher, labels, 
									alpha=1.0, beta=0.5, temperature=1.0).mean()

	elif atkd:
		d_loss = atkd_distill_loss(logits_student, logits_teacher, multiplier=2.0, temperature=4.0,
										version=2).mean()        

	else:
		d_loss = torch.zeros(1, dtype=logits_student.dtype, device=logits_student.device).mean()

	return d_loss
  