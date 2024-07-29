# coding=utf-8
import torchvision
# from .gpu_nms import gpu_nms
# from .cpu_nms import cpu_nms
# from .cpu_soft_nms import cpu_soft_nms


def nms(dets, thresh, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations."""

    return torchvision.ops.nms(dets[..., 0:4], dets[..., 4], thresh)


def soft_nms(dets, Nt=0.3, method=1, sigma=0.5, min_score=0):

    raise NotImplementedError
