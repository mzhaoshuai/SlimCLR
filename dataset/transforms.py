# coding=utf-8
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Reference:
[1] https://github.com/facebookresearch/moco/blob/main/main_moco.py
"""
import torch
import random
import numpy as np
from PIL import Image, ImageFilter, ImageOps


imagenet_pca = {
    'eigval': np.asarray([0.2175, 0.0188, 0.0045]),
    'eigvec': np.asarray([
        [-0.5675, 0.7192, 0.4009],
        [-0.5808, -0.0045, -0.8140],
        [-0.5836, -0.6948, 0.4203],
    ])
}


class Lighting(object):
    def __init__(self, alphastd,
                 eigval=imagenet_pca['eigval'],
                 eigvec=imagenet_pca['eigvec']):
        self.alphastd = alphastd
        assert eigval.shape == (3,)
        assert eigvec.shape == (3, 3)
        self.eigval = eigval
        self.eigvec = eigvec

    def __call__(self, img):
        if self.alphastd == 0.:
            return img
        rnd = np.random.randn(3) * self.alphastd
        rnd = rnd.astype('float32')
        v = rnd
        old_dtype = np.asarray(img).dtype
        v = v * self.eigval
        v = v.reshape((3, 1))
        inc = np.dot(self.eigvec, v).reshape((3,))
        img = np.add(img, inc)
        if old_dtype == np.uint8:
            img = np.clip(img, 0, 255)
        img = Image.fromarray(img.astype(old_dtype), 'RGB')
        return img

    def __repr__(self):
        return self.__class__.__name__ + '()'


class TwoCropsTransformV2:
    """Take two random crops of one image"""

    def __init__(self, base_transform1, base_transform2):
        self.base_transform1 = base_transform1
        self.base_transform2 = base_transform2

    def __call__(self, x):
        im1 = self.base_transform1(x)
        im2 = self.base_transform2(x)
        return [im1, im2]


class TwoCropsTransform:
	"""Take two random crops of one image as the query and key."""

	def __init__(self, base_transform):
		self.base_transform = base_transform

	def __call__(self, x):
		q = self.base_transform(x)
		k = self.base_transform(x)
		return [q, k]


class GaussianBlur(object):
	"""Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

	def __init__(self, sigma=[.1, 2.]):
		self.sigma = sigma

	def __call__(self, x):
		sigma = random.uniform(self.sigma[0], self.sigma[1])
		x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
		return x


class Solarize(object):
    """Solarize augmentation from BYOL: https://arxiv.org/abs/2006.07733"""

    def __call__(self, x):
        return ImageOps.solarize(x)


@torch.no_grad()
def random_solarize_normalize(images, p=0.2, threshold=128.0,
                                mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                std=[0.229 * 255, 0.224 * 255, 0.225 * 255]):
    """
    a random solarize function + normalize
    Args:
        images: torch.Tensor, [B, C, H, W]
    """
    # STEP 1. solarize
    if random.uniform(0.0, 1.0) < p:
        # inverted images
        inverted_img = 255.0 - images
        # flip value beyound the threshold
        images = torch.where(images >= threshold, inverted_img, images)

    # STEP 2. normalize
    # https://pytorch.org/vision/0.9/_modules/torchvision/transforms/functional.html#normalize
    dtype = images.dtype
    mean = torch.as_tensor(mean, dtype=dtype, device=images.device)
    std = torch.as_tensor(std, dtype=dtype, device=images.device)
    if mean.ndim == 1:
        mean = mean.view(-1, 1, 1)
    if std.ndim == 1:
        std = std.view(-1, 1, 1)
    images.sub_(mean).div_(std)
    
    return images
