# coding=utf-8
import os
import torch
import logging
from torchvision import datasets, transforms
from torch.utils.data.distributed import DistributedSampler
from .imagenet_lmdb import ImageFolderLMDB
from .transforms import GaussianBlur, Solarize, Lighting
from .transforms import TwoCropsTransform, TwoCropsTransformV2


def get_dataloader(args=None):
	"""get training dataloader"""
	if args.dali_data_dir is not None and os.path.exists(args.dali_data_dir):
		from .imagenet_dali import get_dali_dataloader

		if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
			logging.info('INFO: [dataset] DALI {} as data provider for augmentations {}'.format(
							args.dali_data_dir, args.ssl_aug))

		train_loader, val_loader, test_loader = get_dali_dataloader(args)

	else:
		train_transforms, val_transforms, test_transforms = data_transforms(args)
		train_set, val_set, test_set = dataset(train_transforms,
												val_transforms,
												test_transforms,
												args=args)
		train_loader, val_loader, test_loader = data_loader(train_set,
															val_set,
															test_set,
															args=args)

	return train_loader, val_loader, test_loader


def data_loader(train_set, val_set, test_set, args=None):
	"""get data loader"""
	train_loader, val_loader, test_loader = None, None, None
	batch_size = args.batch_size_per_gpu
	logging.info('[dataset] use batch size {} and workers {}'.format(batch_size, args.num_workers))

	if args.data_loader == 'imagenet1k_basic':
		if torch.distributed.is_available() and torch.distributed.is_initialized():
			train_sampler = DistributedSampler(train_set, shuffle=True)
			# val_sampler = DistributedSampler(val_set)
		else:
			train_sampler = None
		val_sampler = None

		train_loader = torch.utils.data.DataLoader(
			train_set,
			batch_size=batch_size,
			shuffle=(train_sampler is None),
			sampler=train_sampler,
			pin_memory=True,
			num_workers=args.num_workers,
			drop_last=True)

		val_loader = torch.utils.data.DataLoader(
			val_set,
			batch_size=batch_size,
			shuffle=False,
			sampler=val_sampler,
			pin_memory=True,
			num_workers=args.num_workers,
			drop_last=False)

		test_loader = val_loader
	else:
		raise NotImplementedError('Data loader {} is not yet implemented.'.format(args.data_loader))

	return train_loader, val_loader, test_loader


def dataset(train_transforms, val_transforms, test_transforms, args=None):
	"""get ImageNet dataset for classification"""
	if args.lmdb_dataset in [None, 'None']:
		train_set = datasets.ImageFolder(os.path.join(args.dataset_dir, 'train'),
												transform=train_transforms)
		val_set = datasets.ImageFolder(os.path.join(args.dataset_dir, 'val'),
												transform=val_transforms)
		test_set = None
	else:
		assert os.path.exists(args.lmdb_dataset), "{} does not exist".format(args.lmdb_dataset)
		if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
			logging.info('INFO: [dataset] Using {} as data source for augmentations {}'.format(
							args.lmdb_dataset, args.ssl_aug))
		train_set = ImageFolderLMDB(args.dataset_dir,
									lmdb_dataset=os.path.join(args.dataset_dir, 'train_lmdb/train.lmdb'),
									key_label_json=os.path.join(args.dataset_dir, 'train_list.json'),
									transform=train_transforms,
									target_transform=None,
									albumentations=args.albumentations)
		val_set = ImageFolderLMDB(args.dataset_dir,
									lmdb_dataset=os.path.join(args.dataset_dir, 'val_lmdb/val.lmdb'),
									key_label_json=os.path.join(args.dataset_dir, 'val_list.json'),
									transform=val_transforms,
									target_transform=None,
									albumentations=args.albumentations)
		test_set = None

	return train_set, val_set, test_set


def data_transforms(args=None):
	"""get transform of dataset"""
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])
	if args.ssl_aug == 'mocov3':
		normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
										std=[0.229, 0.224, 0.225])

		# follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733
		augmentation1 = [
			transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
			transforms.RandomApply([
				transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
			], p=0.8),
			transforms.RandomGrayscale(p=0.2),
			transforms.RandomApply([GaussianBlur([.1, 2.])], p=1.0),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		]

		augmentation2 = [
			transforms.RandomResizedCrop(224, scale=(args.crop_min, 1.)),
			transforms.RandomApply([
				transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)  # not strengthened
			], p=0.8),
			transforms.RandomGrayscale(p=0.2),
			transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.1),
			transforms.RandomApply([Solarize()], p=0.2),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		]

		train_transforms = TwoCropsTransformV2(transforms.Compose(augmentation1), 
                                      			transforms.Compose(augmentation2))
	elif args.ssl_aug == 'mocov2':
		# MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
		augmentation = [
			transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
			transforms.RandomApply([
				transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
			], p=0.8),
			transforms.RandomGrayscale(p=0.2),
			transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		]
		train_transforms = TwoCropsTransform(transforms.Compose(augmentation))

	elif args.ssl_aug == 'mocov1':
		# MoCo v1's aug: the same as InstDisc https://arxiv.org/abs/1805.01978
		augmentation = [
			transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
			transforms.RandomGrayscale(p=0.2),
			transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		]
		train_transforms = TwoCropsTransform(transforms.Compose(augmentation))

	elif args.ssl_aug == 'simclr':
		# A. Data Augmentation Details in the simclr paper
		augmentation = [
			transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
			transforms.RandomApply([
				transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)  # not strengthened
			], p=0.8),
			transforms.RandomGrayscale(p=0.2),
			transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize
		]
		train_transforms = TwoCropsTransform(transforms.Compose(augmentation))		

	elif args.ssl_aug == 'slim':
		mean = [0.485, 0.456, 0.406]
		std = [0.229, 0.224, 0.225]
		crop_scale = 0.08
		jitter_param = 0.4
		lighting_param = 0.1

		train_transforms = transforms.Compose([
			transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
			transforms.ColorJitter(
				brightness=jitter_param, contrast=jitter_param,
				saturation=jitter_param),
			Lighting(lighting_param),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=mean, std=std),
		])		

	elif args.ssl_aug == 'normal':
		train_transforms = transforms.Compose([
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			normalize,
		])

	else:
		raise NotImplementedError
	
	val_transforms = transforms.Compose([
		transforms.Resize(256),
		transforms.CenterCrop(224),
		transforms.ToTensor(),
		normalize
	])

	test_transforms = val_transforms

	return train_transforms, val_transforms, test_transforms
