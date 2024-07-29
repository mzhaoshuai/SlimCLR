# coding=utf-8
"""
NVIDIA DALI dataloader

Reference:
[1] https://github.com/NVIDIA/DALI/blob/2b25d340983ad2b6ee8297d718c39a725b0f7453/docs/examples/use_cases/pytorch/resnet50/main.py
[2] https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Classification/ConvNets/image_classification/dataloaders.py
"""

import os
import torch
from glob import glob

try:
	import nvidia.dali.fn as fn		
	import nvidia.dali.types as types
	from nvidia.dali.pipeline import pipeline_def
	from nvidia.dali.plugin.pytorch import DALIClassificationIterator, LastBatchPolicy, DALIGenericIterator
except ImportError:
	raise ImportError("Please install DALI from https://www.github.com/NVIDIA/DALI to run this example.")

_interp_dict = {
					"bicubic": types.INTERP_CUBIC,
					"bilinear": types.INTERP_LINEAR,
					"triangular": types.INTERP_TRIANGULAR,
				}
_interp = _interp_dict["bicubic"]


def get_dali_dataloader(args):
	"""get dali dataloader"""
	if torch.distributed.is_initialized():
		rank = torch.distributed.get_rank()
		world_size = torch.distributed.get_world_size()
	else:
		rank = 0
		world_size = args.world_size
	batch_size = args.batch_size_per_gpu
	
	pipe = create_dali_pipeline(batch_size=batch_size,
								num_threads=args.num_workers,
								device_id=rank,
								seed=12 + rank,
								data_dir=args.dali_data_dir,
								crop=args.image_size,
								size=args.image_size,
								dali_cpu=args.dali_cpu,
								shard_id=rank,
								num_shards=world_size,
								is_training=True,
								precision=args.precision,
								ssl_aug=args.ssl_aug,
								width_mult_list=args.width_mult_list)
	pipe.build()
	if args.ssl_aug in ['normal', 'val', 'slim']:
		train_loader = DALIClassificationIterator(pipe, reader_name="Reader",
													last_batch_policy=LastBatchPolicy.DROP, auto_reset=True)
	elif args.ssl_aug in ['mocov2', 'mocov3']:
		train_loader = DALIGenericIterator(pipe, ["data_q", "data_k", "label"], reader_name="Reader",
													last_batch_policy=LastBatchPolicy.DROP, auto_reset=True)
	elif args.ssl_aug in ['mocov2_sweet', 'mocov2_diff']:
		keys = ["data_q", "data_k"] + ["data_k{}".format(i) for i in range(1, len(args.width_mult_list))]
		train_loader = DALIGenericIterator(pipe, keys + ["label",], reader_name="Reader",
													last_batch_policy=LastBatchPolicy.DROP, auto_reset=True)
	else:
		raise NotImplementedError

	pipe = create_dali_pipeline(batch_size=batch_size,
								num_threads=args.num_workers,
								device_id=rank,
								seed=12 + rank,
								data_dir=args.dali_data_dir,
								crop=args.image_size,
								size=int(args.image_size * 256 / 224),
								dali_cpu=args.dali_cpu,
								shard_id=0,					# we only evaluate on device with rank=0
								num_shards=1,
								is_training=False,
								precision='fp32',
								ssl_aug='val')
	pipe.build()
	val_loader = DALIClassificationIterator(pipe, reader_name="Reader",
												last_batch_policy=LastBatchPolicy.PARTIAL, auto_reset=True)
	
	return train_loader, val_loader, val_loader


@pipeline_def
def create_dali_pipeline(data_dir=None, crop=224, size=256, shard_id=0, num_shards=1, dali_cpu=False,
							is_training=True, prefetch=2, precision='amp', ssl_aug='mocov2',
							width_mult_list=[1.0, 0.5]):
	# load from folders
	# images, labels = fn.readers.file(file_root=data_dir,
	# 								 shard_id=shard_id,
	# 								 num_shards=num_shards,
	# 								 random_shuffle=is_training,
	# 								 pad_last_batch=True,
	# 								 name="Reader")

	# load from MXNet RecordIO
	if is_training:
		rec_filenames = glob(os.path.join(data_dir, 'train', 'train.rec'))
		idx_filenames = glob(os.path.join(data_dir, 'train', 'train.idx'))
	else:
		rec_filenames = glob(os.path.join(data_dir, 'val', 'val.rec'))
		idx_filenames = glob(os.path.join(data_dir, 'val', 'val.idx'))	
	if shard_id == 0: print(rec_filenames, idx_filenames)
	images, labels = fn.readers.mxnet(path=rec_filenames, index_path=idx_filenames,
										random_shuffle=is_training,
										num_shards=num_shards, shard_id=shard_id,
										read_ahead=False, prefetch_queue_depth=prefetch,
										name='Reader')

	dali_device = 'cpu' if dali_cpu else 'gpu'
	decoder_device = 'cpu' if dali_cpu else 'mixed'
	# ask nvJPEG to preallocate memory for the biggest sample in ImageNet for CPU and GPU to avoid reallocations in runtime
	device_memory_padding = 211025920 if decoder_device == 'mixed' else 0
	host_memory_padding = 140544512 if decoder_device == 'mixed' else 0
	# ask HW NVJPEG to allocate memory ahead for the biggest image in the data set to avoid reallocations in runtime
	preallocate_width_hint = 5980 if decoder_device == 'mixed' else 0
	preallocate_height_hint = 6430 if decoder_device == 'mixed' else 0
	
	images = fn.decoders.image(images, device=decoder_device, output_type=types.RGB,
												device_memory_padding=device_memory_padding,
											   	host_memory_padding=host_memory_padding,
											   	preallocate_width_hint=preallocate_width_hint,
											   	preallocate_height_hint=preallocate_height_hint,)

	if ssl_aug == 'mocov3':
		images_q = mocov3_aug1(images, size=size, crop=crop, dali_device=dali_device, precision=precision)
		images_k = mocov3_aug2(images, size=size, crop=crop, dali_device=dali_device, precision=precision)

		return images_q, images_k, labels.gpu()		

	elif ssl_aug == 'mocov2':		
		images_q = mocov2_aug(images, size=size, crop=crop, dali_device=dali_device, precision=precision)
		images_k = mocov2_aug(images, size=size, crop=crop, dali_device=dali_device, precision=precision)

		return images_q, images_k, labels.gpu()

	elif ssl_aug == "mocov2_sweet":
		images_q = mocov2_aug(images, size=size, crop=crop, dali_device=dali_device, precision=precision)
		images_k = mocov2_aug(images, size=size, crop=crop, dali_device=dali_device, precision=precision)

		images_k1 = sweet_aug(images, size=size, crop=crop, dali_device=dali_device, precision=precision)
		if len(width_mult_list) == 2:
			return images_q, images_k, images_k1, labels.gpu()

		images_k2 = sweet_aug(images, size=size, crop=crop, dali_device=dali_device, precision=precision)
		if len(width_mult_list) == 3:
			return images_q, images_k, images_k1, images_k2, labels.gpu()

		raise NotImplementedError

	elif ssl_aug == "mocov2_diff":
		images_q = mocov2_aug(images, size=size, crop=crop, dali_device=dali_device, precision=precision)
		images_k = mocov2_aug(images, size=size, crop=crop, dali_device=dali_device, precision=precision)

		images_k1 = mocov2_aug(images, size=size, crop=crop, dali_device=dali_device, precision=precision)
		if len(width_mult_list) == 2:
			return images_q, images_k, images_k1, labels.gpu()

		images_k2 = mocov2_aug(images, size=size, crop=crop, dali_device=dali_device, precision=precision)
		if len(width_mult_list) == 3:
			return images_q, images_k, images_k1, images_k2, labels.gpu()

		raise NotImplementedError

	elif ssl_aug == 'slim':
		images = slim_aug(images, size=size, crop=crop, dali_device=dali_device, precision=precision)

		return images, labels.gpu()

	elif ssl_aug == 'normal':
		images = normal_aug(images, size=size, crop=crop, dali_device=dali_device, precision=precision)

		return images, labels.gpu()

	else:
		images = fn.resize(images,
						   device=dali_device,
						   size=size,
						   mode="not_smaller",
						   interp_type=_interp)
		mirror = False
		images = fn.crop_mirror_normalize(images.gpu(),
										dtype=types.FLOAT if precision=='fp32' else types.FLOAT16,
										output_layout="CHW",
										crop=(crop, crop),
										mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
										std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
										mirror=mirror)

		labels = labels.gpu()

		return images, labels


##############################
# some DALI operations
##############################
def random_grayscale(images, probability, dali_device='cpu'):
	# https://docs.nvidia.com/deeplearning/dali/main-user-guide/docs/examples/image_processing/hsv_example.html?highlight=randomgrayscale
	saturate = fn.random.coin_flip(probability=1.0 - probability)
	saturate = fn.cast(saturate, dtype=types.FLOAT)
	return fn.hsv(images, saturation=saturate, device=dali_device)


def mux(condition, true_case, false_case):
	# https://docs.nvidia.com/deeplearning/dali/user-guide/docs/examples/general/expressions/expr_conditional_and_masking.html
	neg_condition = condition ^ True
	return condition * true_case + neg_condition * false_case


def random_colorjitter(images, probability=0.8, dali_device='cpu',
						brightness=[0.6, 1.4], contrast=[0.6, 1.4],
						saturation=[0.6, 1.4], hue=[-0.1, 0.1]):
	imgs_adjusted = fn.color_twist(images, device=dali_device,
									brightness=fn.random.uniform(range=brightness),
									contrast=fn.random.uniform(range=contrast),
									saturation=fn.random.uniform(range=saturation),
									hue=fn.random.uniform(range=hue))	
	condition = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, probability=probability)
	out = mux(condition, imgs_adjusted, images)

	return out


def random_gaussianblur(images, probability=0.5, dali_device='cpu', sigma=[0.1, 2.0]):
	imgs_adjusted = fn.gaussian_blur(images, device=dali_device,
										sigma=fn.random.uniform(range=sigma))
	condition = fn.random.coin_flip(dtype=types.DALIDataType.BOOL, probability=probability)
	out = mux(condition, imgs_adjusted, images)

	return out


##############################
# Augmentations Presets
##############################


def normal_aug(images, size=224, crop=224, dali_device='cpu', precision='fp32'):
	images = fn.random_resized_crop(images, device=dali_device,
									size=size,
									interp_type=_interp)

	# If no cropping arguments are specified, only mirroring and normalization will occur.
	images = fn.crop_mirror_normalize(images.gpu(),
									dtype=types.FLOAT if precision=='fp32' else types.FLOAT16,
									output_layout="CHW",
									mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
									std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
									mirror=fn.random.coin_flip(probability=0.5))

	return images


def slim_aug(images, size=224, crop=224, dali_device='cpu', precision='fp32'):
	"""augmentation for slimmable networks"""
	images = fn.random_resized_crop(images, device=dali_device,
									size=size,
									interp_type=_interp)

	images = fn.color_twist(images, device=dali_device,
								brightness=fn.random.uniform(range=[0.6, 1.4]),
								contrast=fn.random.uniform(range=[0.6, 1.4]),
								saturation=fn.random.uniform(range=[0.6, 1.4]))

	# If no cropping arguments are specified, only mirroring and normalization will occur.
	images = fn.crop_mirror_normalize(images.gpu(),
									dtype=types.FLOAT if precision=='fp32' else types.FLOAT16,
									output_layout="CHW",
									mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
									std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
									mirror=fn.random.coin_flip(probability=0.5))

	return images


def mocov2_aug(images, size=224, crop=224, dali_device='cpu', precision='fp32'):
	images = fn.random_resized_crop(images, device=dali_device,
									size=size, random_area=[0.2, 1.0],
									interp_type=_interp)

	images = random_colorjitter(images, probability=0.8, dali_device=dali_device,
									brightness=[0.6, 1.4], contrast=[0.6, 1.4],
									saturation=[0.6, 1.4], hue=[-0.1, 0.1])
	
	images = random_grayscale(images, probability=0.2, dali_device=dali_device)
	
	images = random_gaussianblur(images, probability=0.5, dali_device=dali_device,
									sigma=[0.1, 2.0])

	# If no cropping arguments are specified, only mirroring and normalization will occur.
	images = fn.crop_mirror_normalize(images.gpu(),
									dtype=types.FLOAT if precision=='fp32' else types.FLOAT16,
									output_layout="CHW",
									mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
									std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
									mirror=fn.random.coin_flip(probability=0.5))

	return images


def simclr_aug(images, size=224, crop=224, dali_device='cpu', precision='fp32'):
	images = fn.random_resized_crop(images, device=dali_device,
									size=size, random_area=[0.2, 1.0],
									interp_type=_interp)

	# SimCLR use a more strong colorjitter strategy
	# https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
	images = random_colorjitter(images, probability=0.8, dali_device=dali_device,
									brightness=[0.2, 1.8], contrast=[0.2, 1.8],
									saturation=[0.2, 1.8], hue=[-0.2, 0.2])

	images = random_grayscale(images, probability=0.2, dali_device=dali_device)

	images = random_gaussianblur(images, probability=0.5, dali_device=dali_device,
									sigma=[0.1, 2.0])

	# If no cropping arguments are specified, only mirroring and normalization will occur.
	images = fn.crop_mirror_normalize(images.gpu(),
									dtype=types.FLOAT if precision=='fp32' else types.FLOAT16,
									output_layout="CHW",
									mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
									std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
									mirror=fn.random.coin_flip(probability=0.5))

	return images


def mocov3_aug1(images, size=224, crop=224, dali_device='cpu', precision='fp32', crop_min=0.2):
	"""follow BYOL's augmentation recipe: https://arxiv.org/abs/2006.07733"""

	images = fn.random_resized_crop(images, device=dali_device,
									size=size, random_area=[crop_min, 1.0],
									interp_type=_interp)

	images = random_colorjitter(images, probability=0.8, dali_device=dali_device,
									brightness=[0.6, 1.4], contrast=[0.6, 1.4],
									saturation=[0.6, 1.4], hue=[-0.1, 0.1])
	
	images = random_grayscale(images, probability=0.2, dali_device=dali_device)
	
	# guassion blur with p=1.0
	# https://github.com/facebookresearch/moco-v3/blob/c349e6e24f40d3fedb22d973f92defa4cedf37a7/main_moco.py#L268
	# images = random_gaussianblur(images, probability=0.5, dali_device=dali_device,
	# 								sigma=[0.1, 2.0])
	images = fn.gaussian_blur(images, device=dali_device,
									sigma=fn.random.uniform(range=[0.1, 2.0]))

	# If no cropping arguments are specified, only mirroring and normalization will occur.
	images = fn.crop_mirror_normalize(images.gpu(),
										dtype=types.FLOAT if precision=='fp32' else types.FLOAT16,
										output_layout="CHW",
										mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
										std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
										mirror=fn.random.coin_flip(probability=0.5))

	return images


def mocov3_aug2(images, size=224, crop=224, dali_device='cpu', precision='fp32', crop_min=0.2):
	images = fn.random_resized_crop(images, device=dali_device,
									size=size, random_area=[crop_min, 1.0],
									interp_type=_interp)

	images = random_colorjitter(images, probability=0.8, dali_device=dali_device,
									brightness=[0.6, 1.4], contrast=[0.6, 1.4],
									saturation=[0.6, 1.4], hue=[-0.1, 0.1])

	images = random_grayscale(images, probability=0.2, dali_device=dali_device)

	images = random_gaussianblur(images, probability=0.1, dali_device=dali_device,
									sigma=[0.1, 2.0])

	# If no cropping arguments are specified, only mirroring and normalization will occur.
	# no solarize ops in nvidia DALI, here we only do normalize, random_solarize + normalize will be done by torch
	images = fn.crop_mirror_normalize(images.gpu(),
										dtype=types.FLOAT if precision=='fp32' else types.FLOAT16,
										output_layout="CHW",
										mirror=fn.random.coin_flip(probability=0.5))

	return images


def sweet_aug(images, size=224, crop=224, dali_device='cpu', precision='fp32'):
	"""an weak augmentations for small networks"""
	images = fn.random_resized_crop(images, device=dali_device,
									size=size, random_area=[0.2, 1.0],
									interp_type=_interp)

	images = random_colorjitter(images, probability=0.8, dali_device=dali_device,
									brightness=[0.6, 1.4], contrast=[0.6, 1.4],
									saturation=[0.6, 1.4], hue=[-0.1, 0.1])

	# images = random_grayscale(images, probability=0.2, dali_device=dali_device)

	# If no cropping arguments are specified, only mirroring and normalization will occur.
	images = fn.crop_mirror_normalize(images.gpu(),
									dtype=types.FLOAT if precision=='fp32' else types.FLOAT16,
									output_layout="CHW",
									mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
									std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
									mirror=fn.random.coin_flip(probability=0.5))

	return images
