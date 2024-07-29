# coding=utf-8
import os
import io
# import cv2
import json
import lmdb
import torch
import logging
import numpy as np
from PIL import Image


class ImageFolderLMDB(torch.utils.data.Dataset):
	"""A ImageNet data loader where the lmdb are arranged in this way: ::
		root/train_lmdb/train.lmdb
		root/val_lmdb/val.lmdb
		root/train_list.json
		root/val_list.json
	"""    
	def __init__(self, data_dir,
					lmdb_dataset=None,
					key_label_json=None,
					transform=None,
					target_transform=None,
					albumentations=False):
		logging.info('[dataset] use {} as data source'.format(lmdb_dataset))
		self.data_dir = data_dir
		self.lmdb_dataset = lmdb_dataset
		self.albumentations = albumentations
		with open(key_label_json, 'r') as f:
			self.labels_dict = json.load(f)
		self.keys = list(self.labels_dict.keys())

		self.transform = transform
		self.target_transform = target_transform
		self.length = len(self.keys)

	def __getitem__(self, index):
		key = self.keys[index]
		data = self.db_txn.get(key.encode())
		if not self.albumentations:
			img = Image.open(io.BytesIO(data)).convert('RGB')
		else:
			raise NotImplementedError
			# img = np.frombuffer(data, dtype=np.uint8)  # convert it to numpy
			# img = cv2.imdecode(img, cv2.IMREAD_COLOR)
			# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			# jpeg_reader.decode(data, pixel_format=TJPF_RGB)
		target = self.labels_dict[key]

		if self.transform is not None:
			img = self.transform(img) if not self.albumentations else self.transform(image=img)["image"]

		if self.target_transform is not None:
			target = self.target_transform(target)

		return img, target

	def __len__(self):
		return self.length

	def __repr__(self):
		return self.__class__.__name__ + ' (' + self.db_path + ')'

	def __getstate__(self):
		state = self.__dict__
		state["db_txn"] = None
		return state

	def __setstate__(self, state):
		# https://github.com/pytorch/vision/issues/689
		self.__dict__ = state
		if self.lmdb_dataset not in [None, 'None']:
			env = lmdb.open(self.lmdb_dataset, subdir=os.path.isdir(self.lmdb_dataset),
									readonly=True, lock=False,
									readahead=False, meminit=False,
									map_size=1<<43,)
			self.db_txn = env.begin(write=False)
