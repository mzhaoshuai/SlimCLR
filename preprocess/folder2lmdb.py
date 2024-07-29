# coding=utf-8
"""
Reference:
	[1] https://github.com/Lyken17/Efficient-PyTorch/blob/master/tools/folder2lmdb.py
"""
import os
import sys
import json
import lmdb
import errno
import warnings
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import ImageFolder


def folder2lmdb(data_dir, name="train", write_frequency=5000):
	"""
	create ImageNet LMDB dataset,
	this functtion will generate two files: LMDB database and a json file which contains the key and labels
	"""
	dataset = ImageFolder(os.path.join(data_dir, name))
	smaple_num = len(dataset)
	key_label = dict()
	print("Loading dataset from {}".format(os.path.join(data_dir, name)))
	print('The length of the dataset is {}'.format(smaple_num))

	# where lmdb locate
	if not os.path.exists(os.path.join(data_dir, "{}_lmdb".format(name))):
		os.makedirs(os.path.join(data_dir, "{}_lmdb".format(name)), exist_ok=True)	
	lmdb_path = os.path.join(data_dir, "{}_lmdb".format(name), "{}.lmdb".format(name))    
	isdir = os.path.isdir(lmdb_path)

	print("Generate LMDB to %s" % lmdb_path)
	db = lmdb.open(lmdb_path, subdir=isdir,
				   map_size=1 << 43,
				   readonly=False,
				   meminit=False,
				   map_async=True)

	txn = db.begin(write=True)
	for i in tqdm(range(smaple_num)):
		path, target = dataset.samples[i]
		with open(path, mode='rb') as file:
			image_data = file.read()
		basename_w_folder = path.split('/')[-2] + '/' + path.split('/')[-1]
		key_label[basename_w_folder] = int(target)
		key = basename_w_folder.encode()
		flag = txn.put(key, image_data)
		if not flag:
			raise IOError("LMDB write error!")

		# write after certain samples
		if i % write_frequency == 0:
			# print("[{}/{}]".format(i, smaple_num))
			txn.commit()
			txn = db.begin(write=True)

	# finish iterating through dataset
	txn.commit()
	print("Flushing database ...")
	db.sync()
	db.close()

	# save key-label dict to json
	print('Save key and label to {}'.format(os.path.join(data_dir, '{}_list.json'.format(name))))
	with open(os.path.join(data_dir, '{}_list.json'.format(name)), 'w') as f:
		json.dump(key_label, f, indent=4, sort_keys=True)


def pil_check_images(folder_path):
	"""
	Usage: pil_check_images('/path/to/imagenet/train')
	"""
	warnings.filterwarnings("error")
	list_images = []
	for root, dub_dir, image_files in os.walk(folder_path):
		for image_file in image_files:
			path_tmp = os.path.join(root, image_file)
			if path_tmp.endswith('.JPEG'):
				list_images.append(os.path.join(root, image_file))	

	for image_path in list_images: 
		try:
			img = Image.open(image_path)
			exif_data = img._getexif()
		except UserWarning as err:
			print(err)
			print("Error on image: ", image_path)
		img.close()


if __name__ == "__main__":
	folder2lmdb('/path/to/imagenet', name='train')
	folder2lmdb('/path/to/imagenet', name='val')