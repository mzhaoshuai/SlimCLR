# coding=utf-8
import torch
import torch.nn.functional as F


def matmul_with_split(A, B, split_size=1000, cuda=False):
	"""
	matmul with split to avoid OOM
	Args:
	A: torch.Tensor, [N, M]
	B: torch.Tensor, [M, K]
	"""
	if A.shape[0] > split_size:
		if cuda: B = B.cuda()
		all_A = torch.split(A, split_size, dim=0)
		res_l = []
		for a in all_A:
			res_l.append(torch.matmul(a.cuda() if cuda else a, B).cpu())
		
		return torch.cat(res_l, dim=0)
	else:
		return torch.matmul(A, B)


@torch.no_grad()
def solve_muticlass_linear_regression(X, y, regularizer=0.01, split_size=4, cuda=False):
	"""
	solve a multiclass linear regression problem
	https://www.cs.toronto.edu/~urtasun/courses/CSC411_Fall16/07_multiclass.pdf
	W = (X^T X)^(-1) X^T Y
	Args:
		X: [n_samples, n_features + 1]
		y: [n_samples, ]
	"""
	N, M = X.shape[0], X.shape[1]

	XTX = matmul_with_split(X.T, X, split_size=split_size, cuda=cuda)
	reg = torch.eye(M, dtype=X.dtype, device=X.device) * regularizer
	# a symmetric positive-definite matrix
	XTX.add_(reg)

	# [n_features + 1, n_features + 1]
	XTX_inv = torch.cholesky_inverse(XTX)

	# [n_features + 1, n_samples] x [n_samples, num_classes]
	# = [n_features + 1, num_classes]
	XTY = matmul_with_split(X.T, F.one_hot(y).float(), split_size=split_size, cuda=cuda)

	# weight, [n_features + 1, num_classes]
	w = torch.matmul(XTX_inv, XTY)

	return w


@torch.no_grad()
def lr_predict(X, w):
	"""
	inference of a linear regression	
	Args:
		X: [n_samples, n_features]
		w: [n_features + 1, num_classes]
	Return:
		prediction [n_samples, num_classes]
	"""
	N, M = X.shape[0], X.shape[1]
	# aug, [n_samples, n_features + 1]
	one = torch.ones((N, 1), dtype=X.dtype, device=X.device)
	X = torch.cat([one, X], dim=1)	

	return torch.matmul(X, w)


if __name__ == "__main__":
	import time
	train_num = 1300000
	val_num = 50000
	dim = 2048

	train_X = torch.ones(train_num, dim + 1)
	train_X[:, 1:] = torch.rand(train_num, dim)
	# train_X = train_X

	y = torch.randint(low=0, high=1000, size=(train_num,))
	
	if torch.cuda.is_available(): torch.cuda.synchronize()
	start = time.time()
	print('start to solve the linear regression problem...')
	w = solve_muticlass_linear_regression(train_X, y, regularizer=1e-1, split_size=4, cuda=True)
	print(w[:20, :20])
	if torch.cuda.is_available(): torch.cuda.synchronize()
	all_time = time.time() - start

	print('Solve the linear regression cost {} minutes'.format(all_time / 60))
	print('The maximum GPU memory occupied by this program is {:.2f} GB\n'.format(
						torch.cuda.max_memory_allocated(0) * 1.0 / 1024 / 1024 / 1024))

	# cpu cost ~30 min
	# cuda cost ~1 min, 12GB GPU MEM
	# with MoCov2, 800ep, the top1 acc of multiclass linear regression is ~50%