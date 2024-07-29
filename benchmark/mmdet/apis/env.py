import logging
import os
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from mmcv.runner import get_dist_info


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break
    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break
    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_dist(launcher, backend='nccl', **kwargs):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn')
    if launcher == 'pytorch':
        return _init_dist_pytorch(backend, **kwargs)

    else:
        raise NotImplementedError


def _init_dist_pytorch(backend, **kwargs):
    # TODO: use local_rank instead of rank % num_gpus
    # DDP via torchrun, torch.distributed.launch
    local_rank, _, _ = world_info_from_env()
    # rank = int(os.environ['RANK'])
    # num_gpus = torch.cuda.device_count()
    # torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    
    if torch.cuda.is_available():
        if torch.distributed.is_initialized():
            device = 'cuda:%d' % local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'
    device = torch.device(device)

    return device


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_root_logger(log_level=logging.INFO):
    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            format='%(asctime)s - %(levelname)s - %(message)s',
            level=log_level)
    rank, _ = get_dist_info()
    if rank != 0:
        logger.setLevel('ERROR')
    return logger
