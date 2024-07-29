# Copyright (c) OpenMMLab. All rights reserved.
import logging
import warnings
import torch
from torch import nn
from packaging.version import parse
import torch.distributed as dist


# from .registry import MODULE_WRAPPERS

TORCH_VERSION = torch.__version__


# def is_module_wrapper(module: nn.Module) -> bool:
#     """Check if a module is a module wrapper.

#     The following 3 modules in MMCV (and their subclasses) are regarded as
#     module wrappers: DataParallel, DistributedDataParallel,
#     MMDistributedDataParallel (the deprecated version). You may add you own
#     module wrapper by registering it to mmcv.parallel.MODULE_WRAPPERS or
#     its children registries.

#     Args:
#         module (nn.Module): The module to be checked.

#     Returns:
#         bool: True if the input module is a module wrapper.
#     """

#     def is_module_in_wrapper(module, module_wrapper):
#         module_wrappers = tuple(module_wrapper.module_dict.values())
#         if isinstance(module, module_wrappers):
#             return True
#         for child in module_wrapper.children.values():
#             if is_module_in_wrapper(module, child):
#                 return True
#         return False

#     return is_module_in_wrapper(module, MODULE_WRAPPERS)


def digit_version(version_str: str, length: int = 4):
    """Convert a version string into a tuple of integers.
    This method is usually used for comparing two versions. For pre-release
    versions: alpha < beta < rc.
    Args:
        version_str (str): The version string.
        length (int): The maximum number of version levels. Default: 4.
    Returns:
        tuple[int]: The version info in digits (integers).
    """
    assert 'parrots' not in version_str
    version = parse(version_str)
    assert version.release, f'failed to parse version {version_str}'
    release = list(version.release)
    release = release[:length]
    if len(release) < length:
        release = release + [0] * (length - len(release))
    if version.is_prerelease:
        mapping = {'a': -3, 'b': -2, 'rc': -1}
        val = -4
        # version.pre can be None
        if version.pre:
            if version.pre[0] not in mapping:
                warnings.warn(f'unknown prerelease version {version.pre[0]}, '
                              'version checking may go wrong')
            else:
                val = mapping[version.pre[0]]
            release.extend([val, version.pre[-1]])
        else:
            release.extend([val, 0])

    elif version.is_postrelease:
        release.extend([1, version.post])  # type: ignore
    else:
        release.extend([0, 0])
    return tuple(release)


logger_initialized: dict = {}


def get_logger(name, log_file=None, log_level=logging.INFO, file_mode='w'):
    """Initialize and get a logger by name.
    If the logger has not been initialized, this method will initialize the
    logger by adding one or two handlers, otherwise the initialized logger will
    be directly returned. During initialization, a StreamHandler will always be
    added. If `log_file` is specified and the process rank is 0, a FileHandler
    will also be added.
    Args:
        name (str): Logger name.
        log_file (str | None): The log filename. If specified, a FileHandler
            will be added to the logger.
        log_level (int): The logger level. Note that only the process of
            rank 0 is affected, and other processes will set the level to
            "Error" thus be silent most of the time.
        file_mode (str): The file mode used in opening log file.
            Defaults to 'w'.
    Returns:
        logging.Logger: The expected logger.
    """
    logger = logging.getLogger(name)
    if name in logger_initialized:
        return logger
    # handle hierarchical names
    # e.g., logger "a" is initialized, then logger "a.b" will skip the
    # initialization since it is a child of "a".
    for logger_name in logger_initialized:
        if name.startswith(logger_name):
            return logger

    # handle duplicate logs to the console
    # Starting in 1.8.0, PyTorch DDP attaches a StreamHandler <stderr> (NOTSET)
    # to the root logger. As logger.propagate is True by default, this root
    # level handler causes logging messages from rank>0 processes to
    # unexpectedly show up on the console, creating much unwanted clutter.
    # To fix this issue, we set the root logger's StreamHandler, if any, to log
    # at the ERROR level.
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
    else:
        rank = 0

    # only rank 0 will add a FileHandler
    if rank == 0 and log_file is not None:
        # Here, the default behaviour of the official logger is 'a'. Thus, we
        # provide an interface to change the file mode to the default
        # behaviour.
        file_handler = logging.FileHandler(log_file, file_mode)
        handlers.append(file_handler)

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    for handler in handlers:
        handler.setFormatter(formatter)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if rank == 0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    logger_initialized[name] = True

    return logger


def print_log(msg, logger=None, level=logging.INFO):
    """Print a log message.
    Args:
        msg (str): The message to be logged.
        logger (logging.Logger | str | None): The logger to be used.
            Some special loggers are:
            - "silent": no message will be printed.
            - other str: the logger obtained with `get_root_logger(logger)`.
            - None: The `print()` method will be used to print log messages.
        level (int): Logging level. Only available when `logger` is a Logger
            object or "root".
    """
    if logger is None:
        print(msg)
    elif isinstance(logger, logging.Logger):
        logger.log(level, msg)
    elif logger == 'silent':
        pass
    elif isinstance(logger, str):
        _logger = get_logger(logger)
        _logger.log(level, msg)
    else:
        raise TypeError(
            'logger should be either a logging.Logger object, str, '
            f'"silent" or None, but got {type(logger)}')
