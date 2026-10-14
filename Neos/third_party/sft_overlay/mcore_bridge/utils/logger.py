# Copyright (c) ModelScope Contributors. All rights reserved.
# code borrowed from modelscope/ms-swift
import logging
import os
from types import MethodType
from typing import Optional



def _is_local_master():
    local_rank = int(os.getenv('LOCAL_RANK', -1))
    return local_rank in {-1, 0}


init_loggers = {}



logger_format = logging.Formatter('[%(levelname)s:%(name)s] %(message)s')

info_set = set()
warning_set = set()


def info_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in info_set:
        return
    info_set.add(hash_id)
    self.info(msg)


def warning_once(self, msg, *args, **kwargs):
    hash_id = kwargs.get('hash_id') or msg
    if hash_id in warning_set:
        return
    warning_set.add(hash_id)
    self.warning(msg)


def get_logger(log_level: Optional[int] = None):
    if log_level is None:
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        log_level = getattr(logging, log_level, logging.INFO)
    logger_name = __name__.split('.')[0]
    logger = logging.getLogger(logger_name)
    logger.propagate = False
    if logger_name in init_loggers:
        return logger

    
    
    
    
    
    
    
    for handler in logger.root.handlers:
        if type(handler) is logging.StreamHandler:
            handler.setLevel(logging.ERROR)

    stream_handler = logging.StreamHandler()
    handlers = [stream_handler]

    is_worker0 = _is_local_master()
    for handler in handlers:
        handler.setFormatter(logger_format)
        handler.setLevel(log_level)
        logger.addHandler(handler)

    if is_worker0:
        logger.setLevel(log_level)
    else:
        logger.setLevel(logging.ERROR)

    init_loggers[logger_name] = True

    logger.info_once = MethodType(info_once, logger)
    logger.warning_once = MethodType(warning_once, logger)
    return logger


logger = get_logger()
logger.handlers[0].setFormatter(logger_format)


def _add_file_handler_if_needed(logger, log_file, file_mode, log_level):
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler):
            return

    is_worker0 = _is_local_master()
    if is_worker0 and log_file is not None:
        file_handler = logging.FileHandler(log_file, file_mode)
        file_handler.setFormatter(logger_format)
        file_handler.setLevel(log_level)
        logger.addHandler(file_handler)
