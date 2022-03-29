# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml
import logging
import logging.config
from pathlib import Path


def setup_logging(save_dir, config_file='config.yaml', logging_level=logging.INFO):
    """日志配置

    设置日志输出的属性, 日志存储目录需指定, 默认日志配置文件 config.yaml 的路径在同级目录之下,
    默认日志输出级别为 logging.INFO

    Args:
        save_dir      : [String] 日志文件目录
        config_file   : [String, optional] 日志配置文件路径
        logging_level : [Integer, optional] 日志输出级别, 默认为 logging.INFO

    eg:
        >>> setup_logging(save_dir='../../log/')
    """

    # 新建日志输出目录
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 读取日志配置文件, 配置 logger 属性
    log_config = Path(config_file)

    if log_config.is_file():
        with open(config_file, 'r') as f:
            config = yaml.safe_load(f.read())
        for _, handler in config['handlers'].items():
            if 'filename' in handler:
                handler['filename'] = str(save_dir + '/' + handler['filename'])
        logging.config.dictConfig(config)
    else:
        print("Warning: logging configuration file is not found in {}.".format(log_config))
        logging.basicConfig(level=logging_level)
