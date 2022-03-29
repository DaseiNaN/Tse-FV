# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author      : Dasein
@Contact     : dasein_csd@163.com
@Github      : https://github.com/YagoToasa
-------------*-
@File        : pit_train.py
@Version     : 1.0
@Software    : VS Code
@Description : PIT without Voice Print Train
"""

name = 'pit_train_conv_mfcc_v1'
project_path = '/home/chengshidan/Documents/GraduationProject/serendipity'
data_info_dir = '/home/chengshidan/Documents/Datasets/LIBRI_SPEECH/MIX_V2'
config_dir = project_path + '/examples/pit'
import sys
sys.path.append(project_path)

import os
import torch
import argparse
import logging
from logger import logger as setup_logger
from model import PITNet
from trainer import PIT_Trainer
from dataloader.pit_loader import create_dataloader
from utils.parser import parse
from utils.writer import MyTensorBoardWriter


def make_optimizer(params, options):
    optimizer = getattr(torch.optim, options['optimizer']['name'])
    if options['optimizer']['name'] == 'Adam':
        optimizer = optimizer(params, lr=options['optimizer']['lr'], weight_decay=options['optimizer']['weight_decay'])
    else:
        optimizer = optimizer(params, lr=options['optimizer']['lr'], weight_decay=options['optimizer']['weight_decay'], momentum=options['optimizer']['momentum'])
    return optimizer

def make_dataloader(options):
    train_dl = create_dataloader(data_root_dir=options['datasets']['path'], 
                                data_type='train',
                                batch_size=options['datasets']['dataloader_setting']['batch_size'], 
                                num_workers=options['datasets']['dataloader_setting']['num_workers'],
                                feature_type=options['feature_type'],
                                shuffle=True)
    test_dl = create_dataloader(data_root_dir=options['datasets']['path'], 
                                data_type='test',
                                batch_size=1,
                                num_workers=0,
                                feature_type=options['feature_type'],
                                shuffle=False)
    return train_dl, test_dl


def train():
    # 解析命令行参数
    arg_parser = argparse.ArgumentParser(description='Parameters for PIT train')
    arg_parser.add_argument('--config', type=str, help='Path to config template YAML file.')
    args = arg_parser.parse_args()

    # 获取 yaml 配置信息
    options = parse(args.config, project_path, data_info_dir, name, config_dir)
    if options['feature_type'] == 'None':
        options['feature_type'] = None

    # logger 日志配置
    setup_logger.setup_logging(options['logger']['path'], options['logger']['config'])
    logger = logging.getLogger(options['logger']['name'])
    
    # tensorboard writer
    writer = MyTensorBoardWriter(options['board']['path'])

    logger.info('Building the model of PIT')
    model = PITNet(**options['model'])

    logger.info('Building the optimizer of PIT')
    optimizer = make_optimizer(model.parameters(), options)

    logger.info('Building the dataloader of PIT')
    train_dl, test_dl = make_dataloader(options)


    logger.info('Train Datasets Length: {}, Test Datasets Length: {}'.format(len(train_dl), len(test_dl)))
    logger.info('Builing the Trainer of PIT')
    trainer = PIT_Trainer(train_dl, test_dl, model, optimizer, writer, options)

    if options['feature_type'] is not None:
        trainer.run_with_feature()
    else:
        trainer.run()

if __name__ == '__main__':
    train()
