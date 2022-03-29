# !/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import yaml


def parse(config_template, project_root, data_root_dir, name, config_dir):
    """ 配置文件模版解析

    根据输入参数对默认的训练模版 xxx_.yaml 的内容进行替换, 并生成新的训练参数文件

    Args:
        config_template (String): 训练模版文件路径
        project_root (String): 项目工程根目录的绝对路径
        data_root_dir (String): 数据根目录的绝对路径
        name (String): 生成配置文件的名称

    Returns:
        配置信息
    """

    template_f = open(config_template, mode='r')
    # config_path = os.path.join(config_dir, config_template.split('/')[-1])
    config_path = os.path.join(config_dir, name+'.yaml')
    config_f = open(config_path, mode='w')

    config_f.write('name: ' + name + '\n')
    for line in template_f.readlines():
        line = line.replace('${project_path}', project_root)
        line = line.replace('${data_root_dir}', data_root_dir)
        line = line.replace('${name}', name)
        config_f.write(line)

    template_f.close()
    config_f.close()

    with open(config_path, mode='r') as f:
        _config = yaml.load(f, Loader=yaml.FullLoader)
    return _config
