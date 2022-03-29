# !/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
sys.path.append('../../')

import os
import time
import math
import torch
import logging
import traceback
import torch.nn as nn
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils.audio import Audio
from utils.si_sdr import single_si_sdr
from mir_eval.separation import bss_eval_sources
from losses import Loss
from sklearn.cluster import KMeans

class DPCL_Trainer(object):
    def __init__(self, train_dl, test_dl, model, optimizer, writer, options):
        self.name = options['name']
        # audio
        self.audio = Audio(**options['datasets']['audio_setting'])
        self.clip_norm = options['optimizer']['clip_norm']
        self.writer = writer
        # dataloder
        self.train_dl = train_dl
        self.test_dl = test_dl
        
        # logger & tensorboard
        self.logger = logging.getLogger(options['logger']['name'])
        self.print_freq = options['logger']['print_freq']
        self.summary_interval = options['train']['summary_interval']
        self.eval_interval = options['train']['eval_interval']

        # checkpoint
        self.ckp_path = options['train']['ckp_path']
        self.checkpoint_interval = options['train']['checkpoint_interval']

        self.current_step = 0

        if options['train']['is_gpu']:
            self.device = torch.device('cuda:0')
            self.model = model.to(self.device)
        else:
            self.device = torch.device('cpu')
            self.model = model.to(self.device)

        if options['resume']['state']:
            checkpoint = torch.load(options['resume']['ckp_path'], map_location='cpu')
            self.current_step = checkpoint['step']
            self.logger.info('Resume from checkpoint {}: step {:d}'.format(options['resume']['ckp_path'], self.current_step))
            self.model = model.load_state_dict(checkpoint['model']).to(self.device)
            self.optimizer = optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            self.model = model.to(self.device)
            self.optimizer = optimizer

    def run_with_feature(self):
        try:
            while True:
                self.model.train()
                for feature, mix_mag, target_mask, non_silence in self.train_dl:
                    mix_mag = mix_mag.to(self.device)
                    target_mask = target_mask.to(self.device)
                    non_silence = non_silence.to(self.device)
                    feature = feature.to(self.device)

                    try:
                        embedding = self.model(mix_mag, feature)
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print('|WARNING: ran out of memory')
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            raise e

                    l = Loss(mix_mag=embedding, target_mask=target_mask, non_silent=non_silence)
                    loss = l.loss()
                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                    self.optimizer.step()
                    self.current_step += 1
                    
                    
                    if loss.item() > 1e8 or math.isnan(loss.item()):
                        self.logger.error("Loss exploded to %.02f at step %d!" % (loss.item(), self.current_step))
                        raise Exception("Loss exploded")
                    
                    # write loss to tensorboard
                    if self.current_step % self.summary_interval == 0:
                        self.writer.log_training_v2(self.name, loss, self.current_step)
                        self.logger.info('step:{0}, loss:{1}'.format(self.current_step, loss))

                    if self.current_step % self.eval_interval == 0:
                        self.eval_with_feature(self.current_step)

                    # save ckp file to resume training
                    if self.current_step % self.checkpoint_interval == 0:
                        save_path = os.path.join(self.ckp_path, self.name, 'ckp_%d.pt' % self.current_step)
                        torch.save({
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'step': self.current_step
                        }, save_path)
                        self.logger.info('save checkpoint to {}'.format(save_path))
        except Exception as e:
            self.logger.info('Exiting due to the exception: %s' % e)
            traceback.print_exc()

    
    def run(self):
        try:
            while True:
                self.model.train()
                for mix_mag, target_mask, non_silence in self.train_dl:
                    mix_mag = mix_mag.to(self.device)
                    target_mask = target_mask.to(self.device)
                    non_silence = non_silence.to(self.device)

                    try:
                        embedding = self.model(mix_mag)
                    except RuntimeError as e:
                        if 'out of memory' in str(e):
                            print('|WARNING: ran out of memory')
                            if hasattr(torch.cuda, 'empty_cache'):
                                torch.cuda.empty_cache()
                        else:
                            raise e

                    l = Loss(mix_mag=embedding, target_mask=target_mask, non_silent=non_silence)
                    loss = l.loss()
                    self.optimizer.zero_grad()
                    loss.backward()

                    if self.clip_norm:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
                    self.optimizer.step()
                    self.current_step += 1
                    
                    
                    if loss.item() > 1e8 or math.isnan(loss.item()):
                        self.logger.error("Loss exploded to %.02f at step %d!" % (loss.item(), self.current_step))
                        raise Exception("Loss exploded")
                    
                    # write loss to tensorboard
                    if self.current_step % self.summary_interval == 0:
                        self.writer.log_training_v2(self.name, loss, self.current_step)
                        self.logger.info('step:{0}, loss:{1}'.format(self.current_step, loss))

                    if self.current_step % self.eval_interval == 0:
                        self.eval(self.current_step)

                    # save ckp file to resume training
                    if self.current_step % self.checkpoint_interval == 0:
                        save_path = os.path.join(self.ckp_path, self.name, 'ckp_%d.pt' % self.current_step)
                        torch.save({
                            'model': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'step': self.current_step
                        }, save_path)
                        self.logger.info('save checkpoint to {}'.format(save_path))
        except Exception as e:
            self.logger.info('Exiting due to the exception: %s' % e)
            traceback.print_exc()


