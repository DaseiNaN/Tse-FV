# !/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from tensorboardX import SummaryWriter

from .plotting import plot_spectrogram_to_numpy

class MyTensorBoardWriter(SummaryWriter):
    def __init__(self, log_dir):
        super(MyTensorBoardWriter, self).__init__(log_dir)

    def log_training(self, model_name, train_loss, val_loss, epoch):
        self.add_scalar(model_name + '_train_loss', train_loss, epoch)
        self.add_scalar(model_name + '_cross_validation_loss', val_loss, epoch)

    def log_training_v2(self, model_name, train_loss, step):
        self.add_scalar(model_name + '_train_loss', train_loss, step)

    def log_evaluation_v2(self, model_name, test_loss, sdr, step):
        self.add_scalar(model_name + '_test_loss', test_loss, step)
        self.add_scalar(model_name + '_check_sdr', sdr, step)
    
    def log_audio_info(self, model_name, mix_wav, target_wav, estimate_wav, mix_spec, target_spec, estimate_spec, estimate_mask, step, sample_rate=16000):
        self.add_audio('mix_wav', mix_wav, step, sample_rate)
        self.add_audio('target_wav', target_wav, step, sample_rate)
        self.add_audio('estimate_wav', estimate_wav, step, sample_rate)

        self.add_image('input/mix_spectrogram',
            plot_spectrogram_to_numpy(mix_spec), step, dataformats='HWC')
        self.add_image('input/target_spectrogram',
            plot_spectrogram_to_numpy(target_spec), step, dataformats='HWC')
        self.add_image('result/estimate_spectrogram',
            plot_spectrogram_to_numpy(estimate_spec), step, dataformats='HWC')
        self.add_image('result/estimate_mask',
            plot_spectrogram_to_numpy(estimate_mask), step, dataformats='HWC')
        self.add_image('result/estimation_error',
            plot_spectrogram_to_numpy(np.square(estimate_spec - target_spec)), step, dataformats='HWC')

    def log_evaluation(self, model_name, test_loss, si_sdr, mix_wav, target_wav, estimate_wav, mix_spec, target_spec, estimate_mask, estimate_spec, epoch, sample_rate):
        self.add_scalar(model_name + '_test_loss', test_loss, epoch)
        self.add_scalar(model_name + '_si_sdr/sdr', si_sdr, epoch)

        self.add_audio('mix_wav', mix_wav, epoch, sample_rate)
        self.add_audio('target_wav', target_wav, epoch, sample_rate)
        self.add_audio('estimate_wav', estimate_wav, epoch, sample_rate)

        self.add_image('input/mix_spectrogram',
            plot_spectrogram_to_numpy(mix_spec), epoch, dataformats='HWC')
        self.add_image('input/target_spectrogram',
            plot_spectrogram_to_numpy(target_spec), epoch, dataformats='HWC')
        self.add_image('result/estimate_spectrogram',
            plot_spectrogram_to_numpy(estimate_spec), epoch, dataformats='HWC')
        self.add_image('result/estimate_mask',
            plot_spectrogram_to_numpy(estimate_mask), epoch, dataformats='HWC')
        self.add_image('result/estimation_error',
            plot_spectrogram_to_numpy(np.square(estimate_spec - target_spec)), epoch, dataformats='HWC')

    def test_log_audio(self, test_wav, sample_rate):
        self.add_audio('test_audio', test_wav, 1, 16000)

    def log_infer_info(self, sdr, mix_wav, target_wav, estimate_wav, mix_spec, target_spec, estimate_mask, estimate_spec, sample_rate, order):
        self.add_scalar('sdr', sdr, order)
        
        self.add_audio('mix_wav', mix_wav, order, sample_rate)
        self.add_audio('target_wav', target_wav, order, sample_rate)
        self.add_audio('estimate_wav', estimate_wav, order, sample_rate)

        self.add_image('input/mix_spectrogram',
            plot_spectrogram_to_numpy(mix_spec), order, dataformats='HWC')
        self.add_image('input/target_spectrogram',
            plot_spectrogram_to_numpy(target_spec), order, dataformats='HWC')
        self.add_image('result/estimate_spectrogram',
            plot_spectrogram_to_numpy(estimate_spec), order, dataformats='HWC')
        self.add_image('result/estimate_mask',
            plot_spectrogram_to_numpy(estimate_mask), order, dataformats='HWC')
        self.add_image('result/estimation_error',
            plot_spectrogram_to_numpy(np.square(estimate_spec - target_spec)), order, dataformats='HWC')