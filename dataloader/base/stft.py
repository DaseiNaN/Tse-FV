# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@File        : stft.py
@Version     : 1.0
"""

import librosa
import torch
import torchaudio
import numpy as np


class STFT(object):
    """ 短时傅里叶变换

    对 librosa 内置的 STFT 进行包装, 返回 time x frequency 的 STFT 变换后的信号

    Attributes:
        window (String, optional): 窗函数, Default='hann'
        nfft (int, optional): 用零填充后加窗信号的长度, Default=256
        window_length (int, optional): 窗口长度, Default=256
        hop_length (int, optional): 帧之间的overlap, Default=64
        center (bool, optional): 以窗的中心时刻作为时间标示，还是以窗的开始位置作为时间标示, Default=False
    """

    def __init__(self, window='hann', nfft=256, window_length=256, hop_length=64, center=False):
        self.window = window
        self.nfft = nfft
        self.window_length = window_length
        self.hop_length = hop_length
        self.center = center

    def stft(self, waveform, is_mag=False, is_log=False):
        """ 短时傅里叶变换

        Args:
            waveform (array_like): 音频序列
            is_mag (bool, optional): 是否计算幅度值, 默认为 False
            is_log (bool, optional): 是否计算对数幅值, 默认为 False

        Returns:
            stft_waveform: stft变换后的音频信号 time x frequency

        eg:
            >>> instance_stft = STFT()
            >>> stft_waveform = instance_stft.stft(waveform, is_mag=True, is_log=True)
        """
        stft_waveform = librosa.stft(waveform, n_fft=self.nfft, hop_length=self.hop_length,
                                     win_length=self.window_length, window=self.window, center=self.center)
        # frequency x time -> time x frequency
        stft_waveform = np.transpose(stft_waveform)

        if is_mag:
            stft_waveform = np.abs(stft_waveform)
        if is_log:
            min_z = np.finfo(float).eps
            stft_waveform = np.log(np.maximum(stft_waveform, min_z))
        return stft_waveform

    def istft(self, stft_waveform):
        """ 逆短时傅里叶变换

        Args:
            stft_waveform: [array_like] 输入信号的 stft

        Returns:
            istft_waveform: istft变换后的音频信号

        eg:
            >>> instance_stft = STFT()
            >>> istft_waveform = instance_stft.istft(stft_waveform)
        """
        # time x frequency -> frequency x time
        stft_waveform = np.transpose(stft_waveform)

        istft_waveform = librosa.istft(stft_waveform, hop_length=self.hop_length,
                                       win_length=self.window_length, window=self.window, center=self.center)
        return istft_waveform