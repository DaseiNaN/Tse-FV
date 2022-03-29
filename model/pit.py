# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Reference   : https://github.com/JusperLee/UtterancePIT-Speech-Separation/blob/master/model.py
"""

import warnings
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence

class PITNet(nn.Module):
    """
    model_type:
        0: original pit
        1: dvec
        2: simple mfcc
        3: mfcc + conv
        4: conv + mfcc
    """
    def __init__(self, n_fft=129, rnn='lstm', num_spks=2, num_layers=3, hidden_size=896, dropout=0.0, non_linear='relu', bidirectional=True, model_type=0, f_dim=0):
        super(PITNet, self).__init__()

        self.num_spks = num_spks
        rnn = rnn.upper()

        assert non_linear in ['relu', 'sigmoid', 'tanh'], 'Unsupported non-linear type:{}'.format(non_linear)
        assert rnn in ['RNN', 'LSTM', 'GRU'], 'Unsupported rnn type:{}'.format(rnn)
        
        if model_type in [1, 3, 4]:
            self.conv = nn.Sequential(
                # cnn1
                nn.ZeroPad2d((3, 3, 0, 0)),
                nn.Conv2d(1, 64, kernel_size=(1, 7), dilation=(1, 1)),
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn2
                nn.ZeroPad2d((0, 0, 3, 3)),
                nn.Conv2d(64, 64, kernel_size=(7, 1), dilation=(1, 1)),
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn3
                nn.ZeroPad2d(2),
                nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(1, 1)),
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn4
                nn.ZeroPad2d((2, 2, 4, 4)),
                nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(2, 1)), # (9, 5)
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn5
                nn.ZeroPad2d((2, 2, 8, 8)),
                nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(4, 1)), # (17, 5)
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn6
                nn.ZeroPad2d((2, 2, 16, 16)),
                nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(8, 1)), # (33, 5)
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn7
                nn.ZeroPad2d((2, 2, 32, 32)),
                nn.Conv2d(64, 64, kernel_size=(5, 5), dilation=(16, 1)), # (65, 5)
                nn.BatchNorm2d(64), nn.ReLU(),

                # cnn8
                nn.Conv2d(64, 8, kernel_size=(1, 1), dilation=(1, 1)), 
                nn.BatchNorm2d(8), nn.ReLU(),
            )
        if model_type == 0:
            input_size = n_fft
        elif model_type == 1 or model_type == 4:
            input_size = n_fft * 8 + f_dim
        elif model_type == 2:
            input_size = n_fft + f_dim
        elif model_type == 3:
            input_size = 8 * (n_fft + f_dim)
        
        self.model_type = model_type

        self.rnn = getattr(nn, rnn)(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout, bidirectional=bidirectional)
        self.rnn.flatten_parameters()

        self.dropout = nn.Dropout(dropout)

        self.linear = nn.ModuleList([
            nn.Linear(hidden_size * 2 if bidirectional else hidden_size, n_fft)
            for _ in range(self.num_spks)
        ])

        self.non_linear = {
            'relu': nn.functional.relu,
            'sigmoid': nn.functional.sigmoid,
            'tanh': nn.functional.tanh
        }[non_linear]

        self.n_fft = n_fft

    def forward(self, x, x_feature=None):
        if self.model_type == 1:
            x = x.unsqueeze(1)
            x = self.conv(x) 
            x = x.transpose(1, 2).contiguous()
            x = x.view(x.size(0), x.size(1), -1)
            x_feature = x_feature.unsqueeze(1)
            x_feature = x_feature.repeat(1, x.size(1), 1)
            x = torch.cat((x, x_feature), dim=2)

        elif self.model_type == 2:
            # print(x.shape, x_feature.shape)
            x = torch.cat((x, x_feature), dim=2)

        elif self.model_type == 3:
            x = torch.cat((x, x_feature), dim=2)
            x = x.unsqueeze(1)
            x = self.conv(x)
            x = x.transpose(1, 2).contiguous()
            x = x.view(x.size(0), x.size(1), -1)

        elif self.model_type == 4:
            x = x.unsqueeze(1)
            x = self.conv(x) 
            x = x.transpose(1, 2).contiguous()
            x = x.view(x.size(0), x.size(1), -1)
            x = torch.cat((x, x_feature), dim=2)

        # batch_size x time x frequency
        # -> batch_size x time x hidden_size
        x, _ = self.rnn(x)

        x = self.dropout(x)

        m = []
        for linear in self.linear:
            # batch_size x time x frequency
            y = linear(x)
            y = self.non_linear(y)
            m.append(y)
        return m

    def disturb(self, std):
        for p in self.parameters():
            noise = torch.zeros_like(p).normal_(0, std)
            p.data.add_(noise)

if __name__ == "__main__":
    x = torch.randn(1, 375, 129)

    model = PITNet(model_type = 0)
    temp = torch.stack(model.forward(x))
    s1, s2 = temp
    print(s1.shape)
