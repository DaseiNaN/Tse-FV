# !/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch.nn.utils.rnn import pad_packed_sequence
from sklearn.cluster import KMeans

class DPCL(nn.Module):
    """ Deep clustering model

    Attributes:
        emb_dim: int, optional
            the dimension of embedding vector, default=40
        num_layer: int, optional
            the num of lstm layer, default=4
        input_size: int, optional
            the size of lstm layer's input, default=129
        hidden_size: int, optional
            the size of the lstm hide layer (the number of hidden cells)
            the dimension of the lstm layer's output are equal to the number of hidden cells, default=600
        dropout: float, optional
            if not zero, insert the Dropout layer on all layers except the last one, default=0.0
            dropout layer prevents over-fitting
        bidirectional: bool, optional
            if true, bidirectional lstm, default=True
        activation String, optional
            activation function, Default='Tanh'

        model_type:
            0: original dpcl
            1: dvec
            2: simple mfcc
            3: mfcc + conv
            4: conv + mfcc
        
    """
    def __init__(self, num_layer=4, n_fft=129, hidden_size=600, emb_dim=40, dropout=0.0, bidirectional=True, activation='Tanh', model_type=0, f_dim=0):
        super(DPCL, self).__init__()
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

        self.emb_dim = emb_dim
        self.blstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True, dropout=dropout, bidirectional=bidirectional)

        self.dropout = nn.Dropout(dropout)

        self.linear = nn.Linear(in_features=2 * hidden_size if bidirectional else hidden_size, out_features=n_fft * emb_dim)

        self.activation = getattr(torch.nn, activation)()

    def forward(self, x, x_feature=None):
        """ forward function

        Args:
            x: tensor
                shape = (batch_size x time x frequency)

        Returns:
            Batch x (T-F bin num) x Embedding_Dim
        """
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

        batch_size = x.shape[0]

        # batch_size x time x frequency
        # -> batch_size x time x hidden_size
        x, _ = self.blstm(x)

        # Prevent overfitting
        x = self.dropout(x)

        # -> batch_size x time x (frequency x emb_dim)
        x = self.linear(x)
        x = self.activation(x)
        
        # -> batch_size x (frequency x time) x emb_dim
        x = x.reshape(batch_size, -1, self.emb_dim)
        return x


if __name__ == '__main__':
    
    x = torch.randn(3, 120, 129)

    x_dvec = torch.randn(3, 256)

    x_mfcc = torch.randn(3, 120, 13)

    model = DPCL(model_type=4, f_dim=13)
    emd = model.forward(x, x_mfcc)
    print(emd.shape)


    