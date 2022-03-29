# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author      : Dasein
@Contact     : dasein_csd@163.com
@Github      : https://github.com/YagoToasa
-------------*-
@File        : libri_loader.py
@Version     : 1.0
@Software    : VS Code
@Description : LibriSpeech dataset loader for voice filter
"""
import warnings
warnings.filterwarnings("ignore")
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader

def create_dataloader(data_root_dir, batch_size, num_workers, data_type, shuffle, feature_type=None):
    assert data_type in ['train', 'test'], 'data_type should in list [\'train\', \'test\']'
    return DataLoader(dataset=VF_Dataset(data_root_dir=data_root_dir, data_type=data_type, feature_type=feature_type), 
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=shuffle)


class VF_Dataset(Dataset):
    """ LibriSpeech dataset loader for voice filter

    Atrributes:
        data_dir: str
            directory of the dataset, train or test
        data_type: str
            should be in list ['train', 'test', 'cv']
    """
    def __init__(self, data_root_dir, data_type, feature_type=None):
        def find_all(file_format):
            return sorted(glob.glob(os.path.join(self.data_root_dir, file_format)))

        assert data_type in ['train', 'test'], 'data_type should in list [\'train\', \'test\']'

        self.data_root_dir = os.path.join(data_root_dir, data_type)
        self.data_type = data_type
        self.feature_type = feature_type

        # feature
        if feature_type == 'dvec':
            self.feature_list = find_all('*-dvec.pt')
        elif feature_type == 'mfcc':
            self.feature_list = find_all('*-s1-mfcc.pt')

        # mix, s1, s2 - wav
        self.mix_wav_list = find_all('*-mix.wav')
        self.s1_wav_list = find_all('*-s1.wav')
        self.s2_wav_list = find_all('*-s2.wav')

        # mix, s1, s2 - spec
        self.mix_mag_list = find_all('*-mix.pt')
        self.s1_mag_list = find_all('*-s1.pt')
        self.s2_mag_list = find_all('*-s2.pt')

        self.mix_phase_list = find_all('*-mix-phase.pt')
        self.s1_phase_list = find_all('*-s1-phase.pt')
        self.s2_phase_list = find_all('*-s2-phase.pt')

        assert len(self.mix_wav_list) == len(self.mix_mag_list) == \
        len(self.s1_wav_list) == len(self.s1_mag_list) == \
        len(self.s2_wav_list) == len(self.s2_mag_list), 'number of training files must match'

    
    def __len__(self):
        return len(self.mix_mag_list)
    
    def __getitem__(self, idx):
        """ LibriSpeech get item

        Returns:
            train or cv:
                feature_type != None: feature, mix_mag, target_mag
                feature_type == None: mix_mag, target_mag
            test:
                feature_type == None: feature, mix_mag, mix_phase, target_mag, target_phase
                feature_type != None: mix_mag, mix_phase, target_mag, target_phase
        """
        mix_mag = torch.load(self.mix_mag_list[idx])
        target_mag = torch.load(self.s1_mag_list[idx])

        if self.feature_type is not None:
            feature = torch.load(self.feature_list[idx])

        if self.data_type == 'train':
            if self.feature_type is not None:
                return feature, mix_mag, target_mag
            else:
                return mix_mag, target_mag

        if self.data_type == 'test':
            mix_phase = torch.load(self.mix_phase_list[idx])
            target_phase = torch.load(self.s1_phase_list[idx])
            if self.feature_type is not None:
                return feature, mix_mag, mix_phase, target_mag, target_phase
            else:
                return mix_mag, mix_phase, target_mag, target_phase

