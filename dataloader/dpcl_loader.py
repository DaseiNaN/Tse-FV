# !/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author      : Dasein
@Contact     : dasein_csd@163.com
@Github      : https://github.com/YagoToasa
-------------*-
@File        : dpcl_loader.py
@Version     : 1.0
@Software    : VS Code
@Description : LibriSpeech dataset loader for dpcl
"""
import warnings
warnings.filterwarnings("ignore")
import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

def create_dataloader(data_root_dir, batch_size, num_workers, data_type, feature_type, shuffle):
    assert data_type in ['train', 'test'], 'data_type should in list [\'train\', \'test\']'
    return DataLoader(dataset=DPCL_Dataset(data_root_dir=data_root_dir, data_type=data_type, feature_type=feature_type), 
                                batch_size=batch_size,
                                num_workers=num_workers,
                                shuffle=shuffle)




class DPCL_Dataset(Dataset):
    """ LibriSpeech Dataset for DPCL

    Atrributes:
        data_root_dir: str
            root directory of the dataset, contains train/ and test/
        with_vp: bool
            load for voice print model or not, default True
        data_type: str
            should in list ['train', 'test']
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
        len(self.s2_wav_list) == len(self.s2_mag_list),'number of training files must match'

    
    def __len__(self):
        return len(self.mix_mag_list)
    
    def __getitem__(self, idx):
        """ LibriSpeech get item

        Returns:
            train:
                feature_type != None: feature, mix_mag, target_mask, non_silence
                feature_type == None: mix_mag, target_mask, non_silence
            test:
                feature_type != None: feature, mix_mag, mix_phase, target_mask, target_mag, target_phase, non_silence
                feature_type == None: mix_mag, mix_phase, target_mask, target_mag, target_phase, non_silence
        """
        # mix mag spec
        mix_mag = torch.load(self.mix_mag_list[idx])
        s1_mag = torch.load(self.s1_mag_list[idx])
        s2_mag = torch.load(self.s2_mag_list[idx])
        # target binary mask
        target_mask = torch.tensor(np.argmax(np.array([s1_mag.numpy(), s2_mag.numpy()]), axis = 0))
        # non_silence
        non_silence = torch.from_numpy(torch.ones(s1_mag.shape).numpy().astype(bool))

        if self.feature_type is not None:
            feature = torch.load(self.feature_list[idx])

        if self.data_type == 'train':
            if self.feature_type is not None:
                return feature, mix_mag, target_mask, non_silence
            else:
                return mix_mag, target_mask, non_silence
        
        if self.data_type == 'test':
            target_phase = torch.load(self.s1_phase_list[idx])
            target_mag = s1_mag
            mix_phase = torch.load(self.mix_phase_list[idx])
            if self.feature_type is not None:
                return feature, mix_mag, mix_phase, target_mask, target_mag, target_phase, non_silence
            else:
                return mix_mag, mix_phase, target_mask, target_mag, target_phase, non_silence

# if __name__ == '__main__':
#     dl = create_dataloader(data_root_dir='/Users/dasein/Downloads/LibriSpeech/temp', batch_size=2, num_workers=0, data_type='test', feature_type=None, shuffle=False)
#     for mix_mag, mix_phase, target_mask, target_mag, target_phase, non_silence in dl:
#         print(mix_mag.shape, mix_phase.shape, target_mask.shape, target_mag.shape, target_phase.shape, non_silence.shape)

#     dl = create_dataloader(data_root_dir='/Users/dasein/Downloads/LibriSpeech/temp', batch_size=2, num_workers=0, data_type='train', feature_type='dvec', shuffle=False)
#     for feature, mix_mag, target_mask, non_silence in dl:
#         print(feature.shape, mix_mag.shape, target_mask.shape, non_silence.shape)

