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
@Description : LibriSpeech dataset loader
"""
import warnings
warnings.filterwarnings("ignore")
import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader

def create_libri_dataloader(data_dir, data_type, batch_size=1, num_workers=0, with_vp=True):
    def train_collate_fn_no_vp(batch):
        dvec_list = list()
        s1_mag_list = list()
        s2_mag_list = list()
        mix_mag_list = list()
        for dvec, s1_mag, s2_mag, mix_mag in batch:
            dvec_list.append(dvec)
            s1_mag_list.append(s1_mag)
            s2_mag_list.append(s2_mag)
            mix_mag_list.append(mix_mag)
        s1_mag_list = torch.stack(s1_mag_list, dim=0)
        s2_mag_list = torch.stack(s2_mag_list, dim=0)
        mix_mag_list = torch.stack(mix_mag_list, dim=0)
        return dvec_list, s1_mag_list, s1_mag_list, mix_mag_list

    def test_collate_fn_no_vp(batch):
        dvec_list = list()
        s1_mag_list = list()
        s1_phase_list = list()
        s2_mag_list = list()
        s2_phase_list = list()
        mix_mag_list = list()
        mix_phase_list =  list()
        # for dvec, s1_mag, s1_phase, mix_mag, mix_phase in batch:
        for dvec, s1_mag, s1_phase, s2_mag, s2_phase, mix_mag, mix_phase in batch:
            dvec_list.append(dvec)
            s1_mag_list.append(s1_mag)
            s1_phase_list.append(s1_phase)
            s2_mag_list.append(s2_mag)
            s2_phase_list.append(s2_phase)
            mix_mag_list.append(mix_mag)
            mix_phase_list.append(mix_phase)
        dvec_list = torch.stack(dvec_list, dim=0)
        s1_mag_list = torch.stack(s1_mag_list, dim=0)
        s1_phase_list = torch.stack(s1_phase_list, dim=0)
        s2_mag_list = torch.stack(s2_mag_list, dim=0)
        s2_phase_list = torch.stack(s2_phase_list, dim=0)
        mix_mag_list = torch.stack(mix_mag_list, dim=0)
        mix_phase_list = torch.stack(mix_phase_list, dim=0)
        return dvec_list, s1_mag_list, s1_phase_list, s2_mag_list, s2_phase_list, mix_mag_list, mix_phase_list


    def train_collate_fn(batch):
        dvec_list = list()
        s1_mag_list = list()
        mix_mag_list = list()
        for dvec, s1_mag, mix_mag in batch:
            dvec_list.append(dvec)
            s1_mag_list.append(s1_mag)
            mix_mag_list.append(mix_mag)
        s1_mag_list = torch.stack(s1_mag_list, dim=0)
        mix_mag_list = torch.stack(mix_mag_list, dim=0)
        return dvec_list, s1_mag_list, mix_mag_list

    def test_collate_fn(batch):
        dvec_list = list()
        s1_mag_list = list()
        s1_phase_list = list()
        mix_mag_list = list()
        mix_phase_list =  list()
        for dvec, s1_mag, s1_phase, mix_mag, mix_phase in batch:
            dvec_list.append(dvec)
            s1_mag_list.append(s1_mag)
            s1_phase_list.append(s1_phase)
            mix_mag_list.append(mix_mag)
            mix_phase_list.append(mix_phase)
        dvec_list = torch.stack(dvec_list, dim=0)
        s1_mag_list = torch.stack(s1_mag_list, dim=0)
        s1_phase_list = torch.stack(s1_phase_list, dim=0)
        mix_mag_list = torch.stack(mix_mag_list, dim=0)
        mix_phase_list = torch.stack(mix_phase_list, dim=0)
        return dvec_list, s1_mag_list, s1_phase_list, mix_mag_list, mix_phase_list

    if data_type == 'test':
        if with_vp:
            return DataLoader(dataset=Libri_Dataset(data_dir, data_type), 
                                batch_size=batch_size,
                                shuffle=False, 
                                collate_fn=test_collate_fn, 
                                num_workers=num_workers)
        else:
            return DataLoader(dataset=Libri_Dataset(data_dir, data_type, with_vp), 
                                batch_size=batch_size,
                                shuffle=False, 
                                collate_fn=test_collate_fn_no_vp, 
                                num_workers=num_workers)
    else:
        if with_vp:
            return DataLoader(dataset=Libri_Dataset(data_dir, data_type), 
                                batch_size=batch_size, 
                                shuffle=True,
                                num_workers=num_workers,
                                drop_last=True)
        else:
            return DataLoader(dataset=Libri_Dataset(data_dir, data_type, with_vp), 
                                batch_size=batch_size, 
                                shuffle=True,
                                num_workers=num_workers,
                                drop_last=True)
                            

class Libri_Dataset(Dataset):
    """ LibriSpeech Dataset

    Atrributes:
        data_dir: str
            directory of the dataset, train or test
        with_vp: bool
            load for voice print model or not, default True
        data_type: str
            should be in list ['train', 'test', 'cv']
    """
    def __init__(self, data_dir, data_type, with_vp=True):
        def find_all(file_format):
            return sorted(glob.glob(os.path.join(self.data_dir, file_format)))

        self.data_dir = os.path.join(data_dir, data_type)
        self.data_type = data_type

        self.dvec_list = find_all('*-dvec.pt')
        self.mix_wav_list = find_all('*-mix.wav')
        self.mix_mag_list = find_all('*-mix.pt')
        self.s1_wav_list = find_all('*-s1.wav')
        self.s1_mag_list = find_all('*-s1.pt')
        self.with_vp = with_vp

        if with_vp is False:
            self.s2_wav_list = find_all('*-s2.wav')
            self.s2_mag_list = find_all('*-s2.pt')

        # assert len(self.dvec_list) == \
        #     len(self.s1_wav_list) == len(self.s1_mag_list) == \
        #     len(self.s2_wav_list) == len(self.s2_mag_list) == \
        #     len(self.mix_wav_list) == len(self.mix_mag_list), 'number of training files must match'
        
        assert len(self.dvec_list) != 0, 'no training file found'

        if self.data_type == 'test':
            self.s1_phase_list = find_all('*-s1-phase.pt')
            self.s2_phase_list = find_all('*-s2-phase.pt')
            self.mix_phase_list = find_all('*-mix-phase.pt')


    
    def __len__(self):
        return len(self.dvec_list)
    
    def __getitem__(self, idx):
        """ LibriSpeech get item

        Returns:
            train or cv:
                with_vp == True: dvec, s1_mag, mix_mag
                with_vp == False: dvec, s1_mag, s2_mag, mix_mag
            test:
                with_vp == True: dvec, s1_mag, s1_phase, mix_mag, mix_phase
                with_vp == False: dvec, s1_mag, s1_phase, s2_mag, s2_phase, mix_mag, mix_phase
        """

        if self.data_type == 'test':
            dvec = torch.load(self.dvec_list[idx])
            s1_mag = torch.load(self.s1_mag_list[idx])
            s1_phase = torch.load(self.s1_phase_list[idx])
            mix_mag = torch.load(self.mix_mag_list[idx])
            mix_phase = torch.load(self.mix_phase_list[idx])

            if self.with_vp:
                return dvec, s1_mag, s1_phase, mix_mag, mix_phase
            else:
                s2_mag = torch.load(self.s2_mag_list[idx])
                s2_phase = torch.load(self.s2_phase_list[idx])
            return dvec, s1_mag, s1_phase, s2_mag, s2_phase, mix_mag, mix_phase
        else:
            dvec = torch.load(self.dvec_list[idx])
            s1_mag = torch.load(self.s1_mag_list[idx])
            mix_mag = torch.load(self.mix_mag_list[idx])

            if self.with_vp:
                return dvec, s1_mag, mix_mag
            else:
                s2_mag = torch.load(self.s2_mag_list[idx])
            return dvec, s1_mag, s2_mag, mix_mag

