import os
import sys
sys.path.append('../../')
import glob
import torch
import librosa
import torch.nn as nn
from utils.audio import Audio

def find_all(data_dir, file_format):
    return sorted(glob.glob(os.path.join(data_dir, file_format)))

def generate_mfcc(data_root_dir, data_type, n_mfcc=13):
    print('generate mfcc {}'.format(data_type))
    data_dir = os.path.join(data_root_dir, data_type)
    s1_list = find_all(data_dir, '*-s1.wav')
    total_samp = len(s1_list)
    count = 0
    for s1 in s1_list:
        count += 1
        if count % 1000 == 0:
            print('{} / {}'.format(total_samp, count))
        mfcc_path = s1.replace('.wav', '-mfcc.pt')
        wave, sr = librosa.load(s1, sr=16000)
        mel_spec = librosa.feature.melspectrogram(y=wave, sr=sr, n_fft=256, hop_length=64, fmax=sr/2, n_mels=40)
        mfcc = torch.from_numpy(librosa.feature.mfcc(y=wave, sr=sr, S=librosa.power_to_db(mel_spec), n_mfcc=n_mfcc)).T
        torch.save(mfcc, mfcc_path)

def generate_phase(data_root_dir, data_type='train'):
    print('generate phase')
    audio = Audio(n_fft=256, hop_length=64, win_length=256)
    data_dir = os.path.join(data_root_dir, data_type)
    wav_list = find_all(data_dir, '*.wav')
    total_samp = len(wav_list)
    count = 0
    for wav_path in wav_list:
        count += 1
        if count % 1000 == 0:
            print('{} / {}'.format(total_samp, count))
        save_path = wav_path.replace('.wav', '-phase.pt')
        waveform, _ = librosa.load(wav_path, sr=16000)
        _, phase = audio.wav2spec(waveform)
        torch.save(torch.from_numpy(phase), save_path)

if __name__ == '__main__':
    data_root_dir = '/Volumes/Chaos/Backup/Datasets/ST/MIX/'
    generate_mfcc(data_root_dir, 'train')
    generate_mfcc(data_root_dir, 'test')
    # generate_phase(data_root_dir)