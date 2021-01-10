"""Data module. Downloads data. preprocess data. Data feeder pipeline
    TODO:
        * Build a more memory efficient data feeder pipeline
"""
import os
import random
import pickle
from tqdm import tqdm

import pandas as pd
import numpy as np
import scipy.io.wavfile as wave

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from .speech_utils import get_speech_features_from_file
from .alphabet import Alphabet


class CSVDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_files, alphabet, audio_config={}, shuffle=False, path_prefix=''):
        """
        Args:
            csv_files (list): List of absolute path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.DataFrame()

        for file in csv_files:
            self.data = pd.concat([self.data, pd.read_csv(file, sep=',')])

        if(shuffle):
            self.data = self.data.sample(frac=1).reset_index(drop=True)
        
        self.alphabet = alphabet
        self.audio_config = audio_config
        self.max_duration = audio_config.get('max_duration', 16.7)
        self.audio_config['sample_freq'] = self.audio_config.get('sample_freq', 16000)
        self.audio_config['num_audio_features'] = self.audio_config.get('num_audio_features', 64)
    
        self.path_prefix = path_prefix



    def __len__(self):
        return len(self.data)

    
    def __getitem__(self, idx):
        wav_path = self.data.iloc[idx]['wav_filename']
        if(self.path_prefix != ''):
            wav_path ='/'.join([self.path_prefix] + wav_path.split('/')[2:])
        transcript = self.data.iloc[idx]['transcript']
        features, duration = get_speech_features_from_file(wav_path, self.audio_config)
        if(duration > self.max_duration):
            return {'wav_path' : wav_path, 
            'error' : "{} is longer than max_duration of {}".format(duration, self.max_duration)}

        label = self.alphabet.encode(transcript)

        return {'wav_path': wav_path, 'feats' : features, 'labs' : label}


def pad_collate(batch): 
    feats_list = []
    labs_list = []
    feats_len = []
    labs_len = []
    wav_paths = []
    for b in batch:
        if('error' in b): continue
        feats_list.append(torch.Tensor(b['feats']))
        labs_list.append(torch.IntTensor(b['labs']))
        feats_len.append(len(b['feats']))
        labs_len.append(len(b['labs']))
        wav_paths.append(b['wav_path'])

    maxlen = max(feats_len)

    feats_len = torch.tensor([fl/maxlen for fl in feats_len])
    feats_padded = pad_sequence(feats_list, batch_first=True, padding_value=0)
    labs_padded = pad_sequence(labs_list, batch_first=True, padding_value=0)
    return {'feats' : feats_padded, 'labs' : labs_padded, 'feats_len' : feats_len, 
            'labs_len' : torch.IntTensor(labs_len), 'wav_path' : wav_paths}


class AudioDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        """
        Creates a data loader for AudioDatasets.
        """
        super(AudioDataLoader, self).__init__(*args, **kwargs)
        self.collate_fn = pad_collate
        

if __name__ == "__main__":
    alp = Alphabet('/root/wav2letter/thai_eng_alphabet.txt')
    dataset = CSVDataset(["/root/data/youtube/youtube-dev-shuffle.csv"], alp)
    
    with open('youtube-dev-16s.csv', 'w') as out:
        for i, data in enumerate(dataset):
            if('error' not in data):
                out.write("{},{}\n".format(data['wav_path'], alp.decode(data['labs'])))
            