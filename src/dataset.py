import os

import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence


import pandas as pd
import numpy as np

from .text import text_to_sequence
    
class Dataset(Dataset):
    '''
    Dataset loader
    '''
    def __init__(self, preprocess_hparm, filepath:str, sep='|'):
        self.preprocess_hparm = preprocess_hparm
        self.df = pd.read_csv(os.path.join(self.preprocess_hparm.path.preprocessed_path, filepath), header=None, sep=sep)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        file_name = self.df.iloc[idx][0]
        print(len(self.df.iloc[idx][3]))
        phone = torch.tensor(text_to_sequence(self.df.iloc[idx][3], self.preprocess_hparm.preprocessing.text.text_cleaners))

        mel_path = os.path.join( self.preprocess_hparm.path.preprocessed_path, "mels", f"{file_name}.pt")
        mel = torch.load(mel_path)

        return phone, mel
    
    def collate_fn(self, data):
        # split values into own varable
        texts = [i[0] for i in data]
        mels = [i[1].T for i in data]

        # get original length of elements
        text_lens = [text.shape[0] for text in texts]
        mel_lens = [mel.shape[1] for mel in mels]

        # zero pad
        text = pad_sequence(texts).T
        mels = pad_sequence(mels).permute(1,2,0)

        return text, text_lens, mels, mel_lens

def get_dataloader(dataset:Dataset, train_hparm):
    return DataLoader(dataset, train_hparm.optimizer.batch_size, train_hparm.optimizer.shuffle, collate_fn=dataset.collate_fn)