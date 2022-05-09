import os

import torch
import torchaudio
import pandas as pd
import tgt
import numpy as np

from tqdm import tqdm
from . import audio

class Preprocessor():
    def __init__(self, preprocess_hparm) -> None:
        self.preprocess_hparm = preprocess_hparm

        self.sampling_rate = preprocess_hparm.preprocessing.audio.sampling_rate

        self.STFT = torchaudio.transforms.MelSpectrogram(
            sample_rate = preprocess_hparm.preprocessing.audio.sampling_rate,
            n_fft= preprocess_hparm.preprocessing.stft.n_fft,
            win_length = preprocess_hparm.preprocessing.stft.win_length,
            hop_length = preprocess_hparm.preprocessing.stft.hop_length,
            f_min = preprocess_hparm.preprocessing.mel.mel_fmin,
            f_max = preprocess_hparm.preprocessing.mel.mel_fmax, 
            n_mels = preprocess_hparm.preprocessing.mel.n_mel_channels,
        )

        # self.STFT = audio.stft.TacotronSTFT(
        #     sample_rate = preprocess_hparm.preprocessing.audio.sampling_rate,
        #     n_fft= preprocess_hparm.preprocessing.stft.n_fft,
        #     win_length = preprocess_hparm.preprocessing.stft.win_length,
        #     hop_length = preprocess_hparm.preprocessing.stft.hop_length,
        #     f_min = preprocess_hparm.preprocessing.mel.mel_fmin,
        #     f_max = preprocess_hparm.preprocessing.mel.mel_fmax, 
        #     n_mels = preprocess_hparm.preprocessing.mel.n_mel_channels,
        # )

        self.df = pd.read_csv(os.path.join(preprocess_hparm.path.corpus_path, 'metadata.csv'), sep='|', header=None)

    def get_mel(self, wav_path, start, end):
        # Read and trim wav files
        wav, _ = torchaudio.load(wav_path)
        wav = wav[0][int(self.sampling_rate * start) : int(self.sampling_rate * end)]
        return torchaudio.transforms.AmplitudeToDB()(self.STFT(wav))

    def get_alignment(self, tier):
        sil_phones = ["sil", "sp", "spn"]

        phones = []
        durations = []
        start_time = 0
        end_time = 0
        end_idx = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # Trim leading silences
            if phones == []:
                if p in sil_phones:
                    continue
                else:
                    start_time = s

            if p not in sil_phones:
                # For ordinary phones
                phones.append(p)
                end_time = e
                end_idx = len(phones)
            else:
                # For silent phones
                phones.append(p)

            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.preprocess_hparm.preprocessing.stft.hop_length)
                    - np.round(s * self.sampling_rate / self.preprocess_hparm.preprocessing.stft.hop_length)
                )
            )

        # Trim tailing silences
        phones = phones[:end_idx]
        durations = durations[:end_idx]

        return phones, durations, start_time, end_time


    def __call__(self):
        phones = []
        for index, row in tqdm(self.df.iterrows(), total=len(self.df)):
            basename = row[0]
            basepath  = self.preprocess_hparm.path.corpus_path

            wav_path = os.path.join(basepath, 'wavs',f'{basename}.wav')
            tg_path = os.path.join(basepath, 'TextGrid',f'{basename}.TextGrid')


            # Get alignments
            textgrid = tgt.io.read_textgrid(tg_path)
            phone, duration, start, end = self.get_alignment(
                textgrid.get_tier_by_name("phones"))
            text = "{" + " ".join(phone) + "}"
            if start >= end:
                return None
            phones.append(text)
            
            # get and save melspec
            melspec = self.get_mel(wav_path, start, end)
            mel_save_path = os.path.join(basepath, 'mels')
            os.makedirs(mel_save_path, exist_ok=True)
            torch.save(melspec, os.path.join(mel_save_path, basename+'.pt'))

        df_path = os.path.join(self.preprocess_hparm.path.corpus_path, 'processed.csv')
        self.df['phones'] = phones
        self.df.to_csv(df_path, sep="|", header=None, index=None)
        return df_path