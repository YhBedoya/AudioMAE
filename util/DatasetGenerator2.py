from torch.utils.data import Dataset
import torch
import glob
import torchaudio
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os
import multiprocessing
from tqdm import tqdm

class DatasetGenerator(Dataset):
    """AudioSet Dataset."""

    def __init__(self, data_path, target_sample_rate,  transform):
        """
        Args:
            root_dir (string): Directory with all the sounds.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = data_path
        self.files = self._get_files() #glob.glob(self.root_dir + '/**/--0Oh0JxzjQ_30.wav', recursive=True)

    def __len__(self):
        #TODO: define how to count the len of the dataset
        return len(self.files)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index, audio_sample_path)
        signal = torch.load(audio_sample_path)

        return signal, label

    def _get_audio_sample_path(self, index):
        return self.files[index]

    def _get_audio_sample_label(self, index, audio_path):
        label = audio_path.split("/")[-2]
        return label

    def _get_files(self):
        print('enter to get files')
        audios = []
        for path, subdirs, files in os.walk(self.root_dir):
            print(f'Current path {path}')
            for file in files:
                audios.append(os.path.join(path, file))
            print(f'Cummulated number {len(audios)}')
        return audios

"""if __name__ == "__main__":
    AUDIO_DIR = "/home/yhbedoya/Repositories/AudioMAE/Data/"
    sample_rate = 16000

    hanningWindowSeconds = 25/1000
    win_length = int(sample_rate * hanningWindowSeconds)
    print(f'win_length: {win_length}')
    n_fft = int(win_length * 1.5)
    print(f'n_fft: {n_fft}')
    shiftsWindSeconds = 10/1000
    hop_len = int(sample_rate * shiftsWindSeconds)
    print(f'hop_len: {hop_len}')

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate= sample_rate,
        n_fft = n_fft,
        win_length = win_length,
        hop_length = hop_len,
        n_mels=128
    )

    dataset = DatasetGenerator(AUDIO_DIR, sample_rate, mel_spectrogram)

    DATASET_DIR = "/home/yhbedoya/Repositories/AudioMAE/Dataset/"
    processes = []

    #saveTensorToFile(dataset, 0)
    for i in tqdm(range(len(dataset))):
        saveTensorToFile(dataset, i)"""
