from torch.utils.data import Dataset
import torch
import glob
import torchaudio
import matplotlib.pyplot as plt
import librosa.display
import numpy as np
import os

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
        self.transform = transform
        self.transformation = transform
        self.files = self._get_files() #glob.glob(self.root_dir + '/**/--0Oh0JxzjQ_30.wav', recursive=True)
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        #TODO: define how to count the len of the dataset
        return len(self.files)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index, audio_sample_path)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._mix_down(signal)
        signal = self._resample(signal, sr)
        signal = self._padding(signal)
        signal = self.transformation(signal)
        #fbank = torchaudio.compliance.kaldi.fbank(signal, htk_compat=True, sample_frequency=16000, use_energy=False, window_type='hanning', num_mel_bins=128, dither=0.0, frame_shift=10)
        signal = self._powerToDB(signal)
        signal = torch.squeeze(signal)
        return signal, label

    def _resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _powerToDB(self, signal):
        powerToDB = torchaudio.transforms.AmplitudeToDB(stype='power')
        signal = powerToDB(signal)
        return signal

    def _get_audio_sample_path(self, index):
        return self.files[index]

    def _get_audio_sample_label(self, index, audio_path):
        label = audio_path.split("/")[-2]
        return label

    def _padding(self, signal):
        pad_size = 3700
        diff = ((self.target_sample_rate * 10) + pad_size) - signal.shape[1]
        signal = torch.cat((torch.zeros(1,diff), signal), 1)
        return signal

    def _get_files(self):
        print('enter to get files')
        audios = []
        for path, subdirs, files in os.walk(self.root_dir):
            print(f'Current path {path}')
            for file in files:
                audios.append(os.path.join(path, file))
            print(f'Cummulated number {len(audios)}')
        return audios

def plotMelSpectrogram(signal,sr):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(signal,
                             x_axis="time",
                             y_axis="mel",
                             sr=sr)
    plt.colorbar(format="%+2.f")
    plt.show()

"""if __name__ == "__main__":
    AUDIO_DIR = "/home/yhbedoya/Repositories/AudioMAE/Data/"
    SAMPLE_RATE = 16000

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate= SAMPLE_RATE,
        n_fft = 400,
        #win_length = 400,
        hop_length = 160,
        n_mels=128
    )

    usd = DatasetGenerator(AUDIO_DIR, SAMPLE_RATE, mel_spectrogram)

    signal, label = usd[0]
    plotMelSpectrogram(np.squeeze(signal.detach().numpy()), SAMPLE_RATE)
    print(signal.shape)"""