from torch.utils.data import Dataset
import torch
import glob
import torchaudio
import matplotlib.pyplot as plt
import librosa.display
import numpy as np

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
        self.files = glob.glob(self.root_dir + '/**/--0Oh0JxzjQ_30.wav', recursive=True)
        self.transformation = transform
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
        print(f"Signal shape: {signal.shape}")
        signal = self.transformation(signal)
        signal = self._powerToDB(signal)
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

def plotMelSpectrogram(signal,sr):
    plt.figure(figsize=(25, 10))
    librosa.display.specshow(signal,
                             x_axis="time",
                             y_axis="mel",
                             sr=sr)
    plt.colorbar(format="%+2.f")
    plt.show()

if __name__ == "__main__":
    AUDIO_DIR = "/home/yhbedoya/Drive/Data Science and Engineering - PoliTo2/Thesis/Audioset"
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
    print(signal.shape)