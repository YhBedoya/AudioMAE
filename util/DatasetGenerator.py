from torch.utils.data import Dataset
import torch
import glob
import torchaudio



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
        self.files = glob.glob(self.root_dir + '/**/*.wav', recursive=True)
        self.transformation = transform
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        #TODO: define how to count the len of the dataset
        return len(self.files)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index, audio_sample_path)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample(signal, sr)
        signal = self._mix_down(signal)
        signal = self.transformation(signal)
        return signal, label

        return signal, label

    def _resample(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down(selfself, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        return self.files[index]

    def _get_audio_sample_label(self, index, audio_path):
        label = audio_path.split("\\")[-2]
        return label

if __name__ == "__main__":
    AUDIO_DIR = "G:\My Drive\Data Science and Engineering - PoliTo2\Thesis\models\AudioMAE\scripts\datasets\\audioset201906\\audios\\balanced_train_segments"
    SAMPLE_RATE = 16000

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate= 16000, #Define,
        n_fft = 1024,
        hop_length = 512,
        n_mels=64
    )

    usd = DatasetGenerator(AUDIO_DIR, SAMPLE_RATE, mel_spectrogram)

    signal, label = usd[0]
    a = 1