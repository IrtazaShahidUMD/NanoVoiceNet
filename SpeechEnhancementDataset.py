import torch
from torch.utils.data import Dataset, DataLoader
import torchaudio
import os

class SpeechDataset(Dataset):
    """
   A PyTorch Dataset class for loading paired speech enhancement data.

   Each sample consists of:
       - A noisy waveform: <base_path>/mix.wav
       - A clean ground-truth waveform: <base_path>/gt.wav

   The dataset trims or zero-pads audio to a fixed duration in seconds.
   """
    
    def __init__(self, dataset_directory, file_list, sample_rate=16000, max_duration_sec=1):
        """
        Initializes the dataset with paths and parameters.

        Args:
            dataset_directory (str): Base directory containing folders of audio samples.
            file_list (list): List of subfolder names for each audio sample.
            sample_rate (int): Expected sampling rate for all audio files.
            max_duration_sec (int): Maximum duration (in seconds) of each audio clip.
        """
        
        self.dataset_directory = dataset_directory
        self.file_list = file_list
        self.sample_rate = sample_rate
        self.max_samples = sample_rate * max_duration_sec

    def __len__(self):
        """Returns the total number of examples in the dataset."""
        
        return len(self.file_list)

    def __getitem__(self, idx):
        """
        Loads and returns the noisy-clean waveform pair for a given index.

        Returns:
            mix (Tensor): Noisy waveform [1, T]
            gt (Tensor): Clean (ground-truth) waveform [1, T]
        """
        
        base = self.file_list[idx]
        mix_path = self.dataset_directory + base + "/mix.wav"
        gt_path = self.dataset_directory + base + "/gt.wav"

        mix, sr1 = torchaudio.load(mix_path)
        gt, sr2 = torchaudio.load(gt_path)

        assert sr1 == self.sample_rate and sr2 == self.sample_rate, "Sample rate mismatch!"

        # Trim or pad to max_samples
        mix = self._trim_or_pad(mix)
        gt = self._trim_or_pad(gt)

        return mix, gt

    def _trim_or_pad(self, x):
        if x.size(1) > self.max_samples:
            x = x[:, :self.max_samples]
        elif x.size(1) < self.max_samples:
            pad_size = self.max_samples - x.size(1)
            x = torch.nn.functional.pad(x, (0, pad_size))
        return x