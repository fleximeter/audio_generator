"""
File: dataset.py

This module has a dataset class for storing sequences.
"""

import corpus
import featurizer
import torch
from torch.utils.data import Dataset
from typing import Tuple
import multiprocessing as mp


class AudioDataset(Dataset):
    """
    Makes a dataset of sequenced audio frames based on an audio corpus. This dataset
    will make sequences of audio frames and labels for the next note in the sequence,
    for generative training. It will exhaustively make sequences between a specified
    minimum sequence length and maximum sequence length, and these sequences should
    be provided to a DataLoader in shuffled fashion. Because the sequence lengths
    vary, it is necessary to provide a collate function to the DataLoader, and a
    collate function is provided as a static function in this class.
    """
    def __init__(self, directory, sequence_length, fft_size, mean=None, iqr=None) -> None:
        """
        Makes an AudioDataset
        :param directory: A list of NumPy audio arrays to turn into a dataset
        :param sequence_length: The sequence length
        :param fft_size: The FFT size for the dataset
        :param mean: The mean for Robust Scaling. If None, will be computed.
        :param iqr: The IQR for Robust Scaling. If None, will be computed.
        """
        super(AudioDataset, self).__init__()
        self.sequence_length = sequence_length
        self.fft_size = fft_size
        self.mean = mean
        self.iqr = iqr
        self.data, self.labels = self._load_data(directory)
        
    def __len__(self) -> int:
        """
        Gets the number of entries in the dataset
        :return: The number of entries in the dataset
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Gets the next item and its labels in the dataset
        :return: sample, labels
        """
        sample = self.data[idx]
        label = self.labels[idx]
        return sample, label
    
    def _load_data(self, directory) -> Tuple[list, list]:
        """
        Loads the files and generates sequences and labels from them
        :param directory: A list of NumPy audio arrays to turn into a dataset
        """
        # These lists hold feature matrices and their associated labels.
        sequences = []
        labels = []

        # Load the audio corpus. audio_corpus is a list of audio file dictionaries,
        # which hold audio file information and the sample array.
        audio_corpus = corpus.load_audio(directory)

        # Featurize each audio file dictionary
        ds_list = []
        for audio in audio_corpus:
            featurizer.featurize(audio, self.fft_size)
            for key, val in audio.items():
                if type(val) == torch.Tensor:
                    ds_list.append(val)
        
        # Prepare data scaling
        if self.mean is None or self.iqr is None:
            self.mean, self.iqr = featurizer.prepare_robust_scaler(ds_list)
        else:
            self.mean = torch.tensor(self.mean)
            self.iqr = torch.tensor(self.iqr)

        # self.data_scaler = featurizer.RobustScaler(self.mean, self.iqr)

        # Make n-gram sequences from each audio file
        for audio in audio_corpus:
            file_sequences, file_labels = featurizer.make_n_gram_sequences(audio, self.sequence_length)
            sequences += file_sequences
            labels += file_labels

        # Scale the data
        # for i in range(len(sequences)):
        #     sequences[i] = self.data_scaler(sequences[i])

        return sequences, labels
        