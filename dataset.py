"""
File: dataset.py

This module has a dataset class for storing sequences.
"""

import corpus
import featurizer
from torch.utils.data import Dataset
from typing import Tuple


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
    def __init__(self, directory, sequence_length) -> None:
        """
        Makes an AudioDataset
        :param directory: A list of NumPy audio arrays to turn into a dataset
        :param min_sequence_length: The minimum sequence length
        :param max_sequence_length: The maximum sequence length
        """
        super(AudioDataset, self).__init__()
        self.sequence_length = sequence_length
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
        Parses each MusicXML file and generates sequences and labels from it
        :param directory: A list of NumPy audio arrays to turn into a dataset
        """
        sequences = []
        labels = []
        audio_corpus = corpus.load_audio(directory)
        for audio in audio_corpus:
            featurizer.featurize(audio)
            file_sequences, file_labels = featurizer.make_n_gram_sequences(audio, self.sequence_length)
            sequences += file_sequences
            labels += file_labels
        return sequences, labels
    