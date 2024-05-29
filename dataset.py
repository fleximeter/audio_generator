"""
File: dataset.py

This module has a dataset class for storing sequences.
"""

import featurizer
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
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
    def __init__(self, audio_list, min_sequence_length, max_sequence_length) -> None:
        """
        Makes an AudioDataset
        :param audio_list: A list of NumPy audio arrays to turn into a dataset
        :param min_sequence_length: The minimum sequence length
        :param max_sequence_length: The maximum sequence length
        """
        super(AudioDataset, self).__init__()
        self.min_sequence_length = min_sequence_length
        self.max_sequence_length = max_sequence_length
        self.data, self.labels = self._load_data(audio_list)
        
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
        return sample, *label
    
    def _load_data(self, audio_list) -> Tuple[list, list]:
        """
        Parses each MusicXML file and generates sequences and labels from it
        :param audio_list: A list of NumPy audio arrays to turn into a dataset
        """
        sequences = []
        labels = []
        for audio in audio_list:
            pass
        return sequences, labels
    
    def collate(batch):
        """
        Pads a batch in preparation for training. This is necessary
        because we expect the dataloader to randomize the data, which will
        mix sequences of different lengths. We will pad these sequences with empty
        entries so we can run everything through the model at the same time.
        :param batch: A batch produced by a DataLoader
        :return: The padded sequences, labels, and sequence lengths
        """
        # Sort the batch in order of sequence length. This is required by the pack_padded_sequences function. 
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        sequences, targets1, targets2 = zip(*batch)
        lengths = torch.tensor([seq.shape[0] for seq in sequences])
        sequences_padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        targets1 = torch.tensor(targets1)
        targets2 = torch.tensor(targets2)
        return sequences_padded, targets1, targets2, lengths
    
    def prepare_prediction(sequence, max_length) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Prepares a sequence for prediction. This function does the padding process
        just like the collate function, so the model behaves as expected.
        :param sequence: The sequence to prepare
        :param max_length: The maximum sequence length the model was trained on
        :return: The padded sequence and a list of lengths
        """
        lengths = torch.tensor([sequence.shape[1]])
        if sequence.shape[1] < max_length:
            zeros = torch.zeros((1, max_length - sequence.shape[1], sequence.shape[2]))
            sequence = torch.cat((sequence, zeros), dim=1)
        return sequence, lengths
    