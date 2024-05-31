"""
File: featurizer.py

This module has featurization functions for preparing data to run through the model,
and for postprocessing generated data. It also has a dataset class for storing
sequences.
"""

import feature_definitions
import torch
import torch.nn.functional as F
import math
import numpy as np
from fractions import Fraction
import torchaudio
import analysis


FFT_SIZE = 1024
NUM_MELS = FFT_SIZE // 16


def featurize(audio) -> list:
    """
    Loads an audio file and featurizes it
    :param audio: The audio to load
    """
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=FFT_SIZE, power=1)
    power_spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=FFT_SIZE)
    melscale_transform = torchaudio.transforms.MelScale(NUM_MELS, audio["sample_rate"], n_stft=FFT_SIZE // 2 + 1)
    audio["magnitude_spectrogram"] = spectrogram_transform(audio["audio"])
    audio["power_spectrogram"] = power_spectrogram_transform(audio["audio"])
    audio["melscale_spectrogram"] = melscale_transform(audio["power_spectrogram"])
    analysis.analyzer(audio)
    del audio["magnitude_spectrogram"]  # We don't need the regular spectrum anymore - just the power spectrum


def make_labels(x) -> list:
    """
    Generates a label list for a list of sequences (2D tensors). The label is
    calculated for a particular index in dimension 1 of the 2D tensors.
    Dimension 1 is the batch length
    Dimension 2 is the total sequence length
    Dimension 3 is the features of individual notes in the sequence
    :param x: A list of sequences
    :return: A list of label tuples.
    """
    y = []
    return y


def make_n_gram_sequences(tokenized_dataset, n) -> list:
    """
    Makes N-gram sequences from a tokenized dataset
    :param tokenized_dataset: The tokenized dataset
    :param n: The length of the n-grams
    :return: X
    X is a list of N-gram tensors
      (dimension 1 is the entry in the N-gram)
      (dimension 2 has the features of the entry)
    """
    x = []
    for j in range(n, tokenized_dataset.shape[0] - 1):
        y = []
        for k in range(j-n, j):
            y.append(tokenized_dataset[k, :])
        x.append(torch.vstack(y))
    return x


def make_one_hot_features(dataset: list, batched=True) -> torch.Tensor:
    """
    Turns a dataset into a list of one-hot-featured instances in preparation for 
    running it through a model. You can use this for making predictions if you want.
    :param dataset: The dataset to make one-hot
    :param batched: Whether or not the data is expected to be in batched format (3D) 
    or unbatched format (2D). If you will be piping this data into the make_sequences 
    function, it should not be batched. In all other cases, it should be batched.
    :return: The one-hot data as a 2D or 3D tensor
    """
    instances = []
    for instance in dataset:
        # One-hots
        pass
    instances = torch.vstack(instances)
    if batched:
        instances = torch.reshape(instances, (1,) + instances.shape)
    return instances


def retrieve_class_dictionary(prediction: tuple) -> dict:
    """
    Retrives a predicted class's information based on its id
    :param prediction: The prediction tuple
    :return: The prediction dictionary
    """
    note = {}
    return note


