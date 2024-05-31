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
    audio["num_spectrogram_frames"] = audio["power_spectrogram"].shape[-1]
    analysis.analyzer(audio)
    del audio["magnitude_spectrogram"]  # We don't need the regular spectrum anymore - just the power spectrum


def make_n_gram_sequences(featurized_audio, n) -> list:
    """
    Makes N-gram sequences from a tokenized dataset
    :param tokenized_dataset: The tokenized dataset
    :param n: The length of the n-grams
    :return: X, y
    X is a list of N-gram tensors
      (dimension 1 is the entry in the N-gram)
      (dimension 2 has the features of the entry)
    y is a list of associated label tensors
    """
    x = []
    y = []
    for j in range(n, featurized_audio["num_spectrogram_frames"] - 2):
        sequence = []
        for k in range(j-n, j):
            # The features
            element = torch.hstack((
                featurized_audio["melscale_spectrogram"][0, :, k],
                featurized_audio["spectral_centroid"][0, k],
                featurized_audio["spectral_entropy"][0, k],
                featurized_audio["spectral_flatness"][0, k],
                featurized_audio["spectral_slope"][0, k],
                featurized_audio["spectral_y_int"][0, k],
                featurized_audio["spectral_roll_off_0.5"][0, k],
                featurized_audio["spectral_roll_off_0.75"][0, k],
                featurized_audio["spectral_roll_off_0.9"][0, k],
                featurized_audio["spectral_roll_off_0.95"][0, k],
                featurized_audio["spectral_variance"][0, k],
                featurized_audio["spectral_skewness"][0, k],
                featurized_audio["spectral_kurtosis"][0, k]
            ))
            sequence.append(element)
        x.append(torch.vstack(sequence))

        # The labels are just the next STFT frame
        y.append(featurized_audio["power_spectrogram"][0, :, j])
    return x, y
