"""
File: featurizer.py

This module has featurization functions for preparing data to run through the model,
and for postprocessing generated data. It also has a dataset class for storing
sequences.
"""

import aus_analyzer
import numpy as np
import os
import torch
import torchaudio
from typing import Tuple


# Features to include beyond magnitude and phase spectra
ADDITIONAL_FEATURES = {
    "spectral_centroid",
    "spectral_entropy",
    "spectral_flatness",
    "spectral_roll_off_50",
    "spectral_roll_off_75",
    "spectral_roll_off_90",
    "spectral_roll_off_95",
    "spectral_variance",
    "spectral_skewness",
    "spectral_kurtosis",
    "spectral_slope",
    "spectral_slope_0_1_khz",
    "spectral_slope_1_5_khz",
    "spectral_slope_0_5_khz"
}
NUM_ADDITIONAL_FEATURES = len(ADDITIONAL_FEATURES)


class RobustScaler:
    def __init__(self, median, iqr):
        """
        Initializes the RobustScaler
        :param median: The median of the data to scale
        :param iqr: The IQR of the data to scale
        """
        self.median = torch.tensor(median)
        self.iqr = torch.tensor(iqr)

    def __call__(self, data: torch.Tensor, *args, **kwds):
        return (data - self.median) / self.iqr
        

def featurize(audio: dict, fft_size: int=1024) -> None:
    """
    Adds additional feature information to an audio file dictionary.
    The additional information includes the STFT spectrogram and spectral features.
    :param audio: The audio file dictionary to featurize
    :param fft_size: The FFT size
    """
    # Get the STFT data from the audio file and add it to the audio file dictionary
    audio.update(aus_analyzer.analyze_stft(audio["path"], fft_size, 8))
    audio["num_spectrogram_frames"] = audio["magnitude_spectrogram"].shape[0]
    audio["magnitude_spectrogram"] = torch.from_numpy(audio["magnitude_spectrogram"])
    audio["phase_spectrogram"] = torch.from_numpy(audio["phase_spectrogram"])
    for feature in ADDITIONAL_FEATURES:
        audio[feature] = torch.from_numpy(audio[feature])


def load_audio_file(file_name: str, fft_size: int) -> dict:
    """
    Reads an audio file and featurizes it. Returns the audio file dictionary.
    :param file_name: The file name
    :param fft_size: The FFT size to use
    :return: An audio file dictionary
    """
    audio, sample_rate = torchaudio.load(file_name)

    # Mix down stereo file
    if audio.ndim == 2:
        audio = torch.sum(audio, dim=0)

    audio_dictionary = {
        "name": os.path.split(file_name)[-1],
        "path": file_name,
        "sample_rate": sample_rate,
        "audio": audio,
        "frames": audio.shape[-1],
        "duration": audio.shape[-1] / sample_rate,
        "channels": 1
    }

    featurize(audio_dictionary, fft_size)
    return audio_dictionary


def prepare_robust_scaler(tensor_list: list) -> Tuple[np.float32, np.float32]:
    """
    Prepares the Robust Scaler by finding the median and IQR of the tensors in the provided list
    :param tensor_list: A list of Tensors
    :return: median, iqr
    """
    new_tensor_list = [tensor.flatten() for tensor in tensor_list]
    new_tensor = torch.cat(new_tensor_list)
    # print("nans?", torch.nonzero(torch.isnan(new_tensor)))
    new_np_arr = new_tensor.numpy()
    return np.median(new_np_arr), np.percentile(new_np_arr, 75) - np.percentile(new_np_arr, 25)


def make_feature_frame(fft_mags: torch.Tensor, fft_phases: torch.Tensor, sample_rate: int, fft_size: int) -> dict:
    """
    Makes a feature dictionary for a FFT frame. This is needed if we have
    predicted a FFT frame and need to produce features for it.
    :param fft_mags: The FFT magnitude spectrum
    :param fft_phases: The FFT phase spectrum
    :param sample_rate: The sample rate
    :param fft_size: The FFT size
    :return: The feature dictionary
    """
    # melscale_transform = torchaudio.transforms.MelScale(NUM_MELS, sample_rate, n_stft=fft_size // 2 + 1)
    vector = {
        "magnitude_spectrum": fft_mags,
        "phase_spectrum": fft_phases,
        "sample_rate": sample_rate,
        "channels": 1
    }
    vector.update(aus_analyzer.analyze_rfft(fft_mags.numpy(), fft_size, sample_rate))
    for key, val in vector.items():
        if type(val) == np.ndarray:
            vector[key] = torch.nan_to_num(torch.from_numpy(val))
    return vector


def make_feature_matrix(feature_dict: dict) -> torch.Tensor:
    """
    Makes a feature matrix for a feature frame. This matrix can then be concatenated
    to an existing feature matrix for a N-gram sequence to extend the sequence.
    The most obvious use for this is to create a vector for a single FFT frame.
    :param feature_dict: The feature frame
    :return: The feature matrix. If a vector, the first dimension will have size 1.
    """
    sequence = []
    for i in range(feature_dict["num_spectrogram_frames"]):
        element = torch.hstack((
            feature_dict["magnitude_spectrogram"][i, :],
            feature_dict["phase_spectrogram"][i, :],
            torch.tensor([feature_dict[feature][i] for feature in ADDITIONAL_FEATURES], dtype=torch.float32)
        ))
        # If the FFT is run on a zero vector, the spectral centroid is NaN because of division by 0.
        # Therefore, other features will also be NaN. So we need to simply override these and set them to 0.
        element = torch.nan_to_num(element)
        sequence.append(element)
    sequence = torch.vstack(sequence)
    return sequence


def make_feature_vector(feature_dict: dict) -> torch.Tensor:
    """
    Makes a feature vector for a feature dictionary. This vector can then be concatenated
    to an existing feature matrix for a N-gram sequence to extend the sequence.
    The most obvious use for this is to create a vector for a single FFT frame.
    :param feature_dict: The feature frame
    :return: The feature vector
    """
    element = torch.hstack((
        feature_dict["magnitude_spectrum"],
        feature_dict["phase_spectrum"],
        torch.tensor([feature_dict[feature] for feature in ADDITIONAL_FEATURES], dtype=torch.float32)
    ))
    # If the FFT is run on a zero vector, the spectral centroid is NaN because of division by 0.
    # Therefore, other features will also be NaN. So we need to simply override these and set them to 0.
    element = torch.nan_to_num(element)
    element = torch.unsqueeze(element, 0)
    return element


def make_n_gram_sequences(featurized_audio: dict, n: int) -> Tuple[list, list]:
    """
    Makes N-gram sequences from an audio dictionary
    :param featurized_audio: The audio dictionary
    :param n: The length of the n-grams
    :return: [X], [y]
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
                featurized_audio["magnitude_spectrogram"][k, :],
                featurized_audio["phase_spectrogram"][k, :],
                torch.tensor([featurized_audio[feature][k] for feature in ADDITIONAL_FEATURES], dtype=torch.float32)
            ))
            # If the FFT is run on a zero vector, the spectral centroid is NaN because of division by 0.
            # Therefore, other features will also be NaN. So we need to simply override these and set them to 0.
            sequence.append(torch.nan_to_num(element))
        x.append(torch.vstack(sequence))
        # The labels are just the next STFT frame
        y.append(torch.hstack((featurized_audio["magnitude_spectrogram"][j, :], featurized_audio["phase_spectrogram"][j, :])))
    return x, y
