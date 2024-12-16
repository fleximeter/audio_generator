"""
File: featurizer.py

This module has featurization functions for preparing data to run through the model,
and for postprocessing generated data. It also has a dataset class for storing
sequences.
"""

import caus.analysis as analysis
import numpy as np
import os
import torch
import torchaudio


FFT_SIZE = 1024
NUM_MELS = FFT_SIZE // 16
NUM_FEATURES = FFT_SIZE + 2 + 10


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
        

def featurize(audio: dict):
    """
    Adds additional feature information to an audio file dictionary.
    The additional information includes the STFT spectrogram and spectral features.
    :param audio: The audio file dictionary to featurize
    """
    # Get the STFT data from the audio file and add it to the audio file dictionary
    spectrogram_transform = torchaudio.transforms.Spectrogram(n_fft=FFT_SIZE, power=None, normalized=True)
    melscale_transform = torchaudio.transforms.MelScale(NUM_MELS, audio["sample_rate"], n_stft=FFT_SIZE // 2 + 1)
    complex_out = spectrogram_transform(audio["audio"])
    audio["magnitude_spectrogram"] = torch.sqrt(torch.square(torch.real(complex_out)) + torch.square(torch.imag(complex_out))).numpy()
    audio["phase_spectrogram"] = torch.atan2(torch.imag(complex_out), torch.real(complex_out)).numpy()
    audio["power_spectrogram"] = np.square(audio["magnitude_spectrogram"])
    # audio["melscale_spectrogram"] = melscale_transform(audio["power_spectrogram"])
    audio["num_spectrogram_frames"] = audio["power_spectrogram"].shape[-1]
    
    # Analyze the audio file
    analysis.analyzer(audio, FFT_SIZE)
    audio["magnitude_spectrogram"] = torch.from_numpy(audio["magnitude_spectrogram"])
    audio["phase_spectrogram"] = torch.from_numpy(audio["phase_spectrogram"])
    audio["power_spectrogram"] = torch.from_numpy(audio["power_spectrogram"])
    audio["spectral_centroid"] = torch.from_numpy(audio["spectral_centroid"])
    audio["spectral_variance"] = torch.from_numpy(audio["spectral_variance"])
    audio["spectral_skewness"] = torch.from_numpy(audio["spectral_skewness"])
    audio["spectral_kurtosis"] = torch.from_numpy(audio["spectral_kurtosis"])
    audio["spectral_entropy"] = torch.from_numpy(audio["spectral_entropy"])
    audio["spectral_flatness"] = torch.from_numpy(audio["spectral_flatness"])
    audio["spectral_roll_off_0.5"] = torch.from_numpy(audio["spectral_roll_off_0.5"])
    audio["spectral_roll_off_0.75"] = torch.from_numpy(audio["spectral_roll_off_0.75"])
    audio["spectral_roll_off_0.9"] = torch.from_numpy(audio["spectral_roll_off_0.9"])
    audio["spectral_roll_off_0.95"] = torch.from_numpy(audio["spectral_roll_off_0.95"])
    audio["spectral_slope"] = torch.from_numpy(audio["spectral_slope"])
    audio["spectral_slope_0:1kHz"] = torch.from_numpy(audio["spectral_slope_0:1kHz"])
    audio["spectral_slope_1:5kHz"] = torch.from_numpy(audio["spectral_slope_1:5kHz"])
    audio["spectral_slope_0:5kHz"] = torch.from_numpy(audio["spectral_slope_0:5kHz"])


def load_audio_file(file_name: str) -> dict:
    """
    Reads an audio file and featurizes it. Returns the audio file dictionary.
    :param file_name: The file name
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

    featurize(audio_dictionary)
    return audio_dictionary


def prepare_robust_scaler(tensor_list):
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


def make_feature_frame(fft_mags, fft_phases, sample_rate):
    """
    Makes a feature dictionary for a FFT frame. This is needed if we have
    predicted a FFT frame and need to produce features for it.
    :param fft_mags: The FFT magnitude spectrum
    :param fft_phases: The FFT phase spectrum
    :param sample_rate: The sample rate
    :return: The feature dictionary
    """
    melscale_transform = torchaudio.transforms.MelScale(NUM_MELS, sample_rate, n_stft=FFT_SIZE // 2 + 1)
    vector = {
        "magnitude_spectrogram": torch.reshape(fft_mags, (1, fft_mags.shape[-1], 1)),
        "phase_spectrogram": torch.reshape(fft_phases, (1, fft_phases.shape[-1], 1)),
        "sample_rate": sample_rate,
        "channels": 1
    }
    vector["power_spectrogram"] = torch.square(vector["magnitude_spectrogram"])
    # vector["melscale_spectrogram"] = melscale_transform(vector["power_spectrogram"])
    vector["num_spectrogram_frames"] = 1
    analysis.analyzer(vector, FFT_SIZE)
    del vector["power_spectrogram"]
    return vector


def make_feature_matrix(feature_dict):
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
            feature_dict["magnitude_spectrogram"][0, :, i],
            feature_dict["phase_spectrogram"][0, :, i],
            feature_dict["spectral_centroid"][0, i],
            feature_dict["spectral_entropy"][0, i],
            feature_dict["spectral_flatness"][0, i],
            feature_dict["spectral_roll_off_0.5"][0, i],
            feature_dict["spectral_roll_off_0.75"][0, i],
            feature_dict["spectral_roll_off_0.9"][0, i],
            feature_dict["spectral_roll_off_0.95"][0, i],
            feature_dict["spectral_variance"][0, i],
            feature_dict["spectral_skewness"][0, i],
            feature_dict["spectral_kurtosis"][0, i]
        ))
        sequence.append(element)
    sequence = torch.vstack(sequence)
    return torch.reshape(sequence, (feature_dict["num_spectrogram_frames"],) + sequence.shape)


def make_n_gram_sequences(featurized_audio, n) -> list:
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
                featurized_audio["magnitude_spectrogram"][0, :, k],
                featurized_audio["phase_spectrogram"][0, :, k],
                featurized_audio["spectral_centroid"][0, k],
                featurized_audio["spectral_entropy"][0, k],
                featurized_audio["spectral_flatness"][0, k],
                featurized_audio["spectral_roll_off_0.5"][0, k],
                featurized_audio["spectral_roll_off_0.75"][0, k],
                featurized_audio["spectral_roll_off_0.9"][0, k],
                featurized_audio["spectral_roll_off_0.95"][0, k],
                featurized_audio["spectral_variance"][0, k],
                featurized_audio["spectral_skewness"][0, k],
                featurized_audio["spectral_kurtosis"][0, k]
            ))
            z = torch.isnan(element)
            if z.any():
                print(f"NaN found at {torch.nonzero(z)}")
            sequence.append(element)
        x.append(torch.vstack(sequence))

        # The labels are just the next STFT frame
        y.append(torch.hstack((featurized_audio["magnitude_spectrogram"][0, :, j], featurized_audio["phase_spectrogram"][0, :, j])))
    return x, y
