"""
File: analysis.py
Author: Jeff Martin
Date: 12/17/23

Audio analysis tools developed from Eyben, "Real-Time Speech and Music Classification"
"""

import numpy as np
import torch.fft
import sklearn.linear_model


def analyzer(audio):
    """
    Runs a suite of analysis tools on a provided NumPy array of audio samples
    :param audio: An audio Tensor
    """
    rfftfreqs = torch.fft.rfftfreq(audio["magnitude_spectrogram"].shape[-1], 1/audio["sample_rate"])
    audio['pitch'] = None
    # results['midi'] = midi_estimation_from_pitch(results['pitch'])
    audio['spectral_centroid'] = spectral_centroid(audio["magnitude_spectrogram"], rfftfreqs)
    audio['spectral_entropy'] = spectral_entropy(audio["power_spectrogram"])
    audio['spectral_flatness'] = spectral_flatness(audio["magnitude_spectrogram"])
    audio['spectral_slope'] = spectral_slope(audio["magnitude_spectrogram"], rfftfreqs)
    audio['spectral_roll_off_0.5'] = spectral_roll_off_point(audio["power_spectrogram"], rfftfreqs, 0.5)
    audio['spectral_roll_off_0.75'] = spectral_roll_off_point(audio["power_spectrogram"], rfftfreqs, 0.75)
    audio['spectral_roll_off_0.9'] = spectral_roll_off_point(audio["power_spectrogram"], rfftfreqs, 0.9)
    audio['spectral_roll_off_0.95'] = spectral_roll_off_point(audio["power_spectrogram"], rfftfreqs, 0.95)
    audio['zero_crossing_rate'] = zero_crossing_rate(audio["audio"], audio["sample_rate"])
    audio.update(spectral_moments(audio, rfftfreqs))
    

def midi_estimation_from_pitch(frequency):
    """
    Estimates MIDI note number from provided frequency
    :param frequency: The frequency
    :return: The midi note number (or NaN)
    """
    return 12 * np.log2(frequency / 440) + 69
    

def spectral_centroid(magnitude_spectrum: torch.Tensor, magnitude_freqs: torch.Tensor):
    """
    Calculates the spectral centroid from provided magnitude spectrum
    :param magnitude_spectrum: The magnitude spectrum
    :param magnitude_freqs: The magnitude frequencies
    :return: The spectral centroid
    Reference: Eyben, pp. 39-40
    """
    centroid = torch.zeros(magnitude_spectrum.shape[0], magnitude_spectrum.shape[-1])
    for i in range(magnitude_spectrum.shape[0]):
        for j in range(magnitude_spectrum.shape[-1]):
            centroid[i, j] = torch.sum(torch.multiply(magnitude_spectrum[i, :, j], magnitude_freqs)) / torch.sum(magnitude_spectrum[i, :, j])
    return centroid


def spectral_entropy(power_spectrum: torch.Tensor):
    """
    Calculates the spectral entropy from provided magnitude spectrum
    :param power_spectrum: The power spectrum
    :return: The spectral entropy
    Reference: Eyben, pp. 23, 40, 41
    """
    entropy = torch.zeros(power_spectrum.shape[0], power_spectrum.shape[-1])
    for i in range(power_spectrum.shape[0]):
        for j in range(power_spectrum.shape[-1]):        
            spectrum_pmf = power_spectrum[i, :, j] / torch.sum(power_spectrum)
            local_entropy = 0
            for i in range(spectrum_pmf.size):
                local_entropy += spectrum_pmf[i] * torch.log2(spectrum_pmf[i])
            entropy[i, j] = -local_entropy
    return entropy


def spectral_flatness(magnitude_spectrum: torch.Tensor):
    """
    Calculates the spectral flatness from provided magnitude spectrum
    :param magnitude_spectrum: The magnitude spectrum
    :return: The spectral flatness, in dBFS
    Reference: Eyben, p. 39, https://en.wikipedia.org/wiki/Spectral_flatness
    """
    flatness = torch.zeros(magnitude_spectrum.shape[0], magnitude_spectrum.shape[-1])
    for i in range(magnitude_spectrum.shape[0]):
        for j in range(magnitude_spectrum.shape[-1]):        
            flatness[i, j] = 20 * torch.log10(torch.exp(torch.sum(
                    torch.log(magnitude_spectrum[i, :, j])) / magnitude_spectrum.size(1)) / \
                    (torch.sum(magnitude_spectrum[i, :, j]) / magnitude_spectrum.size(1))
                )
    return flatness


def spectral_moments(audio, magnitude_freqs: torch.Tensor):
    """
    Calculates the spectral moments from provided magnitude spectrum
    :param magnitude_spectrum: The magnitude spectrum
    :param magnitude_freqs: The magnitude frequencies
    :param centroid: The spectral centroid
    :return: The spectral moments
    Reference: Eyben, pp. 23, 39-40
    """
    # This must be a power spectrum!
    # power_spectrum = np.square(audio["magnitude_spectrum"])
    power_spectrum = audio["power_spectrum"]
    spectral_variance = torch.zeros(power_spectrum.shape[0], power_spectrum.shape[-1])
    spectral_skewness = torch.zeros(power_spectrum.shape[0], power_spectrum.shape[-1])
    spectral_kurtosis = torch.zeros(power_spectrum.shape[0], power_spectrum.shape[-1])

    for i in range(power_spectrum.shape[0]):
        for j in range(power_spectrum.shape[-1]):        
            spectrum_pmf = power_spectrum[i, :, j] / torch.sum(power_spectrum)    
            spectral_variance = 0
            spectral_skewness = 0
            spectral_kurtosis = 0
            spectral_variance_vector = magnitude_freqs - audio["spectral_centroid"]
            spectral_variance_vector = torch.square(spectral_variance_vector)
            spectral_variance_vector = torch.multiply(spectral_variance_vector, spectrum_pmf)
            spectral_variance[i, j] = torch.sum(spectral_variance_vector)
            spectral_skewness_vector = magnitude_freqs - audio["spectral_centroid"]
            spectral_skewness_vector = torch.pow(spectral_skewness_vector, 3)
            spectral_skewness_vector = torch.multiply(spectral_skewness_vector, spectrum_pmf)
            spectral_skewness[i, j] = torch.sum(spectral_skewness_vector)
            spectral_kurtosis_vector = magnitude_freqs - audio["spectral_centroid"]
            spectral_kurtosis_vector = torch.pow(spectral_kurtosis_vector, 4)
            spectral_kurtosis_vector = torch.multiply(spectral_kurtosis_vector, spectrum_pmf)
            spectral_kurtosis[i, j] = torch.sum(spectral_kurtosis_vector)

    spectral_skewness /= torch.float_power(spectral_variance, 3/2)
    spectral_kurtosis /= torch.pow(spectral_variance, 2)

    return {"spectral_variance": spectral_variance, "spectral_skewness": spectral_skewness, "spectral_kurtosis": spectral_kurtosis}


def spectral_roll_off_point(power_spectrum: torch.Tensor, magnitude_freqs: torch.Tensor, n):
    """
    Calculates the spectral slope from provided power spectrum
    :param power_spectrum: The power spectrum
    :param magnitude_freqs: The magnitude frequencies
    :param n: The roll-off, as a fraction (0 <= n <= 1.00)
    :return: The roll-off frequency
    Reference: Eyben, p. 41
    """
    energy = np.sum(power_spectrum)
    i = -1
    cumulative_energy = 0
    while cumulative_energy < n and i < magnitude_freqs.size - 1:
        i += 1
        cumulative_energy += power_spectrum[i] / energy
    return magnitude_freqs[i]


def spectral_slope(magnitude_spectrum: torch.Tensor, magnitude_freqs: torch.Tensor):
    """
    Calculates the spectral slope from provided magnitude spectrum
    :param magnitude_spectrum: The magnitude spectrum
    :param magnitude_freqs: The magnitude frequencies
    :return: The slope and y-intercept
    Reference: Eyben, pp. 35-38
    """
    slope = sklearn.linear_model.LinearRegression().fit(np.reshape(magnitude_spectrum, (magnitude_spectrum.shape[-1], 1)), magnitude_freqs)
    return slope.coef_[-1], slope.intercept_


def zero_crossing_rate(audio, sample_rate):
    """
    Extracts the zero-crossing rate
    :param audio: A NumPy array of audio samples
    :param sample_rate: The sample rate of the audio
    :return: The zero-crossing rate
    Reference: Eyben, p. 20
    """
    num_zc = 0
    N = audio.shape[-1]
    for n in range(1, N):
        if audio[n-1] * audio[n] < 0:
            num_zc += 1
        elif n < N-1 and audio[n-1] * audio[n+1] < 0 and audio[n] == 0:
            num_zc += 1
    return num_zc * sample_rate / N
