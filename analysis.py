"""
File: analysis.py
Author: Jeff Martin
Date: 12/17/23

Audio analysis tools developed from Eyben, "Real-Time Speech and Music Classification"
"""

import torch.fft
import sklearn.linear_model


def analyzer(audio):
    """
    Runs a suite of analysis tools on a provided NumPy array of audio samples
    :param audio: An audio dictionary
    """
    rfftfreqs = torch.fft.rfftfreq((audio["magnitude_spectrogram"].shape[-2] - 1) * 2, 1/audio["sample_rate"])
    audio['spectral_centroid'] = spectral_centroid(audio["magnitude_spectrogram"], rfftfreqs)
    audio['spectral_entropy'] = spectral_entropy(audio["power_spectrogram"])
    audio['spectral_flatness'] = spectral_flatness(audio["magnitude_spectrogram"])
    audio['spectral_slope'], audio['spectral_y_int'] = spectral_slope(audio["magnitude_spectrogram"], rfftfreqs)
    audio['spectral_roll_off_0.5'] = spectral_roll_off_point(audio["power_spectrogram"], rfftfreqs, 0.5)
    audio['spectral_roll_off_0.75'] = spectral_roll_off_point(audio["power_spectrogram"], rfftfreqs, 0.75)
    audio['spectral_roll_off_0.9'] = spectral_roll_off_point(audio["power_spectrogram"], rfftfreqs, 0.9)
    audio['spectral_roll_off_0.95'] = spectral_roll_off_point(audio["power_spectrogram"], rfftfreqs, 0.95)
    audio.update(spectral_moments(audio, rfftfreqs))
        

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
            log_product = spectrum_pmf * torch.log2(spectrum_pmf)
            entropy[i, j] = -torch.sum(log_product)
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
    power_spectrum = audio["power_spectrogram"]
    spectral_variance = torch.zeros(power_spectrum.shape[0], power_spectrum.shape[-1])
    spectral_skewness = torch.zeros(power_spectrum.shape[0], power_spectrum.shape[-1])
    spectral_kurtosis = torch.zeros(power_spectrum.shape[0], power_spectrum.shape[-1])

    for i in range(power_spectrum.shape[0]):
        for j in range(power_spectrum.shape[-1]):        
            spectrum_pmf = power_spectrum[i, :, j] / torch.sum(power_spectrum)    
            spectral_variance_vector = magnitude_freqs - audio["spectral_centroid"][i, j]
            spectral_variance_vector = torch.square(spectral_variance_vector)
            spectral_variance_vector = torch.multiply(spectral_variance_vector, spectrum_pmf)
            spectral_variance[i, j] = torch.sum(spectral_variance_vector)
            spectral_skewness_vector = magnitude_freqs - audio["spectral_centroid"][i, j]
            spectral_skewness_vector = torch.pow(spectral_skewness_vector, 3)
            spectral_skewness_vector = torch.multiply(spectral_skewness_vector, spectrum_pmf)
            spectral_skewness[i, j] = torch.sum(spectral_skewness_vector)
            spectral_kurtosis_vector = magnitude_freqs - audio["spectral_centroid"][i, j]
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
    roll_offs = torch.zeros(power_spectrum.shape[0], power_spectrum.shape[-1])

    for i in range(power_spectrum.shape[0]):
        for j in range(power_spectrum.shape[-1]):
            energy = torch.sum(power_spectrum[i, :, j])
            k = -1
            cumulative_energy = 0
            while cumulative_energy < n and k < magnitude_freqs.numel() - 1:
                k += 1
                cumulative_energy += power_spectrum[i, k, j] / energy
            roll_offs[i, j] = magnitude_freqs[k]
    
    return roll_offs


def spectral_slope(magnitude_spectrum: torch.Tensor, magnitude_freqs: torch.Tensor):
    """
    Calculates the spectral slope from provided magnitude spectrum
    :param magnitude_spectrum: The magnitude spectrum
    :param magnitude_freqs: The magnitude frequencies
    :return: The slope and y-intercept
    Reference: Eyben, pp. 35-38
    """
    slopes = torch.zeros(magnitude_spectrum.shape[0], magnitude_spectrum.shape[-1])
    y_ints = torch.zeros(magnitude_spectrum.shape[0], magnitude_spectrum.shape[-1])

    for i in range(magnitude_spectrum.shape[0]):
        for j in range(magnitude_spectrum.shape[-1]):
            slope = sklearn.linear_model.LinearRegression().fit(torch.reshape(magnitude_spectrum[i, :, j], (magnitude_spectrum.shape[-2], 1)), magnitude_freqs)
            slopes[i, j] = torch.tensor(slope.coef_[-1])
            y_ints[i, j] = torch.tensor(slope.intercept_)
    
    return slopes, y_ints
