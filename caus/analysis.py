"""
File: analysis.py
Author: Jeff Martin
Date: 12/17/23

Audio analysis tools developed from Eyben, "Real-Time Speech and Music Classification"
These tools expect audio as a 1D array of samples.
"""

import cython
import numpy as np
import scipy.fft
import scipy.signal


def analyzer(audio: dict, fft_size: int) -> dict:
    """
    Runs a suite of analysis tools on a provided audio dictionary.
    The analysis is done in-place.
    :param audio: An audio dictionary
    """
    audio["spectral_centroid"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["spectral_variance"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["spectral_skewness"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["spectral_kurtosis"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["spectral_entropy"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["spectral_flatness"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["spectral_roll_off_0.5"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["spectral_roll_off_0.75"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["spectral_roll_off_0.9"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["spectral_roll_off_0.95"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["spectral_slope"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["spectral_slope_0:1kHz"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["spectral_slope_1:5kHz"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["spectral_slope_0:5kHz"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))
    audio["zero_crossing_rate"] = np.zeros((audio["channels"], audio["num_spectrogram_frames"]))

    i: cython.int
    j: cython.int

    for i in range(audio["channels"]):
        for j in range(audio["num_spectrogram_frames"]):
            magnitude_spectrum = audio["magnitude_spectrogram"][i, :, j]
            magnitude_spectrum_sum = np.sum(magnitude_spectrum)
            power_spectrum = audio["power_spectrogram"][i, :, j]
            power_spectrum_sum = np.sum(power_spectrum)
            spectrum_pmf = power_spectrum / power_spectrum_sum
            rfftfreqs = scipy.fft.rfftfreq(fft_size, 1/audio["sample_rate"])
            audio['spectral_centroid'][i, j] = spectral_centroid(magnitude_spectrum, rfftfreqs, magnitude_spectrum_sum)
            audio['spectral_variance'][i, j] = spectral_variance(spectrum_pmf, rfftfreqs, audio['spectral_centroid'][i, j])
            audio['spectral_skewness'][i, j] = spectral_skewness(spectrum_pmf, rfftfreqs, audio['spectral_centroid'][i, j], audio['spectral_variance'][i, j])
            audio['spectral_kurtosis'][i, j] = spectral_kurtosis(spectrum_pmf, rfftfreqs, audio['spectral_centroid'][i, j], audio['spectral_variance'][i, j])
            audio['spectral_entropy'][i, j] = spectral_entropy(power_spectrum)
            audio['spectral_flatness'][i, j] = spectral_flatness(magnitude_spectrum, magnitude_spectrum_sum)
            audio['spectral_roll_off_0.5'][i, j] = spectral_roll_off_point(power_spectrum, rfftfreqs, 0.5, power_spectrum_sum)
            audio['spectral_roll_off_0.75'][i, j] = spectral_roll_off_point(power_spectrum, rfftfreqs, 0.75, power_spectrum_sum)
            audio['spectral_roll_off_0.9'][i, j] = spectral_roll_off_point(power_spectrum, rfftfreqs, 0.9, power_spectrum_sum)
            audio['spectral_roll_off_0.95'][i, j] = spectral_roll_off_point(power_spectrum, rfftfreqs, 0.95, power_spectrum_sum)
            audio['spectral_slope'][i, j] = spectral_slope(power_spectrum)
            audio['spectral_slope_0:1kHz'][i, j] = spectral_slope_region(power_spectrum, rfftfreqs, 0, 1000, audio["sample_rate"])
            audio['spectral_slope_1:5kHz'][i, j] = spectral_slope_region(power_spectrum, rfftfreqs, 1000, 5000, audio["sample_rate"])
            audio['spectral_slope_0:5kHz'][i, j] = spectral_slope_region(power_spectrum, rfftfreqs, 0, 5000, audio["sample_rate"])

    np.nan_to_num(audio["spectral_centroid"], False)
    np.nan_to_num(audio["spectral_variance"], False)
    np.nan_to_num(audio["spectral_skewness"], False)
    np.nan_to_num(audio["spectral_kurtosis"], False)
    np.nan_to_num(audio["spectral_entropy"], False)
    np.nan_to_num(audio["spectral_flatness"], False)
    np.nan_to_num(audio["spectral_roll_off_0.5"], False)
    np.nan_to_num(audio["spectral_roll_off_0.75"], False)
    np.nan_to_num(audio["spectral_roll_off_0.9"], False)
    np.nan_to_num(audio["spectral_roll_off_0.95"], False)
    np.nan_to_num(audio["spectral_slope"], False)
    np.nan_to_num(audio["spectral_slope_0:1kHz"], False)
    np.nan_to_num(audio["spectral_slope_1:5kHz"], False)
    np.nan_to_num(audio["spectral_slope_0:5kHz"], False)


def energy(audio: np.ndarray) -> cython.double:
    """
    Extracts the RMS energy of the signal
    :param audio: A NumPy array of audio samples
    :return: The RMS energy of the signal
    Reference: Eyben, pp. 21-22
    """
    energy = np.sqrt((1 / audio.shape[-1]) * np.sum(np.square(audio)))
    if np.isnan(energy) or np.isneginf(energy) or np.isinf(energy):
        energy = 0.0
    return energy


def spectral_centroid(magnitude_spectrum: np.ndarray, magnitude_freqs: np.ndarray, magnitude_spectrum_sum) -> cython.double:
    """
    Calculates the spectral centroid from provided magnitude spectrum
    :param magnitude_spectrum: The magnitude spectrum
    :param magnitude_freqs: The magnitude frequencies
    :param magnitude_spectrum_sum: The sum of the magnitude spectrum
    :return: The spectral centroid
    Reference: Eyben, pp. 39-40
    """
    centroid = np.sum(np.multiply(magnitude_spectrum, magnitude_freqs)) / magnitude_spectrum_sum
    if np.isnan(centroid) or np.isneginf(centroid) or np.isinf(centroid):
        centroid = 0.0
    return centroid


def spectral_entropy(spectrum_pmf: np.ndarray) -> cython.double:
    """
    Calculates the spectral entropy from provided power spectrum
    :param spectrum_pmf: The spectrum power mass function PMF
    :return: The spectral entropy
    Reference: Eyben, pp. 23, 40, 41
    """
    entropy = 0
    spec_mul = spectrum_pmf * np.log2(spectrum_pmf)
    entropy = -np.sum(spec_mul)
    if np.isnan(entropy) or np.isneginf(entropy) or np.isinf(entropy):
        entropy = 0.0
    return entropy


def spectral_flatness(magnitude_spectrum: np.ndarray, magnitude_spectrum_sum) -> cython.double:
    """
    Calculates the spectral flatness from provided magnitude spectrum
    :param magnitude_spectrum: The magnitude spectrum
    :param magnitude_spectrum_sum: The sum of the magnitude spectrum
    :return: The spectral flatness, in dBFS
    Reference: Eyben, p. 39, https://en.wikipedia.org/wiki/Spectral_flatness
    """
    flatness = np.exp(np.sum(np.log(magnitude_spectrum)) / magnitude_spectrum.size) / (magnitude_spectrum_sum / magnitude_spectrum.size)
    flatness = 20 * np.log10(flatness)
    if np.isnan(flatness) or np.isneginf(flatness) or np.isinf(flatness):
        flatness = 0.0
    return flatness


def spectral_variance(spectrum_pmf: np.ndarray, magnitude_freqs: np.ndarray, spectral_centroid: cython.double) -> cython.double:
    """
    Calculates the spectral variance
    :param spectrum_pmf: The spectrum power mass function PMF
    :param magnitude_freqs: The magnitude frequencies
    :param spectral_centroid: The spectral centroid
    :return: The spectral variance
    Reference: Eyben, pp. 23, 39-40
    """
    spectral_variance_arr = np.square(magnitude_freqs - spectral_centroid) * spectrum_pmf
    spectral_variance = np.sum(spectral_variance_arr)
    if np.isnan(spectral_variance) or np.isneginf(spectral_variance) or np.isinf(spectral_variance):
        spectral_variance = 0.0
    return spectral_variance


def spectral_skewness(spectrum_pmf: np.ndarray, magnitude_freqs: np.ndarray, spectral_centroid: cython.double, spectral_variance: cython.double) -> cython.double:
    """
    Calculates the spectral skewness
    :param spectrum_pmf: The spectrum power mass function PMF
    :param magnitude_freqs: The magnitude frequencies
    :param spectral_centroid: The spectral centroid
    :param spectral_variance: The spectral variance
    :return: The spectral skewness
    Reference: Eyben, pp. 23, 39-40
    """
    spectral_skewness_arr = np.power(magnitude_freqs - spectral_centroid, 3) * spectrum_pmf
    spectral_skewness = np.sum(spectral_skewness_arr) / np.float_power(spectral_variance, 3/2)
    if np.isnan(spectral_skewness) or np.isneginf(spectral_skewness) or np.isinf(spectral_skewness):
        spectral_skewness = 0.0
    return spectral_skewness


def spectral_kurtosis(spectrum_pmf: np.ndarray, magnitude_freqs: np.ndarray, spectral_centroid: cython.double, spectral_variance: cython.double) -> cython.double:
    """
    Calculates the spectral kurtosis
    :param spectrum_pmf: The spectrum power mass function PMF
    :param magnitude_freqs: The magnitude frequencies
    :param spectral_centroid: The spectral centroid
    :param spectral_variance: The spectral variance
    :return: The spectral kurtosis
    Reference: Eyben, pp. 23, 39-40
    """
    spectral_kurtosis_arr = np.power(magnitude_freqs - spectral_centroid, 4) * spectrum_pmf
    spectral_kurtosis = np.sum(spectral_kurtosis_arr) / np.power(spectral_variance, 2)
    if np.isnan(spectral_kurtosis) or np.isneginf(spectral_kurtosis) or np.isinf(spectral_kurtosis):
        spectral_kurtosis = 0.0
    return spectral_kurtosis


def spectral_roll_off_point(power_spectrum: np.ndarray, magnitude_freqs: np.ndarray, n: cython.double, power_spectrum_sum) -> cython.double:
    """
    Calculates the spectral roll off frequency from provided power spectrum
    :param power_spectrum: The power spectrum
    :param magnitude_freqs: The magnitude frequencies
    :param n: The roll-off, as a fraction (0 <= n <= 1.00)
    :param power_spectrum_sum: The sum of the power spectrum
    :return: The roll-off frequency
    Reference: Eyben, p. 41
    """
    i: cython.int
    cumulative_energy: cython.double
    i = -1
    cumulative_energy = 0.0
    while cumulative_energy < n and i < magnitude_freqs.size - 1:
        i += 1
        cumulative_energy += power_spectrum[i] / power_spectrum_sum
    roll_off = magnitude_freqs[i]
    if np.isnan(roll_off) or np.isneginf(roll_off) or np.isinf(roll_off):
        roll_off = 0.0
    return roll_off


def spectral_slope(power_spectrum: np.ndarray) -> cython.double:
    """
    Calculates the spectral slope from provided power spectrum.
    :param power_spectrum: The power spectrum
    :return: The slope
    Reference: Eyben, pp. 35-38
    """
    N = power_spectrum.size
    X = np.arange(0, N, 1)
    sum_x = N * (N - 1) / 2
    sum_x_2 = N * (N - 1) * (2 * N - 1) / 6
    slope = (N * np.dot(power_spectrum, X) - sum_x * np.sum(power_spectrum)) / (N * sum_x_2 - sum_x ** 2)
    return slope


def spectral_slope_region(power_spectrum: np.ndarray, rfftfreqs: np.ndarray, f_lower: cython.double, f_upper: cython.double, sample_rate: cython.int) -> cython.double:
    """
    Calculates the spectral slope from provided power spectrum, between the frequencies
    specified. The frequencies specified do not have to correspond to exact bin indices.
    :param power_spectrum: The power spectrum
    :param rfftfreqs: The FFT freqs for the power spectrum bins
    :param f_lower: The lower frequency
    :param f_upper: The upper frequency
    :param sample_rate: The sample rate of the audio
    :return: The slope
    Reference: Eyben, pp. 35-38
    """
    # the fundamental frequency
    f_0 = sample_rate / ((power_spectrum.size - 1) * 2)

    # approximate bin indices for lower and upper frequencies
    m_fl = f_lower / f_0
    m_fu = f_upper / f_0

    N = power_spectrum.size

    m_fl_ceil = int(np.ceil(m_fl))
    m_fl_floor = int(np.floor(m_fl))
    m_fu_ceil = int(np.ceil(m_fu))
    m_fu_floor = int(np.floor(m_fu))
    
    # these complicated formulas come from Eyben, p.37. The idea
    # is to use interpolation in case the lower and upper frequencies
    # do not correspond to exact bin indices.
    sum_x = f_lower + np.sum(rfftfreqs[m_fl_ceil:m_fu_floor]) + f_upper
    sum_y = power_spectrum[m_fl_floor] + (m_fl - m_fl_floor) * \
        (power_spectrum[m_fl_ceil] - power_spectrum[m_fl_floor]) + \
        np.sum(power_spectrum[m_fl_ceil:m_fu_floor]) + \
        power_spectrum[m_fu_floor] + (m_fu - m_fu_floor) * (power_spectrum[m_fu_ceil] - power_spectrum[m_fu_floor])
    sum_x_2 = f_lower ** 2 + np.sum(np.square(rfftfreqs[m_fl_ceil:m_fu_floor])) + f_upper ** 2
    sum_xy = f_lower * (power_spectrum[m_fl_floor] + (m_fl - m_fl_floor) * (power_spectrum[m_fl_ceil] - power_spectrum[m_fl_floor])) + \
        np.sum(np.multiply(power_spectrum[m_fl_ceil:m_fu_floor], rfftfreqs[m_fl_ceil:m_fu_floor])) + \
        f_upper * (power_spectrum[m_fu_floor] + (m_fu - m_fu_floor) * (power_spectrum[m_fu_ceil] - power_spectrum[m_fu_floor]))

    slope = (N * sum_xy - sum_x * sum_y) / (N * sum_x_2 - sum_x ** 2)
    return slope


def zero_crossing_rate(audio: np.ndarray, sample_rate: cython.int) -> cython.double:
    """
    Extracts the zero-crossing rate
    :param audio: A NumPy array of audio samples
    :param sample_rate: The sample rate of the audio
    :return: The zero-crossing rate
    Reference: Eyben, p. 20
    """
    n: cython.int
    num_zc: cython.int
    num_zc = 0
    N = audio.shape[-1]
    for n in range(1, N):
        if audio[n-1] * audio[n] < 0:
            num_zc += 1
        elif n < N-1 and audio[n-1] * audio[n+1] < 0 and audio[n] == 0:
            num_zc += 1
    zcr = num_zc * sample_rate / N
    if np.isnan(zcr) or np.isneginf(zcr) or np.isinf(zcr):
        zcr = 0.0
    return zcr
