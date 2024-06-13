"""
File: plot.py
Author: Jeff Martin
Date: 6/12/24

This file contains functionality for plotting.
"""

import scipy.fft
import numpy as np
import matplotlib.pyplot as plt


def plot_audio(audio: np.ndarray):
    """
    Visualizes audio using matplotlib.
    :param audio: A numpy array of audio samples
    """   
    if len(audio.shape) == 1:
        audio = audio.reshape((1,) + audio.shape)
    rows = audio.shape[0]
    fig, axs = plt.subplots(nrows=rows, ncols=1)
    fig.suptitle("WAV File Visualization")
    for i in range(len(rows)):
        axs[i].set_xlabel("Frame Index")
        axs[i].set_ylabel("Amplitude")
        axs[i].set_title(f"Channel {rows + 1}")
        axs[i].plot([i for i in range(audio.shape[-1])], audio[i, :])
    fig.tight_layout()
    plt.show()
    

def plot_spectrogram(audio: np.ndarray, sample_rate: int, window_size: int = 1024):
    """
    Plots FFT data
    :param audio: A 1D array of audio samples
    :param sample_rate: The sample rate of the audio
    :param window_size: The window size that will be analyzed
    """
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.specgram(audio, NFFT=window_size, Fs=sample_rate, noverlap=sample_rate//2)
    ax.set_title(f"Spectrogram")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Frequency (Hz)")
    plt.show()


def plot_spectrum(spectrum, sample_rate, frequency_range=None):
    """
    Plots FFT data. The FFT data should be in original imaginary form.
    It will be converted to a normalized power spectrum in decibels.
    :param spectrum: An imaginary spectrum to plot
    :param sample_rate: The sample rate (for determining frequencies)
    :param frequency_range: If not None, only the frequencies within this range will be plotted.
    """
    fig, ax = plt.subplots(figsize = (10, 5))
    mags = np.abs(spectrum)
    power = np.square(mags)
    power = 20 * np.log10(np.abs(power)/np.max(np.abs(power)))
    freqs = scipy.fft.rfftfreq((spectrum.shape[-1] - 1) * 2, 1/sample_rate)
    if frequency_range is not None:
        new_freqs = []
        new_power_spectrum = []
        for i in range(freqs.shape[-1]):
            if frequency_range[0] <= freqs[i] <= frequency_range[1]:
                new_freqs.append(freqs[i])
                new_power_spectrum.append(power[i])
        ax.plot(new_freqs, new_power_spectrum)
    else:
        ax.plot(freqs, power)
    ax.set_title(f"Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Amplitude (dB)")
    plt.show()
