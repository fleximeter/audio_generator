"""
File: synthesis.py
Author: Jeff Martin
Date: 7/6/24

This file has audio synthesis functionality.
"""

import cython
import numpy as np



def saw(freq: cython.double, max_harmonic: cython.int, len: cython.int, sample_rate: cython.int = 44100):
    """
    Generates a sawtooth tone
    :param freq: The frequency
    :param max_harmonic: The maximum harmonic index
    :param len: The length of the signal
    :param sample_rate: The audio sample rate
    :return: The sawtooth signal
    """
    sig = np.zeros((len))
    harmonic: cython.int
    for harmonic in range(1, max_harmonic + 1):
        step = 2 * np.pi * freq * harmonic / sample_rate
        stop = len * step
        harmonic_sig = np.sin(np.arange(0, stop, step))
        if harmonic_sig.shape[-1] > len:
            harmonic_sig = harmonic_sig[:len]
        elif harmonic_sig.shape[-1] < len:
            zeros = np.zeros((len - harmonic_sig.shape[-1]))
            harmonic_sig = np.hstack((harmonic_sig, zeros))
        sig += 1 / (2 * harmonic) * harmonic_sig
    return sig



def sine(freq: cython.double, phase: cython.double, len: cython.int, sample_rate: cython.int = 44100):
    """
    Generates a sine tone
    :param freq: The frequency
    :param phase: The phase
    :param len: The length of the signal
    :param sample_rate: The audio sample rate
    :return: The sine signal
    """
    step = 2 * np.pi * freq / sample_rate
    stop = len * step + phase
    sig = np.sin(np.arange(phase, stop, step))
    if sig.shape[-1] > len:
        sig = sig[:len]
    elif sig.shape[-1] < len:
        zeros = np.zeros((len - sig.shape[-1]))
        sig = np.hstack((sig, zeros))
    return sig



def square(freq: cython.double, max_harmonic: cython.int, len: cython.int, sample_rate: cython.int = 44100):
    """
    Generates a square tone
    :param freq: The frequency
    :param max_harmonic: The maximum harmonic index
    :param len: The length of the signal
    :param sample_rate: The audio sample rate
    :return: The square signal
    """
    max_harmonic = (max_harmonic - 1) // 2
    sig = np.zeros((len))
    harmonic: cython.int
    for harmonic in range(0, max_harmonic + 1):
        step = 2 * np.pi * freq * (2 * harmonic + 1) / sample_rate
        stop = len * step
        harmonic_sig = np.sin(np.arange(0, stop, step))
        if harmonic_sig.shape[-1] > len:
            harmonic_sig = harmonic_sig[:len]
        elif harmonic_sig.shape[-1] < len:
            zeros = np.zeros((len - harmonic_sig.shape[-1]))
            harmonic_sig = np.hstack((harmonic_sig, zeros))
        sig += 1 / (2 * harmonic + 1) * harmonic_sig
    return sig



def triangle(freq: cython.double, max_harmonic: cython.int, len: cython.int, sample_rate: cython.int = 44100):
    """
    Generates a triangle tone
    :param freq: The frequency
    :param max_harmonic: The maximum harmonic index
    :param len: The length of the signal
    :param sample_rate: The audio sample rate
    :return: The triangle signal
    """
    max_harmonic = (max_harmonic - 1) // 2
    sig = np.zeros((len))
    harmonic: cython.int
    for harmonic in range(0, max_harmonic + 1):
        step = 2 * np.pi * freq * (2 * harmonic + 1) / sample_rate
        stop = len * step
        harmonic_sig = np.sin(np.arange(0, stop, step))
        if harmonic_sig.shape[-1] > len:
            harmonic_sig = harmonic_sig[:len]
        elif harmonic_sig.shape[-1] < len:
            zeros = np.zeros((len - harmonic_sig.shape[-1]))
            harmonic_sig = np.hstack((harmonic_sig, zeros))
        sig += (-1) ** harmonic / (2 * harmonic + 1) ** 2 * harmonic_sig
    sig = sig * 8 / np.pi ** 2
    return sig
