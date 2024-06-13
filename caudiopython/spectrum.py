"""
File: spectrum.py
Author: Jeff Martin
Date: 2/11/23

This file contains functionality for spectral analysis.
"""

import cython
import numpy as np
from typing import Tuple


@cython.cfunc
def fft_data_decompose(fft_data: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Decomposes FFT data from a Numpy array into arrays of amplitudes and phases.
    This function can handle Numpy arrays of any dimension.
    :param fft_data: The data from a FFT function
    :return: Two arrays: one for amplitudes and one for phases
    """
    amps = np.abs(fft_data)
    phases = np.angle(fft_data)
    return amps, phases


@cython.cfunc
def fft_data_recompose(amps: np.ndarray, phases: np.ndarray) -> np.ndarray:
    """
    Recomposes FFT data from arrays of amplitudes and phases
    This function can handle Numpy arrays of any dimension.
    :param amps: An array of amplitudes
    :param phases: An array of phases
    :return: An array of FFT data
    """
    real = np.multiply(np.cos(phases), amps)
    imag = np.multiply(np.sin(phases), amps)
    return real + (imag * 1j)
