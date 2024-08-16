"""
File: operations.py
Author: Jeff Martin
Date: 12/2/23

This file allows you to perform operations on audio and FFT data.
"""

import cython
import numpy as np
import random

np.seterr(divide="ignore")
_rng = random.Random()


@cython.cfunc
def adjust_level(audio: np.ndarray, max_level: cython.double):
    """
    Adjusts the level of audio to a specified dB level
    :param audio: The audio samples as a NumPy array
    :param max_level: The max level for the audio
    :return: The scaled audio
    """
    current_rms = np.sqrt(np.average(np.square(audio), axis=audio.ndim-1))[0]
    target_rms = 10 ** (max_level / 20)
    return audio * (target_rms / current_rms)


@cython.cfunc
def calculate_dc_bias(audio: np.ndarray) -> cython.double:
    """
    Calculates DC bias of an audio signal
    :param audio: The audio signal
    :return: The DC bias
    """
    return np.average(audio, axis=audio.ndim-1)


@cython.cfunc
def dbfs_audio(audio: np.ndarray):
    """
    Calculates dbfs (decibels full scale) for a chunk of audio. This function will use the RMS method, 
    and assumes that the audio is in float format where 1 is the highest possible peak.
    :param audio: The audio to calculate dbfs for
    :return: A float value representing the dbfs
    """
    try:
        rms = np.sqrt(np.average(np.square(audio), axis=audio.ndim-1))
        return 20 * np.log10(np.abs(rms))
    except RuntimeWarning:
        return -np.inf


@cython.cfunc
def dbfs_max_local(audio: np.ndarray, chunk_size: cython.int = 10, hop_size: cython.int = 5):
    """
    Checks the maximum local dbfs (decibels full scale) of an audio file
    :param audio: The audio
    :param chunk_size: The chunk size to check
    :param hop_size: The number of frames to hop from chunk center to chunk center
    :return: The max local dbfs
    """
    i: cython.int
    dbfs = -np.inf
    for i in range(0, audio.size, hop_size):
        end = min(i + chunk_size, audio.size - 1)
        try:
            if chunk_size > 1:
                rms = np.sqrt(np.average(np.square(audio[i:end]), -1))
                dbfs = max(20 * np.log10(np.abs(rms)), dbfs)
            else:
                dbfs = max(20 * np.log10(np.abs(audio[i])), dbfs)
        except RuntimeWarning:
            pass
    return dbfs


@cython.cfunc
def dbfs_min_local(audio: np.ndarray, chunk_size: cython.int = 10, hop_size: cython.int = 5):
    """
    Checks the minimum local dbfs (decibels full scale) of an audio file
    :param audio: The audio
    :param chunk_size: The chunk size to check
    :param hop_size: The number of frames to hop from chunk center to chunk center
    :return: The min local dbfs
    """
    i: cython.int
    dbfs = 0
    for i in range(0, len(audio), hop_size):
        end = min(i + chunk_size, len(audio) - 1)
        try:
            rms = np.sqrt(np.average(np.square(audio[i:end]), -1))
            dbfs = min(20 * np.log10(np.abs(rms)), dbfs)
        except RuntimeWarning:
            pass
    return dbfs


@cython.cfunc
def dbfs_sample(sample):
    """
    Calculates dbfs (decibels full scale) for an audio sample. This function assumes that the 
    audio is in float format where 1 is the highest possible peak.
    :param sample: The sample to calculate dbfs for
    :return: A float value representing the dbfs
    """
    return 20 * np.log10(np.abs(sample))


@cython.cfunc
def fade_in(audio: np.ndarray, envelope="hanning", duration: cython.int = 100):
    """
    Implements a fade-in on an array of audio samples.
    :param audio: The array of audio samples (may have multiple channels; the fade-in will be applied to all channels)
    :param envelope: The shape of the fade-in envelope. Must be a NumPy envelope. The envelope will be divided in half, and only the first half will be used.
    :param duration: The duration (in frames) of the fade-in envelope half. If the duration is longer than the audio, it will be truncated.
    :return: The audio with a fade in applied.
    """
    duration = min(duration, audio.shape[-1])
        
    if envelope == "bartlett":
        envelope = np.bartlett(duration * 2)[:duration]
    elif envelope == "blackman":
        envelope = np.blackman(duration * 2)[:duration]
    elif envelope == "hanning":
        envelope = np.hanning(duration * 2)[:duration]
    elif envelope == "hamming":
        envelope = np.hamming(duration * 2)[:duration]
    else:
        envelope = np.ones((duration * 2))[:duration]
    envelope = np.hstack((envelope, np.ones((audio.shape[-1] - envelope.shape[-1]))))
    
    return audio * envelope
    

@cython.cfunc
def fade_out(audio: np.ndarray, envelope="hanning", duration: cython.int = 100):
    """
    Implements a fade-out on an array of audio samples.
    :param audio: The array of audio samples (may have multiple channels; the fade-out will be applied to all channels)
    :param envelope: The shape of the fade-out envelope. Must be a NumPy envelope. The envelope will be divided in half, and only the second half will be used.
    :param duration: The duration (in frames) of the fade-out envelope half. If the duration is longer than the audio, it will be truncated.
    :return: The audio with a fade-out applied.
    """
    duration = min(duration, audio.shape[-1])
        
    if envelope == "bartlett":
        envelope = np.bartlett(duration * 2)[duration:]
    elif envelope == "blackman":
        envelope = np.blackman(duration * 2)[duration:]
    elif envelope == "hanning":
        envelope = np.hanning(duration * 2)[duration:]
    elif envelope == "hamming":
        envelope = np.hamming(duration * 2)[duration:]
    else:
        envelope = np.ones((duration * 2))[duration:]
    envelope = np.hstack((np.ones((audio.shape[-1] - envelope.shape[-1])), envelope))
    
    return audio * envelope


@cython.cfunc
def force_equal_energy(audio: np.ndarray, dbfs: cython.double = -6.0, window_size: cython.int = 8192):
    """
    Forces equal energy on a mono signal over time. For example, if a signal initially has high energy, 
    and gets less energetic, this will adjust the energy level so that it does not decrease.
    Better results come with using a larger window size, so the energy changes more gradually.
    :param audio: The array of audio samples
    :param dbfs: The target level of the entire signal, in dbfs
    :param window_size: The window size to consider when detecting RMS energy
    :return: An adjusted version of the signal
    """
    i: cython.int
    j: cython.int
    idx: cython.int
    frame_idx: cython.int
    while audio.ndim > 1:
        audio = audio.sum(-2)
    audio_new = np.empty(audio.shape)  # the new array we'll be returning
    level_float = 10 ** (dbfs / 20)  # the target level, in float rather than dbfs
    num_frames = int(np.ceil(audio.shape[-1] / window_size))  # the number of frames that we'll be analyzing
    energy_levels = np.empty((num_frames + 2))  # the energy level for each frame
    
    # find the energy levels
    idx = 1
    for i in range(0, audio.shape[-1], window_size):
        energy_levels[idx] = np.sqrt(np.average(np.square(audio[i:i+window_size])))
        idx += 1
    energy_levels[0] = energy_levels[1]
    energy_levels[-1] = energy_levels[-2]

    # do the first half frame
    for j in range(0, window_size // 2):
        audio_new[j] = audio[j] * level_float / energy_levels[0]
    
    # do adjacent half frames from 1 and 2, 2 and 3, etc.
    frame_idx = 1
    for i in range(window_size // 2, audio.shape[-1], window_size):
        coef = (energy_levels[frame_idx + 1] - energy_levels[frame_idx]) / window_size
        for j in range(i, min(i + window_size, audio.shape[-1])):
            f = coef * (j - i) + energy_levels[frame_idx]
            g = 1/f
            audio_new[j] = audio[j] * g
        frame_idx += 1

    audio_max = np.max(audio_new)
    return audio_new * level_float / audio_max
    

@cython.cfunc
def leak_dc_bias_averager(audio: np.ndarray):
    """
    Leaks DC bias of an audio signal
    :param audio: The audio signal
    :return: The bias-free signal
    """
    if audio.ndim > 1:
        avg = np.average(audio, axis=audio.ndim-1)
        avg = np.reshape(avg, (avg.shape[0], 1))
        return audio - np.repeat(avg, audio.shape[-1], audio.ndim-1)
    else:
        return audio - np.average(audio, axis=audio.ndim-1)


@cython.cfunc
def leak_dc_bias_filter(audio: np.ndarray):
    """
    Leaks DC bias of an audio signal using a highpass filter, described on pp. 762-763
    of "Understanding Digital Signal Processing," 3rd edition, by Richard G. Lyons
    :param audio: The audio signal
    :return: The bias-free signal
    """
    ALPHA = 0.95
    i: cython.int
    j: cython.int
    new_signal = np.zeros(audio.shape)
    if audio.ndim == 1:
        delay_register = 0
        for i in range(audio.shape[-1]):
            combined_signal = audio[i] + ALPHA * delay_register
            new_signal[i] = combined_signal - delay_register
            delay_register = combined_signal
    elif audio.ndim == 2:
        for j in range(audio.shape[-2]):
            delay_register = 0
            for i in range(audio.shape[-1]):
                combined_signal = audio[j, i] + ALPHA * delay_register
                new_signal[j, i] = combined_signal - delay_register
                delay_register = combined_signal
    return new_signal


@cython.cfunc
def cpsmidi(freq: cython.double) -> cython.double:
    """
    Calculates the MIDI note of a provided frequency
    :param midi_note: The frequency in Hz
    :return: The MIDI note
    """
    midi = np.log2(freq / 440) * 12 + 69
    if np.isnan(midi) or np.isneginf(midi) or np.isinf(midi):
        midi = 0.0
    return midi


@cython.cfunc
def midicps(midi_note: cython.double) -> cython.double:
    """
    Calculates the frequency of a specified midi note
    :param midi_note: The MIDI note
    :return: The frequency in Hz
    """
    cps = 440 * 2 ** ((midi_note - 69) / 12)
    if np.isnan(cps) or np.isneginf(cps) or np.isinf(cps):
        cps = 0.0
    return cps


@cython.cfunc
def midiratio(interval: cython.double) -> cython.double:
    """
    Calculates the MIDI ratio of a specified midi interval
    :param midi_note: The MIDI interval in half steps
    :return: The ratio
    """
    ratio = 2 ** (interval / 12)
    if np.isnan(ratio) or np.isneginf(ratio) or np.isinf(ratio):
        ratio = 0.0
    return ratio


@cython.cfunc
def mixdown(audio: np.ndarray):
    """
    Mixes a multichannel signal to a mono signal. 
    :param audio: The audio to mix if it isn't mono
    :return: The mixed audio
    """
    mix = np.sum(audio, 0)
    mix = np.reshape(mix, (1, mix.size))
    mix /= audio.shape[0]
    return mix


@cython.cfunc
def exchanger(data: np.ndarray, hop: cython.int):
    """
    Exchanges samples in an audio file or STFT frames in a spectrum. Each sample (or STFT frame) 
    is swapped with the sample (or STFT frame) *hop* steps ahead or *hop* steps behind. If audio
    is being processed, it should be in the shape (channels, samples). If STFT data is being
    processed, it should be in the shape (channels, frames, bins).
    :param data: The audio (or spectrum) to process
    :param hop: The hop size
    :return: The exchanged audio (or spectrum)
    """
    i: cython.int
    j: cython.int
    k: cython.int
    new_data = np.empty(data.shape, dtype=data.dtype)
    for i in range(data.shape[0]):
        for j in range(0, data.shape[1] - data.shape[1] % (hop * 2), hop * 2):
            for k in range(j, j+hop):
                new_data[i, k] = data[i, k+hop]
                new_data[i, k+hop] = data[i, k]
    return new_data


@cython.cfunc
def stochastic_exchanger(data: np.ndarray, max_hop: cython.int):
    """
    Stochastically exchanges samples in an audio file or STFT frames in a spectrum. Each sample 
    (or STFT frame) is swapped with the sample (or STFT frame) up to *hop* steps ahead or *hop* 
    steps behind. If audio is being processed, it should be in the shape (channels, samples). 
    If STFT data is being processed, it should be in the shape (channels, frames, bins).
    Warning: if you try to run this on sampled audio rather than STFT data, this will take
    a *very* long time!
    :param data: The audio (or spectrum) to process
    :param hop: The hop size
    :return: The exchanged audio (or spectrum)
    """
    new_data = np.empty(data.shape, dtype=data.dtype)
    i: cython.int
    idx: cython.int
    swap_idx: cython.int

    for i in range(data.shape[0]):
        future_indices = set()
        past_indices = set()
        idx = 0
        while len(future_indices) + len(past_indices) < data.shape[1] and idx < data.shape[1]:
            # We can only perform a swap if this index has not been swapped yet
            if idx not in future_indices:
                # Get all possible future indices we might be able to swap with
                possible_indices = {z for z in range(idx, min(idx + max_hop, data.shape[1]))}
                possible_indices = possible_indices - future_indices

                # Randomly choose which index to swap with, and perform the swap
                swap_idx = _rng.choice(tuple(possible_indices))
                new_data[i, idx] = data[i, swap_idx]
                new_data[i, swap_idx] = data[i, idx]
                # print(f"Swap {idx} {swap_idx}")

                # Update the future and past indices
                future_indices.add(swap_idx)
                past_indices.add(idx)
                future_indices -= past_indices
            
            idx += 1

    return new_data
