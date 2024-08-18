"""
File: operations.py
Author: Jeff Martin
Date: 12/2/23

This file allows you to perform operations on audio and FFT data.
"""

import cython
import math
import numpy as np
import random
from fractions import Fraction

np.seterr(divide="ignore")
_rng = random.Random()



def adjust_level(audio: np.ndarray, max_level: float, max_scalar: float = 1e6):
    """
    Adjusts the level of audio to a specified dB level
    :param audio: The audio samples as a NumPy array
    :param max_level: The max level for the audio
    :param max_scalar: The maximum scalar to use when adjusting audio level
    :return: The scaled audio
    """
    current_level = np.max(audio)
    target_level = 10 ** (max_level / 20)
    scalar = target_level / current_level
    scalar = min(scalar, max_scalar)
    scalar = max(scalar, 1/max_scalar)
    target_audio = audio * scalar
    target_audio = np.nan_to_num(target_audio)
    return audio



def calculate_dc_bias(audio: np.ndarray) -> float:
    """
    Calculates DC bias of an audio signal
    :param audio: The audio signal
    :return: The DC bias
    """
    return np.average(audio, axis=audio.ndim-1)



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



def dbfs_max_local(audio: np.ndarray, chunk_size: int = 10, hop_size: int = 5):
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



def dbfs_min_local(audio: np.ndarray, chunk_size: int = 10, hop_size: int = 5):
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



def dbfs(audio):
    """
    Calculates dbfs (decibels full scale) for an audio sequence or sample. 
    This function assumes that the audio is in float format where 1 is the highest possible peak.
    :param audio: The audio array or sample to calculate dbfs for
    :return: A float value representing the dbfs
    """
    if type(audio) == np.ndarray:
        return 20 * np.log10(np.max(np.abs(audio)))
    else:
        return 20 * np.log10(audio)



def fade_in(audio: np.ndarray, envelope="hanning", duration: int = 100):
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
    


def fade_out(audio: np.ndarray, envelope="hanning", duration: int = 100):
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



def force_equal_energy(audio: np.ndarray, dbfs: float = -6.0, window_size: int = 8192, max_scalar: float = 1e6):
    """
    Forces equal energy on a mono signal over time. For example, if a signal initially has high energy, 
    and gets less energetic, this will adjust the energy level so that it does not decrease.
    Better results come with using a larger window size, so the energy changes more gradually.
    :param audio: The array of audio samples
    :param dbfs: The target level of the entire signal, in dbfs
    :param window_size: The window size to consider when detecting RMS energy
    :param max_scalar: The maximum scalar to use in level adjustment. This is necessary
    because some audio may have an extremely low maximum level, and the scalar required
    to bring it up to the right level could result in computation problems. The computed
    level scalar used in this function will be -max_scalar <= level <= max_scalar.
    :return: An adjusted version of the signal
    """
    i: cython.int
    j: cython.int
    k: cython.int
    idx: cython.int
    frame_idx: cython.int
    if audio.ndim == 1:
        raise Exception("The audio array must have two dimensions.")
    num_channels = audio.shape[0]
    output_audio_arr = np.empty(audio.shape)  # the new array we'll be returning
    target_level = 10 ** (dbfs / 20)  # the target level, in float rather than dbfs
    num_frames = int(np.ceil(audio.shape[-1] / window_size))  # the number of frames that we'll be analyzing
    energy_levels = np.empty((num_channels, num_frames + 2))  # the energy level for each frame
    
    # find the energy levels
    for i in range(num_channels):
        idx = 1
        for j in range(0, audio.shape[-1], window_size):
            end_idx = min(j+window_size, audio.shape[-1])
            energy_levels[i, idx] = np.sqrt(np.average(np.square(audio[i, j:end_idx])))
            idx += 1
        energy_levels[i, 0] = energy_levels[i, 1]
        energy_levels[i, -1] = energy_levels[i, -2]

    # do the first half frame
    for i in range(num_channels):
        coef = 1 / energy_levels[i, 0]
        coef = max(1/max_scalar, coef)
        coef = min(max_scalar, coef)
        for j in range(0, window_size // 2):
            output_audio_arr[i, j] = audio[i, j] * coef
    
    # do adjacent half frames from 1 and 2, 2 and 3, etc.
    for i in range(num_channels):
        frame_idx = 1
        for frame_start_idx in range(window_size // 2, audio.shape[-1], window_size):
            end_idx = min(frame_start_idx + window_size, audio.shape[-1])
            frame_size = end_idx - frame_start_idx
            slope = (energy_levels[i, frame_idx + 1] - energy_levels[i, frame_idx]) / frame_size
            y_int = energy_levels[i, frame_idx]
            for sample_idx in range(frame_start_idx, end_idx):
                f_x = slope * (sample_idx - frame_start_idx) + y_int
                scalar = 1/f_x
                scalar = max(1/max_scalar, scalar)
                scalar = min(max_scalar, scalar)
                output_audio_arr[i, sample_idx] = audio[i, sample_idx] * scalar
            frame_idx += 1

    audio_max = np.max(np.abs(output_audio_arr))
    scalar = target_level / audio_max
    scalar = max(1/max_scalar, scalar)
    scalar = min(max_scalar, scalar)
    return output_audio_arr * scalar
    


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



def cpsmidi(freq: float) -> float:
    """
    Calculates the MIDI note of a provided frequency
    :param midi_note: The frequency in Hz
    :return: The MIDI note
    """
    midi = np.log2(freq / 440) * 12 + 69
    if np.isnan(midi) or np.isneginf(midi) or np.isinf(midi):
        midi = 0.0
    return midi



def midicps(midi_note: float) -> float:
    """
    Calculates the frequency of a specified midi note
    :param midi_note: The MIDI note
    :return: The frequency in Hz
    """
    cps = 440 * 2 ** ((midi_note - 69) / 12)
    if np.isnan(cps) or np.isneginf(cps) or np.isinf(cps):
        cps = 0.0
    return cps



def midiratio(interval: float) -> float:
    """
    Calculates the MIDI ratio of a specified midi interval
    :param midi_note: The MIDI interval in half steps
    :return: The ratio
    """
    ratio = 2 ** (interval / 12)
    if np.isnan(ratio) or np.isneginf(ratio) or np.isinf(ratio):
        ratio = 0.0
    return ratio



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



def exchanger(data: np.ndarray, hop: int):
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



def stochastic_exchanger(data: np.ndarray, max_hop: int):
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


def beat_envelope(tempo, sample_rate, beat_sequence, envelope_type="hanning", envelope_duration=1000) -> np.ndarray:
    """
    Generates a beat envelope. You specify the tempo, and provide a list of beat durations in order.
    This function will generate a volume envelope that can be applied to a sound file to make it
    pulse with these beat durations.
    :param tempo: The tempo
    :param sample_rate: The audio sample rate
    :param beat_sequence: A sequence of Fractions representing beat durations
    :param envelope_type: The envelope used for fade-in and fade-out on each beat
    :param envelope_duration: The envelope duration. The envelope will be split in half and applied to the beginning and end of the beat.
    If the beat is too short, a shorter envelope will be generated for that beat.
    :return: An array with the beat envelope
    """
    # the fade in/fade out envelope for each beat
    
    if envelope_type == "bartlett":
        envelope_fn = np.bartlett
    elif envelope_type == "blackman":
        envelope_fn = np.blackman
    elif envelope_type == "hamming":
        envelope_fn = np.hamming
    else:
        envelope_fn = np.hanning

    local_envelope = envelope_fn(envelope_duration)
    envelope = np.zeros((0))
    
    # Track the number of beats that have elapsed so that we can adjust for quantization
    CHECK_INT = 10
    accumulated_beats = Fraction(0, 1)
    accumulated_samples = 0

    for i, beat in enumerate(beat_sequence):
        beat_dur = int(Fraction(sample_rate, 1) * Fraction(60,tempo) * beat)
        
        # Track the accumulated beat durations to adjust for quantization
        accumulated_beats += beat
        accumulated_samples += beat_dur

        # Check total duration every N beats
        if i % CHECK_INT == CHECK_INT - 1:
            # The total number of samples that should have elapsed
            total_samples = int(Fraction(sample_rate, 1) * Fraction(60,tempo) * accumulated_beats)
            
            # The difference between that and the number of samples that have actually elapsed
            difference = accumulated_samples - total_samples

            # Adjust the beat duration of the Nth beat to avoid quantization error
            beat_dur -= difference
            accumulated_samples -= difference
        
        if beat_dur > envelope_duration:
            current_beat = np.hstack((local_envelope[:envelope_duration//2], np.ones((beat_dur - envelope_duration)), local_envelope[envelope_duration//2:]))
        else:
            current_beat = envelope_fn(beat_dur)
        envelope = np.hstack((envelope, current_beat))
    
    return envelope


def beat_envelope_multichannel(tempo: float, sample_rate: int, beat_sequence: list, num_channels: int, channel_levels: list, envelope_type="hanning", envelope_duration=1000) -> np.ndarray:
    """
    Generates a multichannel beat envelope. You specify the tempo, and provide a list of N beat durations in order.
    You also specify the number of output channels M and a list of level tuples NxM - one level coefficient for each channel,
    for each beat.
    This function will generate a volume envelope that can be applied to a sound file to make it
    pulse with these beat durations. You will need to provide a multichannel sound file with the correct number of channels
    when you apply the envelope.
    :param tempo: The tempo
    :param sample_rate: The audio sample rate
    :param beat_sequence: A sequence of Fractions representing beat durations
    :param num_channels: The number of channels in the envelope
    :param channel_levels: A list of channel level tuples (or lists). Each sublist corresponds to the level coefficients for the current beat.
    :param envelope_type: The envelope used for fade-in and fade-out on each beat
    :param envelope_duration: The envelope duration. The envelope will be split in half and applied to the beginning and end of the beat.
    If the beat is too short, a shorter envelope will be generated for that beat.
    :return: An array with the beat envelope
    """
    # the fade in/fade out envelope for each beat
    
    if envelope_type == "bartlett":
        envelope_fn = np.bartlett
    elif envelope_type == "blackman":
        envelope_fn = np.blackman
    elif envelope_type == "hamming":
        envelope_fn = np.hamming
    else:
        envelope_fn = np.hanning

    local_envelope = envelope_fn(envelope_duration)
    envelope = np.zeros((num_channels, 0))
    
    # Track the number of beats that have elapsed so that we can adjust for quantization
    CHECK_INT = 10
    accumulated_beats = Fraction(0, 1)
    accumulated_samples = 0

    for i, beat in enumerate(beat_sequence):
        beat_dur = int(Fraction(sample_rate, 1) * beat * Fraction(60, 1) / Fraction(tempo))
        
        # Track the accumulated beat durations to adjust for quantization
        accumulated_beats += beat
        accumulated_samples += beat_dur

        # Check total duration every N beats
        if i % CHECK_INT == CHECK_INT - 1:
            # The total number of samples that should have elapsed
            total_samples = int(Fraction(sample_rate, 1) * Fraction(60,tempo) * accumulated_beats)
            
            # The difference between that and the number of samples that have actually elapsed
            difference = accumulated_samples - total_samples

            # Adjust the beat duration of the Nth beat to avoid quantization error
            beat_dur -= difference
            accumulated_samples -= difference
        
        if beat_dur > envelope_duration:
            current_beat = np.hstack((local_envelope[:envelope_duration//2], np.ones((beat_dur - envelope_duration)), local_envelope[envelope_duration//2:]))
        else:
            current_beat = envelope_fn(beat_dur)

        # Apply channel-specific levels
        current_beat = np.vstack([current_beat * channel_levels[i][j] for j in range(len(channel_levels[i]))])
        envelope = np.hstack((envelope, current_beat))
    
    return envelope



def panner(num_channels: int, start_pos: int, end_pos: int, num_iterations: int, pan_law: str = "constant_power"):
    """
    Multichannel panner, moving from start_pos to end_pos over num_iterations.
    It generates a list of pan coefficients (each coefficient is the volume coefficient
    for the corresponding channel).
    (https://www.cs.cmu.edu/~music/icm-online/readings/panlaws/panlaws.pdf)

    :param num_channels: The number of channels
    :param start_pos: The start panning position
    :param end_pos: The end panning position
    :param num_iterations: The number of steps to take to move from start_pos to end_pos
    :param pan_law: The pan law ("linear", "constant_power", "neg_4_5_db")
    :return: An array of pan coefficients
    """
    i: cython.int
    j: cython.int
    pos_arr = np.linspace(start_pos, end_pos, num_iterations) % num_channels
    pan_coefficients = []
    for i in range(pos_arr.shape[-1]):
        frac, pos = math.modf(float(pos_arr[i]))
        pos = int(pos) % num_channels
        next_pos = (pos + 1) % num_channels
        coefficients = np.zeros((num_channels))

        # Constant power panning
        if pan_law == "constant_power":
            theta = frac * np.pi / 2
            coefficients[pos] = np.cos(theta)
            coefficients[next_pos] = np.sin(theta)

        # -4.5 dB panning
        elif pan_law == "neg_4_5_db":
            theta = frac * np.pi / 2
            coefficients[pos] = np.sqrt((np.pi / 2 - theta) * 2 / np.pi * np.cos(theta))
            coefficients[next_pos] = np.sqrt(frac * np.sin(theta))
        
        # Linear panning
        else:
            coefficients[pos] = 1 - frac
            coefficients[next_pos] = frac
        
        pan_coefficients.append(coefficients)
    return np.vstack(pan_coefficients)



def pan_mapper(pan_coefficients: np.ndarray, mapper: np.ndarray):
    """
    Maps pan positions to actual speaker positions. You pass a mapping array 
    that lists the speaker numbers in panning order.
    
    This is useful if you want to use a different numbering system for your 
    pan positions than the numbering system used for the actual output channels.
    For example, you might want to pan in a circle for a quad-channel setup,
    but the hardware is set up for stereo pairs.

    Example: Suppose you have a quad setup. Your mapper would be [0, 1, 3, 2] 
    if you are thinking clockwise, or [1, 0, 2, 3] if you are thinking counterclockwise. 
    If you have an 8-channel setup, your mapper would be [0, 1, 3, 5, 7, 6, 4, 2] 
    for clockwise and [1, 0, 2, 4, 6, 7, 5, 3] for counterclockwise.
    
    :param pan_coefficients: A list of pan coefficient lists
    :param mapper: The mapper for reordering the pan coefficients
    :return: A new, mapped pan coefficient list
    """
    newlist = np.zeros(pan_coefficients.shape)
    if pan_coefficients.ndim == 1:
        for j, pos in enumerate(mapper):
            newlist[pos] = pan_coefficients[j]
    elif pan_coefficients.ndim == 2:
        for i in range(pan_coefficients.shape[0]):
            for j, pos in enumerate(mapper):
                newlist[i, pos] = pan_coefficients[i, j]
    return newlist



def pan_level_adjuster(pan_levels: np.ndarray):
    """
    Adjusts pan levels in a list for a power sum of 1. The values in the list should be fractional 
    volume levels that sum to 1. After applying this operations, the values in the list will be adjusted
    so that their squares now sum to 1. The levels are adjusted in place.

    The idea is that you might want to arbitrarily divide the total volume of a sound over several
    channels. However, you want the sum of the signal power to equal to 1. So you need to
    adjust these fractional levels so that the power sum is correct. This function computes
    a scalar that is applied to all of the pan levels to make the summed power level equal to 1.

    :param pan_levels: A list of pan levels (one level for each channel)
    """
    if pan_levels.ndim == 1:
        scaler = 1 / np.sqrt(np.sum(np.square(pan_levels)))
        pan_levels = np.multiply(pan_levels, scaler, out=pan_levels)
    
    else:
        i: cython.int
        for i in range(pan_levels.shape[0]):
            scaler = 1 / np.sqrt(np.sum(np.square(pan_levels[i])))
            pan_levels[i] *= scaler
