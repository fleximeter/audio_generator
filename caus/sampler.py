"""
File: sampler.py
Author: Jeff Martin
Date: 1/26/23

This file contains functionality for processing audio files for use with samplers.
"""

import cython
import numpy as np


def extract_samples(audio: np.ndarray, amplitude_regions: list, pre_frames_to_include: int = 0, 
                    post_frames_to_include: int = 0, pre_envelope_type="hanning", pre_envelope_frames: int = 20,
                    post_envelope_type="hanning", post_envelope_frames: int = 20) -> list:
    """
    Extracts samples from an AudioFile.
    
    :param audio: A 2D numpy ndarray of audio samples
    :param amplitude_regions: A list of amplitude regions from the identify_amplitude_regions function
    :param pre_frames_to_include: If this is set to greater than 0, the sample will be extended backward
    to include these additional frames. This is useful for ensuring a clean sample onset.
    :param post_frames_to_include: If this is set to greater than 0, the sample will be extended forward
    to include these additional frames. This is useful for ensuring a clean sample release.
    :param pre_envelope_type: The envelope type to apply to the beginning of the sound, for a clean onset.
    Supported envelope types are Bartlett, Blackman, Hamming, and Hanning.
    :param pre_envelope_frames: The duration in frames of the pre envelope
    :param post_envelope_type: The envelope type to apply to the end of the sound, for a clean release.
    Supported envelope types are Bartlett, Blackman, Hamming, and Hanning.
    :param post_envelope_frames: The duration in frames of the post envelope
    
    :return: A list of AudioFile objects with the samples
    """
    # Update the audio array with zero padding to allow for pre and post frames
    audio = np.hstack((
        np.zeros((audio.shape[0], pre_frames_to_include), dtype=audio.dtype),
        audio,
        np.zeros((audio.shape[0], post_frames_to_include), dtype=audio.dtype)
    ))

    # Holds the samples that we extract
    samples = []

    # Create the samples
    for region in amplitude_regions:
        # Extract the samples. Note that since we zero-padded the start of the audio
        # array, it is necessary to add the pre and post frames to the *last* index
        # of the audio range.
        sample = audio[region["channel"], region["range"][0]:region["range"][1]+1+pre_frames_to_include+post_frames_to_include]
        sample = np.reshape(sample, (1,) + sample.shape)

        # Get the windows
        if pre_envelope_type == "bartlett":
            pre_envelope = np.bartlett(pre_envelope_frames * 2)[:pre_envelope_frames]
        elif pre_envelope_type == "blackman":
            pre_envelope = np.blackman(pre_envelope_frames * 2)[:pre_envelope_frames]
        elif pre_envelope_type == "hamming":
            pre_envelope = np.hamming(pre_envelope_frames * 2)[:pre_envelope_frames]
        else:
            pre_envelope = np.hanning(pre_envelope_frames * 2)[:pre_envelope_frames]
        
        if post_envelope_type == "bartlett":
            post_envelope = np.bartlett(post_envelope_frames * 2)[post_envelope_frames:]
        elif post_envelope_type == "blackman":
            post_envelope = np.blackman(post_envelope_frames * 2)[post_envelope_frames:]
        elif post_envelope_type == "hamming":
            post_envelope = np.hamming(post_envelope_frames * 2)[post_envelope_frames:]
        else:
            post_envelope = np.hanning(post_envelope_frames * 2)[post_envelope_frames:]
        
        # Apply the windows
        i: cython.int
        j: cython.int
        for i in range(pre_envelope_frames):
            for j in range(sample.shape[0]):
                sample[j, i] *= pre_envelope[i]
        for i in range(post_envelope_frames):
            for j in range(sample.shape[0]):
                sample[j, sample.shape[1] - post_envelope_frames + i] *= post_envelope[i]

        samples.append(sample)
    
    return samples



def identify_amplitude_regions(audio: np.ndarray, level_delimiter: cython.double = -30, 
                               num_consecutive: cython.int = 10, scale_level_delimiter: bool = True) -> list:
    """
    Identifies amplitude regions in a sound. You provide a threshold, and any time the threshold is
    breached, we start a new amplitude region which ends when we return below the threshold. This is
    useful for pulling out individual samples from a file that has multiple samples in it.

    :param audio: An audio array
    :param level_delimiter: The lowest level (dBFS) allowed in a region. This will be scaled by the maximum amplitude
    in the audio file channel that is being analyzed, unless that feature is turned off by the next parameter. 
    :param num_consecutive: The number of consecutive samples below the threshold required to end a region.
    Note that these samples will not be included in the amplitude region; they will only be used to determine
    if an amplitude region is ending.
    :param scale_level_delimiter: Whether or not to scale the level delimiter by the maximum amplitude in
    the audio file channel that is being analyzed
    
    :return: A list of tuples. Each tuple contains the starting and ending frame index of an amplitude region.
    """
    current_region_start_frame: cython.int
    num_consecutive_below_threshold: cython.int
    last_frame_above_threshold: cython.int
    i: cython.int
    j: cython.int
    
    regions = []
    current_region_active = False
    current_region_start_frame = 0
    num_consecutive_below_threshold = 0
    last_frame_above_threshold = 0
    
    # Scale the level delimiter by the maximum amplitude in the audio file if necessary.
    # Also convert the level delimiter away from dBFS for less computation.
    level_delimiter = 10 ** (level_delimiter / 20)
    if scale_level_delimiter:
        level_delimiter *= np.max(np.abs(audio))
        
    if audio.size > 0 and len(audio.shape) == 2:
        for i in range(audio.shape[0]):
            for j in range(audio.shape[1]):
                # If we've broken above the minimum level
                if np.abs(audio[i, j]) >= level_delimiter:
                    last_frame_above_threshold = j
                    num_consecutive_below_threshold = 0
                    if not current_region_active:
                        current_region_active = True
                        current_region_start_frame = j

                # If we haven't broken above the minimum level, we need to track how long
                # we've been below the minimum level
                else:
                    num_consecutive_below_threshold += 1
                    if current_region_active and num_consecutive_below_threshold >= num_consecutive:
                        regions.append({"channel": i, "range": (current_region_start_frame, last_frame_above_threshold)})
                        current_region_active = False

            # If we finish with a current region active, we need to close the region
            if current_region_active:
                regions.append({"channel": i, "range": (current_region_start_frame, audio.shape[-1] - 1)})
    
    else:
        raise Exception("The audio must have samples in it, and it must be a 2D array, " \
                        "where the first dimension is the channel index and the second " \
                        "dimension is the frame index.")

    return regions



def detect_peaks(audio: np.ndarray) -> list:
    """
    Detects peaks in an audio file. A peak is located at a sample N where the waveform changes direction.
    :param audio: An AudioFile object with the contents of a WAV file
    :return: Returns a list of indices; each index corresponds to a frame with a peak in the selected channel.
    """
    i: cython.int
    j: cython.int
    peaks = []
    if audio.size > 0 and len(audio.shape) == 2:
        for i in range(audio.shape[0]):
            for j in range(1, audio.shape[1] - 1):
                if audio[i, j-1] < audio[i, j] > audio[i, j+1] \
                    and audio[i, j] > 0:
                    peaks.append((i, j))
                elif audio[i, j-1] > audio[i, j] < audio[i, j+1] \
                    and audio[i, j] < 0:
                    peaks.append((i, j))
    else:
        raise Exception("The audio must have samples in it, and it must be a 2D array, " \
                        "where the first dimension is the channel index and the second " \
                        "dimension is the frame index.")
    return peaks



def fit_amplitude_envelope(audio: np.ndarray, chunk_width: cython.int = 5000) -> list:
    """
    Fits an amplitude envelope to a provided audio file.
    Detects peaks in an audio file. Peaks are identified by being surrounded by lower absolute values to either side.
    :param audio: An audio array
    :param chunk_width: The AudioFile is segmented into adjacent chunks, and we look for the highest peak amplitude 
    in each chunk.
    :return: Returns a list of tuples; the tuple has a channel, an index, and an amplitude value.
    """
    i: cython.int
    j: cython.int
    envelope = []
    abs_audio = np.abs(audio)
    if audio.size > 0 and len(audio.shape) == 2:
        for i in range(audio.shape[0]):
            for j in range(0, audio.shape[1], chunk_width):
                peak_idx = np.argmax(abs_audio[i, j:j+chunk_width])
                envelope.append((i, j + peak_idx, abs_audio[i, j + peak_idx]))
    else:
        raise Exception("The audio must have samples in it, and it must be a 2D array, " \
                        "where the first dimension is the channel index and the second " \
                        "dimension is the frame index.")
    return envelope



def detect_major_peaks(audio: np.ndarray, min_percentage_of_max: cython.double = 0.9, chunk_width: cython.int = 5000) -> list:
    """
    Detects major peaks in an audio file. A major peak is a sample peak that is one of the highest in its "local region."
    This is useful for identifying the period length (and therefore the fundamental frequency) of audio with a strong
    first harmonic, since the first harmonic will create the highest peak of the waveform. It will not
    work very well on Gaussian noise or audio with a very weak fundamental.
    
    The "local region" is specified by the chunk width. We segment the audio file into C segments of width chunk_width,
    and search for the highest peak P in that chunk. Then we identify all other peaks that are close in height
    to the highest peak P. A peak is close in height to said peak P if it is greater than or equal to min_percentage_of_max
    of that peak. (For example, suppose the highest peak is 1, and the min_percentage_of_max is 0.9. Then any peak with
    amplitude from 0.9 to 1 will be considered a major peak.)
    
    :param audio: An audio array
    :param min_percentage_of_max: A peak must be at least this percentage of the maximum peak to be included as a major
    peak.
    :param chunk_width: The width of the chunk to search for the highest peak
    :return: Returns a list of tuples; the tuple has a channel index, a frame index, and an amplitude value.
    """
    i: cython.int
    j: cython.int
    k: cython.int
    peaks = []

    if audio.size > 0 and len(audio.shape) == 2:
        for i in range(audio.shape[0]):
            for j in range(1, audio.shape[1] - 1, chunk_width):
                # Get the index and absolute value of the highest peak in the current chunk
                peak_idx = j + np.argmax(audio[i, j:j+chunk_width])
                peak_val = audio[i, peak_idx]
                
                # print(peak_idx, peak_val)

                # Iterate through the current chunk and find all major peaks
                k = j
                while k < j + chunk_width and k < audio.shape[1] - 1:
                    if (
                        # If the current sample is a positive peak (both neighboring samples are lower)
                        (audio[i, k-1] < audio[i, k] > audio[i, k+1] \
                            and audio[i, k] > 0) \
                        
                        # And the peak is a major peak
                        and audio[i, k] >= peak_val * min_percentage_of_max
                    ):
                        peaks.append((i, k, audio[i, k]))
                    k += 1
    else:
        raise Exception("The audio must have samples in it, and it must be a 2D array, " \
                        "where the first dimension is the channel index and the second " \
                        "dimension is the frame index.")
    
    return peaks



def detect_loop_points(audio: np.ndarray, num_periods: cython.int = 5, effective_zero: cython.double = 0.001, 
                       maximum_amplitude_variance: cython.double = 0.1, sample_amplitude_level_boundary: cython.double = 0.1, 
                       loop_left_padding: cython.int=100, loop_right_padding: cython.int=100) -> list:
    """
    Detects loop points in an audio sample. Loop points are frame indices that could be used for
    a seamless repeating loop in a sampler. Ideally, if you choose loop points correctly, no crossfading
    would be needed within the loop.

    We have several requirements for a good loop:
    1. The standard deviation of peak amplitudes should be minimized (i.e. the loop is not increasing or decreasing in amplitude)
    2. The distance between successive wave major peaks should be consistent (i.e. we are looking for periodic waveforms)
    3. The frames at which looping begins and ends should have values as close to 0 as possible (we want to avoid clicks)
    
    :param audio: An AudioFile object
    :param num_periods: The number of periods to include from the waveform
    :param effective_zero: The threshold below which we just consider the amplitude to be 0. This is assumed to be a 
    floating-point value between 0 (no amplitude) and 1 (max amplitude). If your file is fixed format, this will be 
    automatically scaled.
    :param maximum_amplitude_variance: The maximum percentage difference between the biggest and 
    smallest major peak in the loop
    :param sample_amplitude_level_boundary: Used to determine where the sample sits in the provided audio file. This is useful
    because we don't want to detect loop points in just any region of the sample - the loop points should have a similar dynamic
    level to the rest of the sample.
    :param loop_left_padding: The number of frames to ignore at the start of the amplitude region. This is needed to keep the loop
    from being in the attack portion of the sound.
    :param loop_right_padding: The number of frames to ignore at the end of the amplitude region. This is needed to keep the loop
    from being too close to the decay portion of the sound.
    
    :return: A list of tuples that are channel index, start frame, and end frame for looping
    """
    i: cython.int
    j: cython.int
    k: cython.int

    frame_tuples = []

    if audio.size > 0 and len(audio.shape) == 2:
        for i in range(audio.shape[0]):
            # The audio file amplitude level ranges
            amplitude_level_ranges = identify_amplitude_regions(audio, sample_amplitude_level_boundary, 10, i)

            if len(amplitude_level_ranges) > 0:
                # combine the amplitude level ranges
                amplitude_level_range = (amplitude_level_ranges[0][0], amplitude_level_ranges[-1][-1])

                # The major peaks in the sound file.
                major_peaks = detect_major_peaks(audio, 0.9, 5000)
                
                # This stores frame tuples that identify potential loop points.
                
                # We will try to build a loop starting at each peak, then shifting backward to a zero point.
                for j in range(len(major_peaks)):
                    potential_loop_peaks = []
                    
                    # We will use these two valuse to determine if there is too much dynamic variation
                    # within the proposed loop.
                    max_peak = -np.inf  # the absolute value of the major peak with greatest magnitude
                    min_peak = np.inf  # the absolute value of the major peak with least magnitude
                
                    # We start by grabbing peaks for the minimum number of periods necessary. We have to 
                    # grab an extra peak to complete the final period.
                    for k in range(j, min(j + num_periods + 1, len(major_peaks))):
                        potential_loop_peaks.append(major_peaks[k])
                        peak_abs = np.abs(major_peaks[k][1])
                        if peak_abs > max_peak:
                            max_peak = peak_abs
                        if peak_abs < min_peak:
                            min_peak = peak_abs
                    
                    # If we weren't able to pull enough periods, we can't continue with making the loop.
                    if len(potential_loop_peaks) < num_periods:
                        break

                    # If there's too much dynamic variation in this audio chunk, we can't continue with
                    # making the loop. We want loops that have a steady sound, rather than fluctuating
                    # wildly in dynamic level.
                    if (max_peak - min_peak) / max_peak > maximum_amplitude_variance:
                        continue

                    # We need to record loop points now. Recall that the final peak is actually the beginning
                    # of the next period, so we need to move back one sample.
                    loop_points = [potential_loop_peaks[0][0], potential_loop_peaks[-1][0] - 1]
                    period_width = (loop_points[1] - loop_points[0]) // num_periods

                    # Now we shift back to make the loop start and end on 0. There might be multiple possible
                    # places where the loop could start and end on 0.
                    while loop_points[0] + period_width > potential_loop_peaks[0][0] and loop_points[0] >= 0:
                        loop_points[0] -= 1
                        loop_points[1] -= 1

                        # If we've found good loop points, we will record them. They must match the following criteria:
                        # 1. Start and end at the effective zero level (to avoid clicks)
                        # 2. Be located within the amplitude level range that was identified earlier (to avoid identifying
                        #    loop points within the attack or decay portions of the sample)
                        if np.abs(audio[i, loop_points[0]]) < effective_zero \
                            and np.abs(audio[i, loop_points[1]]) < effective_zero:
                            if loop_points[0] >= amplitude_level_range[0] + loop_left_padding \
                                and loop_points[1] <= amplitude_level_range[1] - loop_right_padding:
                                frame_tuples.append((loop_points[0], loop_points[1]))
                            break
    
    else:
        raise Exception("The audio must have samples in it, and it must be a 2D array, " \
                        "where the first dimension is the channel index and the second " \
                        "dimension is the frame index.")
    
    return frame_tuples
