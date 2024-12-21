"""
File: audiofile.py
Author: Jeff Martin
Date: 1/25/23

This is a no-nonsense audio file reader, writer, etc. It supports fixed (int) files up to
32-bit and float files up to 64-bit (for WAV), and fixed (int) files up to 32-bit (for AIFF).
Note that float format is not supported for AIFF. If you are crazy enough to think you need 
something bigger than 32/64-bit (hint: you do not), you can easily modify this code to support 
128-bit and higher. There is also basic functionality for plotting and scaling audio files. 

References: 
https://ccrma.stanford.edu/courses/422-winter-2014/projects/WaveFormat/ (for the canonical WAV format)
https://www.sounddevices.com/32-bit-float-files-explained/ (for 32-bit float files)
https://www.mmsp.ece.mcgill.ca/Documents/AudioFormats/WAVE/WAVE.html (for audio format specification other than 1 - PCM)
http://paulbourke.net/dataformats/audio/ (AIFF)
http://midi.teragonaudio.com/tech/aiff.htm (AIFF)
"""

import numpy as np
import math
import os
import pedalboard as pb
import re
import struct

LARGE_FIELD = 4
SMALL_FIELD = 2


class AudioFile:
    """
    A representation of a RIFF WAV audio file
    """
    def __init__(self, **kwargs):
        """
        Creates a new AudioFile
        """
        self.audio_format = 1 if "audio_format" not in kwargs else kwargs["audio_format"]
        self.bits_per_sample = 0 if "bits_per_sample" not in kwargs else kwargs["bits_per_sample"]
        self.block_align = 0 if "block_align" not in kwargs else kwargs["block_align"]
        self.byte_rate = 0 if "byte_rate" not in kwargs else kwargs["byte_rate"]
        self.bytes_per_sample = 0 if "bytes_per_sample" not in kwargs else kwargs["bytes_per_sample"]
        self.duration = 0 if "duration" not in kwargs else kwargs["duration"]
        self.file_name = "" if "file_name" not in kwargs else kwargs["file_name"]
        self.num_channels = 0 if "num_channels" not in kwargs else kwargs["num_channels"]
        self.frames = 0 if "frames" not in kwargs else kwargs["frames"]
        self.sample_rate = 0 if "sample_rate" not in kwargs else kwargs["sample_rate"]
        self.samples = None

    def copy_header(other):
        """
        Copies the header of another AudioFile and returns a new AudioFile object
        :param other: The other AudioFile
        :return: A new AudioFile
        """
        return AudioFile(audio_format=other.audio_format, 
                         bits_per_sample=other.bits_per_sample, 
                         block_align=other.block_align, 
                         byte_rate=other.byte_rate, 
                         bytes_per_sample=other.bytes_per_sample, 
                         duration=other.duration, 
                         frames=other.frames, 
                         num_channels=other.num_channels, 
                         sample_rate=other.sample_rate)


def _unpack_float80(bytes) -> int:
    """
    A hack to get the sample rate from a float 80 number. Since Python doesn't really have
    native support for the float80 format, we have to be creative. We take advantage of the
    fact that the sample rate will always be a whole number, and discard the decimal part.
    :param bytes: Some bytes representing a float 80 number
    :return: An integer with the whole number part
    """
    chunk1 = int.from_bytes(bytes[:2], byteorder="big", signed=False)
    # sign = chunk1 >> 15  # extract the sign bit (this is unnecessary because sample rates are never negative)
    exponent = (chunk1 << 1) >> 1  # get rid of the sign bit
    exponent -= 16383  # 2 ** 14 - 1 (the sign part is two's complement)
    num_bytes = math.ceil(exponent/8) + 1
    int_part = int.from_bytes(bytes[2:2+num_bytes], byteorder="big", signed=False)
    int_part >>= num_bytes * 8 - exponent - 1
    return int_part
    

def _pack_float80(num: int):
    """
    Packs a sample rate to float 80 format.
    :param num: An integer to convert
    :return: The packed format
    """
    exp = math.ceil(math.log2(num)) - 1
    exp_part = (exp + 16383).to_bytes(2, byteorder="big", signed=False)
    mantissa = num << (32 - exp - 1)
    mantissa = mantissa.to_bytes(length=4, byteorder="big", signed=False)
    return exp_part + mantissa + b'\x00\x00\x00\x00'


def convert(file: AudioFile, format: str):
    """
    Converts an AudioFile from one sample format to another. Use this to change between
    int and float format, or to change bit depth (e.g. 16 to 24 bit).
    Supported conversion types: 'int16', 'int24', 'int32', 'float32', 'float64'
    :param file: An AudioFile
    :param format: The destination format
    """
    
    # Converts to intxx format
    if "int" in format:
        format_bits = int(format[3:])
        if format_bits <= 16:
            new_dtype = np.int16
        else:
            new_dtype = np.int32
        
        if file.audio_format == 1:
            format_ratio = format_bits / file.bits_per_sample
            float_arr = file.samples * format_ratio
            float_arr = float_arr.round(decimals=0)
            file.samples = float_arr.astype(new_dtype)

        elif file.audio_format == 3:
            float_arr = file.samples * ((2 ** format_bits) / 2 - 1)
            float_arr = float_arr.round(decimals=0)
            file.samples = float_arr.astype(new_dtype)

        file.audio_format = 1
        file.bits_per_sample = format_bits
        file.bytes_per_sample = format_bits // 8

    # Converts to float32 or float64 format
    elif "float" in format:
        format_bits = int(format[5:])
        format_ratio = format_bits / file.bits_per_sample
        
        if format_bits == 32:
            new_dtype = np.float32
        else:
            new_dtype = np.float64
        
        if file.audio_format == 1:
            float_arr = np.divide(file.samples, ((2 ** file.bits_per_sample) / 2 - 1), None, dtype=new_dtype)
            file.samples = float_arr

        elif file.audio_format == 3:
            file.samples = file.samples.astype(new_dtype)

        file.audio_format = 3
        file.bits_per_sample = format_bits
        file.bytes_per_sample = format_bits // 8


def find_files(directory_name: str) -> list:
    """
    Finds all WAV and AIFF files within a directory (and its subdirectories, if recurse=True)

    :param directory_name: The directory name
    :param format: The destination format (supported formats are 'int16', 'int24', 'int32', 'float32',
    and 'float64')
    :return: A list of file names
    """
    files_audio = []
    search = re.compile(r"(\.wav$)|(\.aif$)|(\.aiff$)")

    for path, subdirectories, files in os.walk(directory_name):
        for name in files:
            result = search.search(name)
            if result:
                files_audio.append(os.path.join(path, name))

    return files_audio


def read(file_name: str, target_sample_rate: int = 44100) -> AudioFile:
    """
    Reads an audio file (AIFF or WAV) and returns an AudioFile object containing the contents of the
    file. It uses pedalboard for speed.
    :param file_name: The name of the file
    :param target_sample_rate: The target sample rate (audio will be resampled)
    :return: An AudioFile
    """
    audio = None
    if re.search(r'(\.aif$)|(\.aiff$)', file_name):
        audio = _read_aiff(file_name, True)
    elif re.search(r'\.wav$', file_name):
        audio = _read_wav(file_name, True)
    with pb.io.AudioFile(file_name, 'r').resampled_to(target_sample_rate) as infile:
        audio.samples = infile.read(infile.frames)
    return audio


def _read_aiff(file_name: str, header_only=False) -> AudioFile:
    """
    Reads an AIFF file and returns an AudioFile object with the data. Currently this implementation
    supports reading fixed (int) files up to 64-bit. Larger files could be supported, but who would 
    actually need to use them?

    :param file_name: The name of the file
    :param header_only: Whether or not to just read the header of the file and ignore the samples.
    :return: An AudioFile object with the contents of the AIFF file
    """
    audio_file = AudioFile()
    audio_file.audio_format = 1
    
    with open(file_name, "rb") as audio:
        HEADER1 = b'FORM'
        HEADER2 = b'AIFF'
        HEADER3 = b'COMM'
        HEADER4 = b'SSND'
        
        # We use this to validate the file as we go. If we encounter corrupt data, we will flip this to False.
        valid_file = True
        eof = False

        while not eof:
            chunk_title = audio.read(4)
            chunk_size = audio.read(LARGE_FIELD)
            chunk_size = int.from_bytes(chunk_size, byteorder="big", signed=False)
            
            if len(chunk_title) < LARGE_FIELD:
                eof = True
            
            # read the form chunk
            elif chunk_title == HEADER1:
                label = audio.read(LARGE_FIELD)

                # We can't read files that are in a weird format
                if label != HEADER2:
                    valid_file = False
            
            # read the common chunk
            elif chunk_title == HEADER3:
                common_chunk = audio.read(chunk_size)
                audio_file.num_channels = int.from_bytes(common_chunk[:2], byteorder="big", signed=False) # number of channels
                audio_file.frames = int.from_bytes(common_chunk[2:6], byteorder="big", signed=False) # number of frames
                audio_file.bits_per_sample = int.from_bytes(common_chunk[6:8], byteorder="big", signed=False) # sample size (bits)
                audio_file.bytes_per_sample = audio_file.bits_per_sample // 8
                audio_file.sample_rate = _unpack_float80(common_chunk[8:])
                audio_file.byte_rate = audio_file.sample_rate * audio_file.num_channels * audio_file.bytes_per_sample
                audio_file.block_align = audio_file.num_channels * audio_file.bytes_per_sample
                audio_file.duration = audio_file.frames / audio_file.sample_rate
                if header_only:
                    eof = True

            # read the samples
            elif chunk_title == HEADER4:
                # This adjustment is necessary because I've encountered a file that has a subchunk size
                # 1 byte too big.
                data_size = min(chunk_size, audio_file.frames * audio_file.block_align)
                
                # Read the offset and block size, which will probably be 0
                offset = audio.read(LARGE_FIELD)
                # offset = int.from_bytes(offset, byteorder="big", signed=False)
                block_size = audio.read(LARGE_FIELD)
                # block_size = int.from_bytes(block_size, byteorder="big", signed=False)
                
                # Read the rest of the file
                data = audio.read()
                if audio_file.bits_per_sample <= 16:
                    audio_file.samples = np.zeros((audio_file.num_channels, audio_file.frames), dtype=np.int16)
                elif audio_file.bits_per_sample <= 32:
                    audio_file.samples = np.zeros((audio_file.num_channels, audio_file.frames), dtype=np.int32)
                else:
                    audio_file.samples = np.zeros((audio_file.num_channels, audio_file.frames), dtype=np.int64)
                
                k = 0
                for i in range(0, data_size, audio_file.block_align):
                    for j in range(0, audio_file.num_channels):
                        start_point = i + j * audio_file.bytes_per_sample
                        audio_file.samples[j, k] = int.from_bytes(data[start_point:start_point+audio_file.bytes_per_sample], byteorder="big", signed=True)
                    k += 1

            # Otherwise it's a chunk that we aren't concerned with, so we will discard it.
            else:
                audio.read(chunk_size)

    # If the AIFF file was formatted unusually, we return nothing and raise a warning.
    if valid_file:
        return audio_file
    else:
        raise RuntimeWarning(f"The AIFF file {file_name} was unusually formatted and could not be read.")


def _read_wav(file_name: str, header_only=False) -> AudioFile:
    """
    Reads a WAV file and returns an AudioFile object with the data. Currently this implementation
    supports reading fixed (int) files up to 32-bit and float files up to 64-bit. Larger files could
    be supported, but who would actually need to use them?

    The advantage of this implementation over the SciPy wavfile functionality is that it preserves
    more information so that you can reconstruct the original WAV file more accurately. For example,
    SciPy (at the time of writing) stores 24-bit samples in np.int32 format *but* left-shifts the
    absolute value of every sample by 8 bytes, inflating the amplitude. This implementation does
    not do that. Furthermore, it records the bit depth of the file so that you can actually write
    in 24-bit format.

    This implementation ignores all information in JUNK chunks, and assumes that there will be
    only one DATA chunk in the file. If for some reason you have multiple data chunks, only the
    contents of the last one will be preserved.

    Finally, at this time, this implementation can only read RIFF, not RF64, WAV files.

    :param file_name: The name of the file
    :param header_only: Whether or not to just read the header of the file and ignore the samples.
    :return: An AudioFile object with the contents of the WAV file
    """
    audio_file = AudioFile()
    with open(file_name, "rb") as audio:
        # Read the header. We will store the entire header in the AudioFile object which will make it easier to write
        # back to the file.
        CHUNK_HEADER_SIZE = 8
        RIFF_CHUNK_SIZE = 12
        RIFF_CHUNK_1 = b'RIFF'
        RIFF_CHUNK_3 = b'WAVE'
        FMT_CHUNK_1 = b'fmt '
        DATA_CHUNK_1 = b'data'
        remaining_size = 0
        chunk_riff = audio.read(RIFF_CHUNK_SIZE)
        audio_file.file_name = file_name
        
        # We use this to validate the file as we go. If we encounter corrupt data, we will flip this to False.
        valid_file = True

        # Validate the RIFF chunk before proceeding
        if chunk_riff[:4] != RIFF_CHUNK_1 or chunk_riff[8:] != RIFF_CHUNK_3:
            valid_file = False
            print("BAD RIFF")
        else:
            remaining_size = int.from_bytes(chunk_riff[4:8], byteorder="little", signed=False)
        # print(remaining_size)
        # Now that the file has been read, we continue to read the remaining subchunks.
        # The only remaining subchunk we are interested in is the data subchunk.
        if valid_file:
            while remaining_size > 0:
                subchunk_header = audio.read(CHUNK_HEADER_SIZE)
                remaining_size -= CHUNK_HEADER_SIZE
                subchunk_size = int.from_bytes(subchunk_header[4:8], byteorder="little", signed=False)
                subchunk_data = audio.read(subchunk_size)
                remaining_size -= subchunk_size

                # If we've read the FMT chunk
                if subchunk_header[:4] == FMT_CHUNK_1:
                    audio_file.audio_format = int.from_bytes(subchunk_data[:2], byteorder="little", signed=False)
                    audio_file.num_channels = int.from_bytes(subchunk_data[2:4], byteorder="little", signed=False)                    
                    audio_file.sample_rate = int.from_bytes(subchunk_data[4:8], byteorder="little", signed=False)
                    audio_file.byte_rate = int.from_bytes(subchunk_data[8:12], byteorder="little", signed=False)
                    audio_file.block_align = int.from_bytes(subchunk_data[12:14], byteorder="little", signed=False)
                    audio_file.bits_per_sample = int.from_bytes(subchunk_data[14:16], byteorder="little", signed=False)
                    audio_file.bytes_per_sample = audio_file.bits_per_sample // 8

                # Detect if we've read a data subchunk. If this is something else
                # (e.g. a JUNK subchunk), we ignore it.
                elif subchunk_header[:4] == DATA_CHUNK_1:
                    if audio_file.num_channels * audio_file.bytes_per_sample > 0:
                        audio_file.frames = subchunk_size // (audio_file.num_channels * audio_file.bytes_per_sample)
                        audio_file.duration = audio_file.frames / audio_file.sample_rate
                    else:
                        audio_file.frames = 1
                        audio_file.duration = 1

                    if header_only:
                        remaining_size = 0
                        break

                    # This adjustment is necessary because I've encountered a file that has a subchunk size
                    # 1 byte too big.
                    subchunk_size = min(subchunk_size, audio_file.frames * audio_file.block_align)
                    j = 0  # frame index

                    # This is for 8-, 16-, 24-, and 32-bit fixed (int) format. Theoretically we could support 64-bit
                    # int, but who would want to use it?
                    if audio_file.audio_format == 1 and audio_file.bits_per_sample <= 32:
                        audio_file.samples = np.zeros((audio_file.num_channels, audio_file.frames), dtype=np.int32)                        
                        for i in range(0, subchunk_size, audio_file.block_align):
                            for k in range(0, audio_file.num_channels):
                                audio_file.samples[k, j] = int.from_bytes(
                                    subchunk_data[i + k * audio_file.bytes_per_sample : i + k * audio_file.bytes_per_sample + audio_file.bytes_per_sample],
                                    byteorder="little",
                                    signed=True
                                )
                            j += 1  # move to next frame

                    # This is for 32-bit float format.
                    elif audio_file.audio_format == 3 and audio_file.bits_per_sample == 32:
                        audio_file.samples = np.zeros((audio_file.num_channels, audio_file.frames), dtype=np.float32)
                        for i in range(0, subchunk_size, audio_file.block_align):
                            for k in range(0, audio_file.num_channels):
                                audio_file.samples[k, j] = struct.unpack('f',
                                    subchunk_data[i + k * audio_file.bytes_per_sample : i + k * audio_file.bytes_per_sample + audio_file.bytes_per_sample]
                                )[0]
                            j += 1  # move to next frame

                    # This is for 64-bit float format. Theoretically we could also support 128-bit, but who would be
                    # crazy enough to want to use it?
                    elif audio_file.audio_format == 3 and audio_file.bits_per_sample == 64:
                        audio_file.samples = np.zeros((audio_file.num_channels, audio_file.frames), dtype=np.float64)
                        for i in range(0, subchunk_size, audio_file.block_align):
                            for k in range(0, audio_file.num_channels):
                                audio_file.samples[k, j] = struct.unpack('d',
                                    subchunk_data[i + k * audio_file.bytes_per_sample : i + k * audio_file.bytes_per_sample + audio_file.bytes_per_sample]
                                )[0]
                            j += 1  # move to next frame

    # If the WAV file was formatted unusually (for example, not in PCM or float), we return nothing
    # and raise a warning.
    if valid_file:
        return audio_file
    else:
        raise RuntimeWarning(f"The WAV file {file_name} was unusually formatted and could not be read. This might be because you tried to read a WAV file that was not in PCM format.")
    

def _write_wav(file: AudioFile, path: str, write_junk_chunk=False):
    """
    Writes an audio file. Note that the audio_format must match the format used! For example,
    you cannot specify an audio_format of 3 (float) and use PCM (int32) data in the samples.

    Also, note that in the AudioFile, the following parameters should be properly set (other
    parameters will not be consulted when building the WAV file):
    - audio_format (1 for PCM, 3 for float)
    - bits_per_sample
    - num_channels
    - frames
    - sample_rate
    - scaling_factor

    You can specify to write a JUNK chunk (of size 36) if you wish. Since the header will be
    44 bytes, this makes the total size of header + JUNK to be 80 bytes. Note that the JUNK 
    chunk is traditionally used in CD audio for alignment, but more recently it is written 
    to allow recording without specifying RIFF or RF64 format; this can be specified later,
    after recording. RF64 allows for larger WAV files than RIFF.
    
    :param file: An AudioFile representation of the file
    :param path: A file path
    :param write_junk_chunk: Whether or not to write the junk chunk. 
    :return: None
    """
    with open(path, "wb") as audio:
        # Write the header
        header = b"".join([
            b"RIFF",
            (36 + file.frames * file.num_channels * (file.bits_per_sample // 8)).to_bytes(LARGE_FIELD, byteorder="little", signed=False),
            b"WAVE",
            b"fmt ",
            int(16).to_bytes(LARGE_FIELD, byteorder="little", signed=False),  # subchunk1size
            file.audio_format.to_bytes(SMALL_FIELD, byteorder="little", signed=False),  # audio format (1 for PCM; 3 for float)
            file.num_channels.to_bytes(SMALL_FIELD, byteorder="little", signed=False),  # num channels
            file.sample_rate.to_bytes(LARGE_FIELD, byteorder="little", signed=False),  # sample rate
            (file.sample_rate * file.num_channels * (file.bits_per_sample // 8)).to_bytes(LARGE_FIELD, byteorder="little", signed=False),  # byte rate
            (file.num_channels * (file.bits_per_sample // 8)).to_bytes(SMALL_FIELD, byteorder="little", signed=False),  # block align
            file.bits_per_sample.to_bytes(SMALL_FIELD, byteorder="little", signed=False),  # bits per sample
        ])
        audio.write(header)

        # Write the junk chunk if we are including it
        if write_junk_chunk:
            junk_header = b"".join([
                b"JUNK",  # junk id
                int(28).to_bytes(LARGE_FIELD, byteorder="little", signed=False),  # subchunk size of junk
                int(0).to_bytes(28, byteorder="little", signed=False)  # contents of junk
            ])
            audio.write(junk_header)

        # Write the data header
        data_header = b"".join([
            b"data",
            (file.frames * file.num_channels * (file.bits_per_sample // 8)).to_bytes(LARGE_FIELD, byteorder="little", signed=False)
        ])
        audio.write(data_header)

        # Write PCM data
        if file.audio_format == 1:
            for i in range(file.frames):
                for j in range(file.num_channels):
                    audio.write(int(file.samples[j, i]).to_bytes(file.bytes_per_sample, byteorder="little", signed=True))
        
        # Write float data
        elif file.audio_format == 3 and file.bits_per_sample == 32:
            for i in range(file.frames):
                for j in range(file.num_channels):
                    audio.write(struct.pack('f', file.samples[j, i]))
        
        # Write double data
        elif file.audio_format == 3 and file.bits_per_sample == 64:
            for i in range(file.frames):
                for j in range(file.num_channels):
                    audio.write(struct.pack('d', file.samples[j, i]))

        else:
            raise Exception(message="Invalid audio format.")


def _write_aiff(file: AudioFile, path: str):
    """
    Writes an audio file. Note that the audio_format must match the format used! For example,
    you cannot specify an audio_format of 3 (float) and use PCM (int32) data in the samples.

    Also, note that in the AudioFile, the following parameters should be properly set (other
    parameters will not be consulted when building the WAV file):
    - audio_format (1 for PCM, 3 for float)
    - bits_per_sample
    - num_channels
    - frames
    - sample_rate
    - scaling_factor
    
    :param file: An AudioFile representation of the file
    :param path: A file path
    :return: None
    """
    audio_size = file.num_channels * file.frames * (file.bits_per_sample // 8)
    with open(path, "wb") as audio:
        # Write the header
        header = b"".join([
            b"FORM",
            (4 + 26 + 16 + audio_size).to_bytes(LARGE_FIELD, byteorder="big", signed=False),
            b"AIFF",
            b"COMM",
            int(18).to_bytes(LARGE_FIELD, byteorder="big", signed=False),  # common chunk size
            file.num_channels.to_bytes(SMALL_FIELD, byteorder="big", signed=False),  # num channels
            file.frames.to_bytes(LARGE_FIELD, byteorder="big", signed=False),  # num frames
            file.bits_per_sample.to_bytes(SMALL_FIELD, byteorder="big", signed=False),  # bits per sample
            _pack_float80(file.sample_rate)
        ])
        audio.write(header)

        # Write the data header
        data_header = b"".join([
            b"SSND",
            (8 + audio_size).to_bytes(LARGE_FIELD, byteorder="big", signed=False),
            b'\x00\x00\x00\x00\x00\x00\x00\x00'
        ])
        audio.write(data_header)

        # Write PCM data
        if file.audio_format == 1:
            for i in range(file.frames):
                for j in range(file.num_channels):
                    audio.write(int(round(file.samples[j, i], None)).to_bytes(file.bits_per_sample // 8, byteorder="big", signed=True))
        
        else:
            raise Exception(message="Invalid audio format. AIFF only supports PCM fixed (int) format (1).")


def write_with_pedalboard(audio, file_name):
    """
    Writes an audio file using the Pedalboard library
    :param audio: The AudioFile object
    :param file_name: The file name to use
    """
    with pb.io.AudioFile(file_name, 'w', audio.sample_rate, audio.num_channels, audio.bits_per_sample) as outfile:
        outfile.write(audio.samples)
