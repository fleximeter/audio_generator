"""
File: corpus.py

This module finds MusicXML files and prepares them for training. It can find files
in a music21 corpus or in a directory and its subfolders.
"""

import os
import re
import torchaudio


def load_audio(directory_name: str) -> list:
    """
    Finds all audio files within a directory (and its subdirectories, if recurse=True).
    Loads them, and returns a list of 1D NumPy arrays. All stero files will be mixed down.
    All files will be resampled to the same sampling rate.
    :param directory_name: The directory name
    :return: A list of arrays
    """
    audio_corpus = []
    search = re.compile(r"(\.wav$)|(\.wave$)|(\.aif$)|(\.aiff$)")

    for path, subdirectories, files in os.walk(directory_name):
        for name in files:
            result = search.search(name)
            if result:
                file_path = os.path.join(*os.path.split(path), name)
                audio, sample_rate = torchaudio.load(file_path)
                audio_corpus.append({
                    "name": name,
                    "path": file_path,
                    "sample_rate": sample_rate,
                    "audio": audio,
                    "frames": audio.shape[-1],
                    "duration": audio.shape[-1] / sample_rate,
                    "channels": audio.shape[-2]
                })

    return audio_corpus
