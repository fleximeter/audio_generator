"""
File: corpus.py

This module finds MusicXML files and prepares them for training. It can find files
in a music21 corpus or in a directory and its subfolders.
"""

import os
import re


def load_audio(directory_name: str, sample_rate=44100) -> list:
    """
    Finds all audio files within a directory (and its subdirectories, if recurse=True).
    Loads them, and returns a list of 1D NumPy arrays. All stero files will be mixed down.
    All files will be resampled to the same sampling rate.
    :param directory_name: The directory name
    :param sample_rate: All files will be resampled to this sample rate.
    :return: A list of NumPy arrays
    """
    files = []
    search = re.compile(r"(\.wav$)|(\.wave$)|(\.aif$)|(\.aiff$)")

    for path, subdirectories, files in os.walk(directory_name):
        for name in files:
            result = search.search(name)
            if result:
                files.append(os.path.join(path, name))

    return files
