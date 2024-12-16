"""
File: corpus.py

This module finds MusicXML files and prepares them for training. It can find files
in a music21 corpus or in a directory and its subfolders.
"""

import os
import re
import torch
import torchaudio


def load_audio(directory_name: str) -> list:
    """
    Finds all audio files within a directory and its subdirectories.
    Loads them, mixes them to mono, and returns a list of dictionaries. 
    Each dictionary contains information about the audio file, as well as the 
    audio samples in the file.
    :param directory_name: The directory name
    :return: A list of audio file dictionaries
    """
    audio_corpus = []
    audio_file_regex = re.compile(r"(\.wav$)|(\.wave$)|(\.aif$)|(\.aiff$)")

    for path, subdirectories, files in os.walk(directory_name):
        for name in files:
            search_result = audio_file_regex.search(name)
            if search_result:
                file_path = os.path.join(*os.path.split(path), name)
                audio, sample_rate = torchaudio.load(file_path)
                if audio.ndim == 2:
                    audio = torch.sum(audio, 0)
                audio_corpus.append({
                    "name": name,
                    "path": file_path,
                    "sample_rate": sample_rate,
                    "audio": audio,
                    "frames": audio.shape[-1],
                    "duration": audio.shape[-1] / sample_rate,
                    "channels": 1
                })

    return audio_corpus
