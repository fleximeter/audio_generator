# Music Generator

## About
This repository is a generative AI system for producing the next FFT frame in an audio file. It can train models on a collection of audio files. The architecture is the PyTorch LSTM architecture, and it will likely need considerable resources for training if you want an acceptable model. Predictions generate output logits that consist of the magnitude and phase spectrums concatenated.

## Resource needs
The default training device is CUDA, and MPS is the first fallback, with a CPU as the last-level default. When training, the estimated time remaining is output. This helps with gauging resource consumption.

## Training a model
To train a model, you run the `train.py` program. You will need to specify the location to save the model metadata dictionary, as well as items in the dictionary (hyperparameters, etc.) While training, the model will routinely save its state to a file, specified in the model metadata dictionary. Once the model has saved its state for the first time, you can start making predictions as the model continues to train and periodically updates its state file.

## Making predictions
To make predictions, you run the `predict.py` program. You will need to specifiy the location of the model metadata file, provide a MusicXML prompt file, and specify the number of FFT frames to generate.

## Dependencies
You will need to install the following packages to use this repository:
`numpy`, `pytorch`, `pytorch-cuda` (if you are running on CUDA), `torchaudio`, `regex`, `scikit-learn`

To install on a Python virtualenv, run `pip install music21 numpy pytorch pytorch-cuda torchaudio regex scikit-learn`

## File descriptions
`corpus.py` - Contains functionality for loading all audio files in a given directory.

`dataset.py` - Contains the definition for a `torch.utils.data.Dataset` subclass, `AudioDataset`.

`featurizer.py` - Contains functionality for featurizing audio.

`model_definition.py` - Contains the model definition.

`predictor.py` - Contains functionality for making predictions based on a sequence of audio frames and a given model.

`train.py` - Contains functionality for training models.
