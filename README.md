# Audio Generator

## About
This repository is a generative AI system for producing the next FFT frame in an audio file. It can train models on a collection of audio files. The architecture is the PyTorch LSTM architecture, and it will likely need considerable resources for training if you want an acceptable model. Predictions generate output logits that consist of the magnitude and phase spectrums concatenated.

## Resource needs
The default training device is CUDA, and MPS is the first fallback, with a CPU as the last-level default. When training, the estimated time remaining is output. This helps with gauging resource consumption.

## Setup
You will need to install the following packages to use this repository:
`Cython`, `numpy`, `torch`, `torchaudio`, `regex`, and `soundfile`.

Visit https://pytorch.org/get-started/locally/ for PyTorch installation instructions (this is a good idea if you want to use CUDA).

You will also need to compile the Cython code in the `caus` directory. Run the command `python setup.py build_ext --inplace`.

## Training a model
Install the dependencies listed above, and compile the Cython code, then follow these steps:
1. Specify the location of your audio corpus in the `train.py` file, using the `TRAINING_PATH` variable.
2. Run the `train.py` program. NOTE: Before running, make sure that file locations are correctly specified.

While training, the model will routinely save its state to a file, specified in the model metadata dictionary. Once the model has saved its state for the first time, you can start making predictions as the model continues to train and periodically updates its state file.

## Making predictions
To make predictions, you run the `predictor.py` program. You will need to specify the location of the audio prompt file, as well as the location of the model.

## File descriptions
`corpus.py` - Contains functionality for loading all audio files in a given directory.

`dataset.py` - Contains the definition for a `torch.utils.data.Dataset` subclass, `AudioDataset`.

`featurizer.py` - Contains functionality for featurizing audio.

`model_definition.py` - Contains the model definition.

`predictor.py` - Contains functionality for making predictions based on a sequence of audio frames and a given model.

`train.py` - Contains functionality for training models.

`train_hpc.py` - A modified version of `train.py` for running on the University of Iowa Argon high-performance computing system.

## Other notes
The `caudiopython` modules are included in this repo because of compatibility issues with building Cython packages on the Argon HPC system at the University of Iowa.
