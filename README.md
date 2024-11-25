# Audio Generator

## About
This repository is an in-progress generative AI system for producing the next FFT frame in an audio file.

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
