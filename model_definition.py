"""
File: model_definition.py

This file contains the neural network definition for the music sequence
generator. At this point it uses a LSTM model and outputs three labels:
pitch class, octave, and quarter length.
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from typing import Tuple

class LSTMAudio(nn.Module):
    """
    A class for making audio LSTM models. It expects 3D tensors for prediction.
    Dimension 1 size: Batch size
    Dimension 2 size: Sequence length
    Dimension 3 size: Number of features
    
    There is one output:
    y: FFT logits
    """
    def __init__(self, input_size, output_size, hidden_size=128, num_layers=1, device="cpu"):
        """
        Initializes the audio LSTM
        :param input_size: The input size
        :param output_size: The number of output FFT bins
        :param hidden_size: The size of the hidden state vector
        :param num_layers: The number of layers to use
        :param device: The device on which the model will be operating
        """
        super(LSTMAudio, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.output = nn.Linear(hidden_size, output_size)
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.device = device

        # Use He initialization to help avoid vanishing gradients
        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.kaiming_uniform_(param, mode="fan_in", nonlinearity="relu")

    def forward(self, x, hidden_states) -> tuple:
        """
        Runs a batch of sequences forward through the model
        :param x: The batch of sequences
        :param hidden_states: A tuple of hidden state matrices
        :return y, hidden: Returns a logit vector and updated hidden states
        """
        output, hidden_states = self.lstm(x, hidden_states)
        logits = self.output(output[:, -1, :])
        return logits, hidden_states
    
    def init_hidden(self, batch_size=1) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes the hidden state
        :param batch_size: The batch size
        :return: Returns a tuple of empty hidden matrices
        """
        return (torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device), 
                torch.zeros((self.num_layers, batch_size, self.hidden_size), device=self.device))
    