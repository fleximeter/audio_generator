"""
File: predictor.py

This module makes predictions based on an existing model that was saved to file.
You will need to provide the model metadata file name so that it can load important
information about the model, such as the number of layers and the hidden size.
"""

import json
import featurizer
import model_definition
import torch
import torchaudio
from typing import Tuple


def predict_from_sequence(model, sequence) -> Tuple[dict, torch.Tensor]:
    """
    Predicts the next note, based on an existing model and a sequence of notes
    :param model: The model
    :param sequence: The tokenized sequence of notes
    :param training_sequence_max_length: The maximum sequence length the model was trained on.
    This is necessary because the DataLoader will pad sequences that are shorter than the maximum
    length, and the model might not behave as predicted if we don't pad sequences that we use
    as prompts.
    :return: The prediction as a note dictionary, and the hidden states as a tuple
    """
    prediction, hidden = model(sequence, model.init_hidden())
    return prediction, hidden


if __name__ == "__main__":
    #######################################################################################
    # YOU WILL NEED TO EDIT THIS MANUALLY
    #######################################################################################

    PROMPT_FILE = "./data/train/sample.48.Viola.pizz.sulC.ff.C3B3.mono.wav"
    MODEL_METADATA_FILE = "./data/model1.json"
    FRAMES_TO_PREDICT = 25
    START_FRAME = 100

    #######################################################################################
    # YOU PROBABLY DON'T NEED TO EDIT ANYTHING BELOW HERE
    #######################################################################################
    
    # Load the model information
    model_metadata = None
    abort = False

    try:
        with open(MODEL_METADATA_FILE, "r") as model_json_file:
            model_metadata = json.loads(model_json_file.read())
    except FileNotFoundError as e:
        abort = True
        print("ERROR: Could not open the model metadata file. Aborting.")
    
    try:
        audio = featurizer.load_audio_file(PROMPT_FILE)
        # Discard all frames beyond the first N frames
        audio["magnitude_spectrogram"] = audio["magnitude_spectrogram"][:, :, :START_FRAME]
        audio["phase_spectrogram"] = audio["phase_spectrogram"][:, :, :START_FRAME]
    except Exception as e:
        abort = True
        print("ERROR: Could not read the audio prompt file. Aborting.")

    if not abort:
        # Predict only for the top staff
        feature_vectors = featurizer.make_feature_vector(audio)
        new_audio_frames = []
        
        # Load the model state dictionary from file
        model = model_definition.LSTMAudio(model_metadata["num_features"], model_metadata["output_size"], 
                                           model_metadata["hidden_size"], model_metadata["num_layers"])
        model.load_state_dict(torch.load(model_metadata["state_dict"]))
        
        # Predict the next N notes
        for i in range(FRAMES_TO_PREDICT):
            # Make an abbreviated sequence of the proper length for running through the model
            feature_vectors_predict = feature_vectors[:, :, feature_vectors.shape[-1] - model_metadata["training_sequence_length"]:]
            predicted, hidden = predict_from_sequence(model, feature_vectors)
            predicted = torch.squeeze(torch.detach(predicted))
            new_audio_frames.append(featurizer.make_feature_frame(predicted[:predicted.numel()//2], predicted[predicted.numel()//2:], audio["sample_rate"]))
            new_feature_vector = featurizer.make_feature_vector(new_audio_frames[-1])
            feature_vectors = torch.hstack((feature_vectors, new_feature_vector))

        output_mag_spectrum = [audio["magnitude_spectrogram"]]
        output_phase_spectrum = [audio["phase_spectrogram"]]
        for frame in new_audio_frames:
            output_mag_spectrum.append(frame["magnitude_spectrogram"])
            output_phase_spectrum.append(frame["phase_spectrogram"])
        output_mag_spectrum = torch.cat(output_mag_spectrum, dim=2)
        output_phase_spectrum = torch.cat(output_phase_spectrum, dim=2)
        output_complex_spectrum = torch.cos(output_phase_spectrum) * output_mag_spectrum + 1j * torch.sin(output_phase_spectrum) * output_mag_spectrum
        istft = torchaudio.transforms.InverseSpectrogram(featurizer.FFT_SIZE)
        new_audio = istft(output_complex_spectrum)
        torchaudio.save("data/output1.wav", new_audio, audio["sample_rate"])
        