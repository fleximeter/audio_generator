"""
File: predictor.py

This module makes predictions based on an existing model that was saved to file.
You will need to provide the model metadata file name so that it can load important
information about the model, such as the number of layers and the hidden size.
"""

import aus_analyzer
import json
import featurizer
import model_definition
import torch
import torchaudio
from typing import Tuple


def load_model_metadata(file) -> dict:
    """
    Loads model metadata
    :param file: The model metadata file path
    :return: A model metadata dictionary
    """
    try:
        with open(file, "r") as model_json_file:
            return json.loads(model_json_file.read())
    except FileNotFoundError as e:
        print("ERROR: Could not open the model metadata file. Aborting.")
    return None


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
    

def save_predicted_audio(audio, new_audio_frames, start_frame_in_audio, start_frame_for_prediction, fft_size, file_name):
    """
    Saves predicted audio to file
    :param audio: The original audio dictionary
    :param new_audio_frames: A list of new audio frame dictionaries
    :param file_name: The name of the file to save
    """
    # Generate the final output STFT magnitude and phase spectra
    output_mag_spectrum = [audio["magnitude_spectrogram"][start_frame_in_audio:start_frame_for_prediction, :]]
    output_phase_spectrum = [audio["phase_spectrogram"][start_frame_in_audio:start_frame_for_prediction, :]]
    for frame in new_audio_frames:
        output_mag_spectrum.append(torch.unsqueeze(frame["magnitude_spectrum"], 0))
        output_phase_spectrum.append(torch.unsqueeze(frame["phase_spectrum"], 0))
    output_mag_spectrum = torch.cat(output_mag_spectrum, dim=0)
    output_phase_spectrum = torch.cat(output_phase_spectrum, dim=0)
    out_audio = aus_analyzer.irstft(output_mag_spectrum.numpy(), output_phase_spectrum.numpy(), fft_size)
    torchaudio.save(file_name, torch.unsqueeze(torch.from_numpy(out_audio), 0), audio["sample_rate"])


if __name__ == "__main__":
    #######################################################################################
    # YOU WILL NEED TO EDIT THIS MANUALLY
    #######################################################################################

    PROMPT_FILE = "./data/train/217800__minian89__wind_chimes_eq.wav"
    MODEL_METADATA_FILE = "./data/model_12_24_24.json"
    NUM_FRAMES_TO_PREDICT = 50

    # We might not want to use the whole file, so specify start and end STFT frames. Everything after the end frame will be predicted.
    START_FRAME_IN_AUDIO = 0
    START_FRAME_FOR_PREDICTION = 40

    #######################################################################################
    # YOU PROBABLY DON'T NEED TO EDIT ANYTHING BELOW HERE
    #######################################################################################
    
    # Load the model information
    model_metadata = load_model_metadata(MODEL_METADATA_FILE)

    # scaler = featurizer.RobustScaler(model_metadata["median"], model_metadata["iqr"])
    
    # If no errors in loading the files
    if model_metadata is not None:
        audio_file_dict = featurizer.load_audio_file(PROMPT_FILE, model_metadata["fft_size"])
        featurizer.featurize(audio_file_dict, model_metadata["fft_size"])
        feature_matrix = featurizer.make_feature_matrix(audio_file_dict)
        # print("Feature matrix shape:", feature_matrix.shape)
        new_audio_frame_dictionaries = []
        
        # Load the model state dictionary from file
        model = model_definition.LSTMAudio(model_metadata["num_features"], model_metadata["output_size"], 
                                           model_metadata["hidden_size"], model_metadata["num_layers"])
        model.load_state_dict(torch.load(model_metadata["state_dict"], weights_only=True))
        
        # Predict the next N notes
        for i in range(NUM_FRAMES_TO_PREDICT):
            # Make an abbreviated sequence of the proper length for running through the model
            feature_matrix_for_prediction = feature_matrix[feature_matrix.shape[0] - model_metadata["training_sequence_length"]:, :]
            feature_matrix_for_prediction = torch.unsqueeze(feature_matrix_for_prediction, 0)
            # print("Feature matrix for prediction shape:", feature_matrix_for_prediction.shape)
            predicted, hidden = predict_from_sequence(model, feature_matrix_for_prediction)
            predicted = torch.squeeze(torch.detach(predicted))

            # Get a new audio frame dictionary, and append the featurized audio to the feature matrix
            new_audio_frame_dictionaries.append(
                featurizer.make_feature_frame(
                    predicted[:predicted.numel()//2], 
                    predicted[predicted.numel()//2:], 
                    audio_file_dict["sample_rate"], 
                    model_metadata["fft_size"]
                ))
            new_feature_vector = featurizer.make_feature_vector(new_audio_frame_dictionaries[-1])
            new_feature_vector = torch.unsqueeze(new_feature_vector, 0)
            # print("New feature vector shape:", new_feature_vector.shape)
            feature_matrix_for_prediction = torch.cat((feature_matrix_for_prediction[:, 1:, :], new_feature_vector), dim=1)

        # Dump data to file for inspection
        with open("data/outdata.json", "w") as out_json:
            new_mags = []
            new_phases = []
            for frame in new_audio_frame_dictionaries:
                new_mags.append(frame["magnitude_spectrum"].tolist())
                new_phases.append(frame["phase_spectrum"].tolist())
            out_json.write(json.dumps([new_mags, new_phases]))

        # Save the new audio to file
        FILE_NAME = "data/output_12_24_24.wav"
        print(f"Writing {FILE_NAME}")
        save_predicted_audio(audio_file_dict, new_audio_frame_dictionaries, START_FRAME_IN_AUDIO, START_FRAME_FOR_PREDICTION, model_metadata["fft_size"], FILE_NAME)
