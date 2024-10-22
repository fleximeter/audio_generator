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


def load_file_for_prediction(file, start_frame, end_frame) -> dict:
    """
    Loads an audio file for prediction and featurizes it
    :param file: The file
    :param start_frame: The first frame to keep
    :pram end_frame: The frame after the last frame to keep
    :return: The featurized audio as a dictionary
    """
    try:
        audio = featurizer.load_audio_file(file)
        audio["magnitude_spectrogram"] = audio["magnitude_spectrogram"][:, :, start_frame:end_frame]
        audio["phase_spectrogram"] = audio["phase_spectrogram"][:, :, start_frame:end_frame]
        audio["num_spectrogram_frames"] = end_frame - start_frame
        return audio
    except Exception as e:
        print("ERROR: Could not read the audio prompt file. Aborting.")
    return None
    

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
    

def save_predicted_audio(audio, new_audio_frames, file_name):
    """
    Saves predicted audio to file
    :param audio: The original audio dictionary
    :param new_audio_frames: A list of new audio frame dictionaries
    :param file_name: The name of the file to save
    """
    # Generate the final output STFT magnitude and phase spectra
    output_mag_spectrum = [audio["magnitude_spectrogram"]]
    output_phase_spectrum = [audio["phase_spectrogram"]]
    for frame in new_audio_frames:
        output_mag_spectrum.append(frame["magnitude_spectrogram"])
        output_phase_spectrum.append(frame["phase_spectrogram"])
    output_mag_spectrum = torch.cat(output_mag_spectrum, dim=2)
    output_phase_spectrum = torch.cat(output_phase_spectrum, dim=2)
    
    # Assemble into a complex STFT spectrum
    output_complex_spectrum = torch.cos(output_phase_spectrum) * output_mag_spectrum + 1j * torch.sin(output_phase_spectrum) * output_mag_spectrum
    
    # Save the audio
    istft = torchaudio.transforms.InverseSpectrogram(featurizer.FFT_SIZE)
    new_audio = istft(output_complex_spectrum)
    torchaudio.save(file_name, new_audio, audio["sample_rate"])


if __name__ == "__main__":
    #######################################################################################
    # YOU WILL NEED TO EDIT THIS MANUALLY
    #######################################################################################

    PROMPT_FILE = "./data/train/217800__minian89__wind_chimes_eq.wav"
    MODEL_METADATA_FILE = "./data/model_8_13_24.json"
    NUM_FRAMES_TO_PREDICT = 100
    START_FRAME_FOR_PREDICTION = 30

    #######################################################################################
    # YOU PROBABLY DON'T NEED TO EDIT ANYTHING BELOW HERE
    #######################################################################################
    
    # Load the model information
    model_metadata = load_model_metadata(MODEL_METADATA_FILE)

    # scaler = featurizer.RobustScaler(model_metadata["median"], model_metadata["iqr"])
    audio = load_file_for_prediction(PROMPT_FILE, 0, START_FRAME_FOR_PREDICTION)

    # If no errors in loading the files
    if audio is not None and model_metadata is not None:
        feature_matrix = featurizer.make_feature_vector(audio)
        new_audio_frame_dictionaries = []
        
        # Load the model state dictionary from file
        model = model_definition.LSTMAudio(model_metadata["num_features"], model_metadata["output_size"], 
                                           model_metadata["hidden_size"], model_metadata["num_layers"])
        model.load_state_dict(torch.load(model_metadata["state_dict"]))
        
        # Predict the next N notes
        for i in range(NUM_FRAMES_TO_PREDICT):
            # Make an abbreviated sequence of the proper length for running through the model
            feature_matrix_for_prediction = feature_matrix[:, :, feature_matrix.shape[-1] - model_metadata["training_sequence_length"]:]
            predicted, hidden = predict_from_sequence(model, feature_matrix)
            predicted = torch.squeeze(torch.detach(predicted))
            
            # Get a new audio frame dictionary, and append the featurized audio to the feature matrix
            new_audio_frame_dictionaries.append(featurizer.make_feature_frame(predicted[:predicted.numel()//2], predicted[predicted.numel()//2:], audio["sample_rate"]))
            new_feature_vector = featurizer.make_feature_vector(new_audio_frame_dictionaries[-1])
            feature_matrix = torch.cat((feature_matrix, new_feature_vector), dim=1)

        # Dump data to file for inspection
        with open("data/outdata.json", "w") as out_json:
            new_mags = []
            new_phases = []
            for frame in new_audio_frame_dictionaries:
                new_mags.append(frame["magnitude_spectrogram"].tolist())
                new_phases.append(frame["phase_spectrogram"].tolist())
            out_json.write(json.dumps([new_mags, new_phases]))

        # Save the new audio to file
        save_predicted_audio(audio, new_audio_frame_dictionaries, "data/output_10_22_24.wav")
