"""
File: train_hpc.py

This module trains the music sequence generator. You can either train a model from
scratch, or you can choose to continue training a model that was previously saved
to disk. The training function will output status messages and save periodically.
"""

import dataset
import datetime
import json
import featurizer
import model_definition
import os
import torch
import torch.distributed
import torch.multiprocessing as mp
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.optim as optim
from torch.utils.data import DataLoader


def setup_parallel(rank, world_size):
    torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)


def cleanup():
    torch.distributed.destroy_process_group()


def train_sequences(model, dataloader, loss_fn, optimizer, num_epochs, status_interval, 
                    save_interval, model_metadata, device="cpu") -> None:
    """
    Trains the model. This training function expects a DataLoader which will feed it batches
    of sequences in randomized order. The DataLoader takes care of serving up labels as well.
    This function will output a routine status message with loss and estimated time remaining.
    The model state is routinely saved to disk, so you can use it while it is training.
         
    :param model: The model to train
    :param dataloader: The dataloader
    :param loss_fn: The loss function
    :param optimizer: The optimizer
    :param num_epochs: The number of epochs for training
    :param status_interval: How often (in epochs) to print an update message
    :param save_interval: Save to disk every N epochs
    :param model_metadata: The model metadata
    :param device: The device that is being used for training
    """

    training_stats = {
        "time": datetime.datetime.now(), 
        "total_time": datetime.timedelta(), 
        "last_epoch_duration": datetime.timedelta(),
        "seconds_remaining": 0.0,
        "average_loss_this_epoch": 0.0,
        "last_completed_epoch": 0,
        "total_epochs": num_epochs,
        "status_interval": status_interval
    }

    for epoch in range(num_epochs):        
        # Track the total loss and number of batches processed this epoch
        total_loss_this_epoch = 0
        num_batches_this_epoch = 0

        # Iterate through each batch in the dataloader. The batch will have 3 labels per sequence.
        for x, y in dataloader:
            optimizer.zero_grad()

            # Prepare for running through the net
            x = x.to(device)
            y = y.to(device)
            hidden = model.init_hidden(x.shape[0])
            
            for i in range(x.shape[0]):
                found = False
                for j in range(x.shape[1]):
                    for k in range(x.shape[2]):
                        if torch.isnan(y[i, j]):
                            print("NAN in X", num_batches_this_epoch)
                            found = True
                            break
                    if found:
                        break
                if found:
                    break
            
            
            for i in range(y.shape[0]):
                found = False
                for j in range(y.shape[1]):
                    if torch.isnan(y[i, j]):
                        print("NAN in Y", num_batches_this_epoch)
                        found = True
                        break
                if found:
                    break
            # This is necessary because of the size of the output labels.
            model.lstm.flatten_parameters()

            # Run the current batch through the net
            output, _ = model(x, hidden)
            for i in range(output.shape[0]):
                found = False
                for j in range(output.shape[1]):
                    if torch.isnan(output[i, j]):
                        print("NAN in output", num_batches_this_epoch)
                        found = True
                        break
                if found:
                    print(output[0, :5])
                    break
                        
            # Compute loss
            loss = loss_fn(output, y)
            total_loss_this_epoch += loss.item()
            num_batches_this_epoch += 1
            
            # Update weights. Clip gradients to help avoid exploding and vanishing gradients.
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        
        # Generate status. The status consists of the epoch number, average loss, epoch completion
        # time, epoch duration (MM:SS), and estimated time remaining (HH:MM:SS).
        training_stats["last_completed_epoch"] = epoch
        time_new = datetime.datetime.now()
        training_stats["last_epoch_duration"] = time_new - training_stats["time"]
        training_stats["total_time"] += training_stats["last_epoch_duration"]
        training_stats["time"] = time_new
        training_stats["average_loss_this_epoch"] = round(total_loss_this_epoch / num_batches_this_epoch, 4)
        status_output(training_stats)

        # Save to disk if it is the right epoch interval
        if epoch % save_interval == save_interval - 1:
            model_metadata["loss"] = training_stats["average_loss_this_epoch"]
            with open(FILE_NAME, "w") as model_json_file:
                model_json_file.write(json.dumps(model_metadata))
            torch.save(model.state_dict(), model_metadata["state_dict"])
            print("Saved to disk\n")


def status_output(training_stats):
    """
    Outputs training status
    :param training_stats: A dictionary with training statistics
    """
    seconds_remaining = int((training_stats["total_time"].seconds / (training_stats["last_completed_epoch"] + 1)) * \
                            (training_stats["total_epochs"] - training_stats["last_completed_epoch"] - 1))
    
    status_message = "----------------------------------------------------------------------\n" + \
                     "epoch {0:<4}\nloss: {1:<6} | completion time: {2} | epoch duration (MM:SS): " + \
                     "{3:02}:{4:02}\nest. time remaining (HH:MM:SS): {5:02}:{6:02}:{7:02}\n"
    status_message = status_message.format(
        training_stats["last_completed_epoch"] + 1, training_stats["average_loss_this_epoch"], 
        training_stats["time"].strftime("%m-%d %H:%M:%S"), 
        training_stats["last_epoch_duration"].seconds // 60, 
        training_stats["last_epoch_duration"].seconds % 60, 
        seconds_remaining // (60 ** 2), 
        seconds_remaining // 60 % 60, 
        seconds_remaining % 60
    )
    
    # Output status
    if training_stats["status_interval"] is not None and \
        training_stats["last_completed_epoch"] % training_stats["status_interval"] == training_stats["status_interval"] - 1:
        print(status_message)


if __name__ == "__main__":
    ROOT_PATH = "/Users/jmartin50/audio_generator"
    ROOT_PATH = "."    
    TRAINING_PATH = os.path.join(ROOT_PATH, "data/train")    # The path to the training corpus
    FILE_NAME = os.path.join(ROOT_PATH, "data/model1.json")  # The path to the model metadata JSON file
    RETRAIN = False                                          # Whether or not to continue training the same model
    NUM_EPOCHS = 100                                         # The number of epochs to train
    LEARNING_RATE = 0.00001                                    # The model learning rate
    NUM_DATALOADER_WORKERS = 4                               # The number of workers for the dataloader
    PRINT_UPDATE_INTERVAL = 1                                # The epoch interval for printing training status
    MODEL_SAVE_INTERVAL = 10                                 # The epoch interval for saving the model
    model_metadata = None
    
    # The model metadata - load it from file if it exists already
    if (RETRAIN and not os.path.exists(FILE_NAME)) or not RETRAIN:
        model_metadata = {
            "model_name": "audio",
            "path": FILE_NAME,
            "training_sequence_length": 20,
            "num_layers": 2,
            "hidden_size": 1024,
            "batch_size": 100,
            "state_dict": os.path.join(ROOT_PATH, "data/audio_sequencer_1.pth"),
            "num_features": featurizer.NUM_FEATURES,
            "output_size": featurizer.FFT_SIZE + 2,
            "loss": None,
            "mean": None,
            "iqr": None
        }
    else:
        with open(FILE_NAME, "r") as model_json_file:
            model_metadata = json.loads(model_json_file.read())
    
    # Load the dataset
    print("Loading dataset...")
    sequence_dataset = dataset.AudioDataset(TRAINING_PATH, model_metadata["training_sequence_length"], 
                                            model_metadata["mean"], model_metadata["iqr"])
    dataloader = DataLoader(sequence_dataset, model_metadata["batch_size"], True, 
                            num_workers=NUM_DATALOADER_WORKERS)
    
    # Save the model metadata if it is new. We need to store the mean and IQR, so
    # we couldn't save the model metadata until after making the dataset.
    if model_metadata["mean"] is None or model_metadata["iqr"] is None:
        model_metadata["mean"] = float(sequence_dataset.mean)
        model_metadata["iqr"] = float(sequence_dataset.iqr)
        with open(FILE_NAME, "w") as model_json_file:
            model_json_file.write(json.dumps(model_metadata))
        print("Dataset loaded.")
    
    # Prefer CUDA if available, otherwise MPS (if on Apple), or CPU as a last-level default
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device {device}")
    
    # Load and prepare the model. If retraining the model, we will need to load the
    # previous state dictionary so that we aren't training from scratch.
    model = model_definition.LSTMAudio(model_metadata["num_features"], model_metadata["output_size"], 
                                       model_metadata["hidden_size"], model_metadata["num_layers"], device).to(device)
    if RETRAIN:
        print(f"Retraining model from state dict {model_metadata['state_dict']}")
        model.load_state_dict(torch.load(model_metadata["state_dict"]))
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # Train the model
    print(f"Training for {NUM_EPOCHS} epochs...")
    train_sequences(model, dataloader, loss_fn, optimizer, NUM_EPOCHS, PRINT_UPDATE_INTERVAL, MODEL_SAVE_INTERVAL, model_metadata, device)
    