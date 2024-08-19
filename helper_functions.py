from datetime import datetime
import h5py
import torch
import glob
import signal
from torch.utils.data import Dataset, Subset
import matplotlib.pyplot as plt
import sys
import os
class NiftiHDF5Dataset(Dataset):
    """
    Creates the datasets.
    """
    def __init__(self, hdf5_file, input_channels, condition=None):
        self.hdf5_file = hdf5_file
        self.input_channels = input_channels
        self.condition = condition
        self.file = h5py.File(hdf5_file, 'r')

    def __len__(self):
        return len(self.file[next(iter(self.input_channels))])

    def __getitem__(self, idx):
        data_tensors = [torch.tensor(self.file[channel][idx], dtype=torch.float32) for channel in self.input_channels]
        if self.condition:
            combined = {}
            cond_tensor = [torch.tensor(self.file[channel][idx], dtype=torch.float32) for channel in self.condition]
            combined['image'] = torch.stack(data_tensors, dim=0) if len(data_tensors) > 1 else data_tensors[0].unsqueeze(0)
            combined['cond'] = torch.stack(cond_tensor, dim=0) if len(cond_tensor) > 1 else cond_tensor[0].unsqueeze(0)
        else:    
            combined = torch.stack(data_tensors, dim=0) if len(data_tensors) > 1 else data_tensors[0].unsqueeze(0)
        return combined

    def __del__(self):
        if self.file:
            self.file.close()
            print(f"HDF5 file {self.hdf5_file} closed.")

def setup_datasets(hdf5_file, input_channels, validation_split=0, condition=None):
    """
    Prepares and splits the dataset into training and validation subsets.
    If validation_split is 0, all data will be used for training, otherwise it will be split.
    """
    dataset = NiftiHDF5Dataset(hdf5_file, input_channels, condition)
    
    if validation_split > 0:
        # Calculate the size and split points for validation and training sets
        dataset_size = len(dataset)
        indices = torch.randperm(dataset_size).tolist()
        validation_size = int(validation_split * dataset_size)
        training_size = dataset_size - validation_size
        train_indices, val_indices = indices[:training_size], indices[training_size:]
        validation_dataset = Subset(dataset, val_indices)
        train_dataset = Subset(dataset, train_indices)
    else:
        # Use entire dataset as the training set if no validation split is specified
        train_dataset = dataset
        validation_dataset = None

    return train_dataset, validation_dataset

def save_checkpoint(epoch, filename, **components):
    """
    Saves checkpoints of models, optimizers, schedulers, losses, and other metrics.
    """
    checkpoint = {'epoch': epoch}
    for key, value in components.items():
        if hasattr(value, 'state_dict'):
            checkpoint[key] = value.state_dict()
        else:
            checkpoint[key] = value
    torch.save(checkpoint, filename)

def load_model(config, model_class, file_prefix, model_prefix, device, path=''):
    """
    Load a model from the most recent checkpoint or model file available.
    """
    model = model_class(**config).to(device)
    file = glob.glob(os.path.join(path, f'{file_prefix}_model*.pth')) + glob.glob(os.path.join(path, f'{file_prefix}_checkpoint*.pth'))
    model_state = torch.load(file[0])
    if list(model_state[f'{model_prefix}_state_dict'].keys())[0].startswith('module.'):
        new_state_dict = {k[len("module."):]: v for k, v in model_state[f'{model_prefix}_state_dict'].items()}
        model.load_state_dict(new_state_dict)
    else:
        new_state_dict = model_state[f'{model_prefix}_state_dict']
        model.load_state_dict(new_state_dict)
    return model

def plot_learning_curves(epoch_losses, val_losses, lr_rates, epoch, val_epochs, save_path,
                            lr_rates_g=None, lr_rates_d=None):
    """
    Plot training and validation loss along with learning rates over epochs, with options for additional loss types and multiple learning rates.
    """
    fig, ax1 = plt.subplots(figsize=(10, 5))
    ax1.set_xlabel('Epochs', fontsize=16)
    ax1.set_ylabel('Loss', color='tab:blue')

    # tick spacing calculation
    tick_spacing = max(1, epoch // 10)
    ax1.set_xticks(range(0, epoch + 1, tick_spacing))

    # Plot training and validation losses
    ax1.plot(range(epoch + 1), epoch_losses, color='tab:blue', label="Train Loss")
    ax1.plot(val_epochs, val_losses, 'b--', label="Validation Loss")
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.legend(loc='upper left')

    # Twin axis for learning rates
    ax2 = ax1.twinx()
    ax2.set_ylabel('Learning Rate', color='tab:green')
    if lr_rates_g and lr_rates_d: # if train_vae
        ax2.plot(val_epochs, lr_rates_g, label='Generator Learning Rate', linestyle='dotted', color='tab:green')
        ax2.plot(val_epochs, lr_rates_d, label='Discriminator Learning Rate', linestyle='dotted', color='tab:red')
    else:
        ax2.plot(val_epochs, lr_rates, color='tab:green', label='Learning Rate')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    ax2.legend(loc='upper right')

    fig.tight_layout()
    plt.title('Learning Curves and Learning Rates', fontsize=20)
    plt.savefig(save_path)
    plt.close()

def print_with_timestamp(message):
    """
    Prints a given message with a timestamp.
    """
    current_time = datetime.now()
    print(f"{current_time} - {message}")

def cleanup(signum, frame, train_dataset, validation_dataset):
    print("Received termination signal. Performing cleanup...")
    del train_dataset
    del validation_dataset
    sys.exit(0)

def subtract_mean(x: torch.Tensor) -> torch.Tensor:
    mean = [0.406, 0.456, 0.485]
    x[:, 0, :, :] -= mean[0]
    x[:, 1, :, :] -= mean[1]
    x[:, 2, :, :] -= mean[2]
    return x

def spatial_average(x: torch.Tensor, keepdim: bool = True) -> torch.Tensor:
    return x.mean([2, 3], keepdim=keepdim)

def get_features(image, radnet):
    # If input has just 1 channel, repeat channel to have 3 channels
    if image.shape[1]:
        image = image.repeat(1, 3, 1, 1)

    # Change order from 'RGB' to 'BGR'
    image = image[:, [2, 1, 0], ...]

    # Subtract mean used during training
    image = subtract_mean(image)

    # Get model outputs
    with torch.no_grad():
        feature_image = radnet(image)
        # flattens the image spatially
        feature_image = spatial_average(feature_image, keepdim=False)

    return feature_image