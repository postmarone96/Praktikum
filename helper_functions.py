from datetime import datetime
import h5py
import torch
import glob
from torch.utils.data import Dataset, Subset

class NiftiHDF5Dataset(Dataset):
    """
    Creates the datasets.
    """
    def __init__(self, hdf5_file, input_channels):
        self.hdf5_file = hdf5_file
        self.input_channels = input_channels

    def __len__(self):
        with h5py.File(self.hdf5_file, 'r') as f:
            return len(f[next(iter(self.input_channels))])

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            data_tensors = [torch.tensor(f[channel][idx], dtype=torch.float32) for channel in self.input_channels]
            combined = torch.stack(data_tensors, dim=0) if len(data_tensors) > 1 else data_tensors[0].unsqueeze(0)
        return combined

def setup_datasets(hdf5_file, input_channels, validation_split):
    """
    Prepares and splits the dataset into training and validation subsets.
    """
    dataset = NiftiHDF5Dataset(hdf5_file, input_channels)
    dataset_size = len(dataset)
    indices = torch.randperm(dataset_size).tolist()

    validation_size = int(validation_split * dataset_size)
    training_size = dataset_size - validation_size
    train_indices, val_indices = indices[:training_size], indices[training_size:]

    train_dataset = Subset(dataset, train_indices)
    validation_dataset = Subset(dataset, val_indices)

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


def save_checkpoint_cn(epoch, controlnet, unet, optimizer, scaler, scheduler, scheduler_lr, epoch_losses, val_losses, val_epochs, lr_rates, filename):
    checkpoint = {
        'epoch': epoch,
        'cn_state_dict': controlnet.module.state_dict(),
        'unet_state_dict': unet.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict':scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scheduler_lr_state_dict': scheduler_lr.state_dict(),
        'epoch_losses': epoch_losses,
        'val_losses': val_losses,
        'val_epochs': val_epochs,
        'lr_rates': lr_rates,
        
    }
    torch.save(checkpoint, filename)

def load_model(config, model_class, file_prefix, model_prefix, device):
    """
    Load a model from the most recent checkpoint or model file available.
    """
    model = model_class(**config).to(device)
    file = glob.glob(f'{file_prefix}_model*.pth') + glob.glob(f'{file_prefix}_checkpoint*.pth')
    model_state = torch.load(file[0])
    if list(unet_model[f'{model_prefix}_state_dict'].keys())[0].startswith('module.'):
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


