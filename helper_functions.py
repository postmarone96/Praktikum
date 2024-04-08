from datetime import datetime
import h5py
import torch
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

def save_checkpoint_vae(epoch, autoencoder_model, discriminator_model, optimizer_g, optimizer_d, scheduler_d, scheduler_g, val_recon_losses, epoch_recon_losses, epoch_gen_losses, epoch_disc_losses, intermediary_images, lr_rates_g, lr_rates_d, val_epochs, filename):
    """
    Saves checkpoints of the VAE models, metrics and outputs.
    """
checkpoint = {
    'epoch': epoch,
    'autoencoder_state_dict': autoencoder_model.module.state_dict(),
    'discriminator_state_dict': discriminator_model.module.state_dict(),
    'optimizer_g_state_dict': optimizer_g.state_dict(),
    'optimizer_d_state_dict': optimizer_d.state_dict(),
    'scheduler_d_state_dict': scheduler_d.state_dict(),
    'scheduler_g_state_dict': scheduler_g.state_dict(),
    'val_recon_losses': val_recon_losses,
    'epoch_recon_losses': epoch_recon_losses,
    'epoch_gen_losses': epoch_gen_losses,
    'epoch_disc_losses': epoch_disc_losses,
    'intermediary_images': intermediary_images,
    'lr_rates_g': lr_rates_g,
    'lr_rates_d': lr_rates_d,
    'val_epochs': val_epochs,
}
torch.save(checkpoint, filename)

def save_checkpoint_ldm(epoch, unet, optimizer, scaler, scheduler, scheduler_lr, scale_factor, epoch_losses, val_losses, val_epochs, lr_rates, filename):
    """
    Saves checkpoints of the LDM models, metrics and outputs.
    """
    checkpoint = {
        'epoch': epoch,
        'unet_state_dict': unet.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict':scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scheduler_lr_state_dict': scheduler_lr.state_dict(),
        'epoch_losses': epoch_losses,
        'val_losses': val_losses,
        'scale_factor': scale_factor,
        'val_epochs': val_epochs,
        'lr_rates': lr_rates,
    }
    torch.save(checkpoint, filename)

def print_with_timestamp(message):
    """
    Prints a given message with a timestamp.
    """
    current_time = datetime.now()
    print(f"{current_time} - {message}")