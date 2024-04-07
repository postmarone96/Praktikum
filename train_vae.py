import os
import glob
import argparse
import h5py
import json
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import monai
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

# clear CUDA
torch.cuda.empty_cache()

with open('model_config.json') as json_file:
    config = json.load(json_file)
    vae_config = json.load(f)['vae']

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_file", type=str, required=True)
args = parser.parse_args()


def print_with_timestamp(message):
    """
    prints a given message with a timestamp
    """
    current_time = datetime.now()
    print(f"{current_time} - {message}")


print_with_timestamp("Starting the script")

def save_checkpoint_vae(epoch, autoencoder_model, discriminator_model, optimizer_g, optimizer_d, scheduler_d, scheduler_g, val_recon_losses, epoch_recon_losses, epoch_gen_losses, epoch_disc_losses, intermediary_images, lr_rates_g, lr_rates_d, val_epochs, filename):
    """
    saves checkpoints of the desired models, metrics and outputs
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


class NiftiHDF5Dataset(Dataset):
    def __init__(self, hdf5_file, input_channels):
        self.hdf5_file = hdf5_file
        self.input_channels = input_channels
        
    def __len__(self):
        with h5py.File(self.hdf5_file, 'r') as f:
            return len(f[next(iter(self.input_channels))]) 

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            data_tensors = [torch.tensor(f[channel][idx]) for channel in self.input_channels]
            combined = torch.stack(data_tensors, dim=0) if len(data_tensors) > 1 else data_tensors[0].unsqueeze(0)
        return combined

vae_best_val_loss = float('inf')
ldm_best_val_loss = float('inf')

print_with_timestamp("Loading data")
dataset = NiftiHDF5Dataset(args.output_file, config['input_channels'])
validation_split = config["dataset"]["validation_split"]
dataset_size = len(dataset)
validation_size = int(validation_split * dataset_size)
training_size = dataset_size - validation_size
indices = torch.randperm(len(dataset))
train_indices = indices[:training_size]
val_indices = indices[training_size:]

train_dataset = Subset(dataset, train_indices)
validation_dataset = Subset(dataset, val_indices)

print_with_timestamp("Splitting data for training and validation")
train_loader = DataLoader(train_dataset, 
                            batch_size=config["dataset"]["batch_size"], 
                            shuffle=config["dataset"]["shuffle"], 
                            num_workers=config["dataset"]["num_workers"], 
                            persistent_workers=config["dataset"]["persistent_workers"])

val_loader = DataLoader(validation_dataset, 
                            batch_size=config["dataset"]["batch_size"], 
                            shuffle=config["dataset"]["shuffle"], 
                            num_workers=config["dataset"]["num_workers"], 
                            persistent_workers=config["dataset"]["persistent_workers"])

print_with_timestamp("Setting up device and models")
device = torch.device("cuda")

print_with_timestamp("AutoEncoder setup")

# Before the training loop:
start_epoch = 0
checkpoint_path = glob.glob('vae_checkpoint_epoch_*.pth')

# Model Initialization using the 'vae' section of the configuration
autoencoder_config = config['model']['autoencoder']
autoencoderkl = AutoencoderKL(**autoencoder_config).to(device)

discriminator_config = vae_config['model']['discriminator']
discriminator = PatchDiscriminator(**discriminator_config).to(device)

# DataParallel
autoencoderkl = torch.nn.DataParallel(autoencoderkl)
discriminator = torch.nn.DataParallel(discriminator)

# Optimizer Initialization
optimizer_g = Adam(autoencoderkl.parameters(), lr=vae_config['optimizer']['lr_g'])
optimizer_d = Adam(discriminator.parameters(), lr=vae_config['optimizer']['lr_d'])

# Scheduler Initialization
scheduler_g = ReduceLROnPlateau(optimizer_g, 'min', **vae_config['optimizer']['scheduler']['reduce_on_plateau'])
scheduler_d = ReduceLROnPlateau(optimizer_d, 'min', **vae_config['optimizer']['scheduler']['reduce_on_plateau'])

# Upload Parameters from Checkpoint
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path[0])
    start_epoch = checkpoint['epoch'] + 1  # because we start the next epoch
    autoencoderkl.module.load_state_dict(checkpoint['autoencoder_state_dict'])
    discriminator.module.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    scheduler_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    scheduler_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    val_recon_losses = checkpoint['val_recon_losses']
    epoch_recon_losses = checkpoint['epoch_recon_losses']
    epoch_gen_losses = checkpoint['epoch_gen_losses']
    epoch_disc_losses = checkpoint['epoch_disc_losses']
    intermediary_images = checkpoint['intermediary_images']
    val_epochs = checkpoint['val_epochs']
    lr_rates_g = checkpoint['lr_rates_g']
    lr_rates_d = checkpoint['lr_rates_d']
    print_with_timestamp(f"Resuming from epoch {start_epoch}...")
else:
    val_recon_losses = []
    epoch_recon_losses = []
    epoch_gen_losses = []
    epoch_disc_losses = []
    intermediary_images = []
    val_epochs = []
    lr_rates_g = []
    lr_rates_d = []

# Losses and other parameters
adv_loss = PatchAdversarialLoss(criterion=vae_config['loss']['adv_loss'])
adv_weight = vae_config['loss']['adv_weight']
perceptual_loss = PerceptualLoss(spatial_dims=vae_config['loss']['spatial_dims'], network_type=vae_config['loss']['perceptual_loss']).to(device)
perceptual_weight = vae_config['loss']['perceptual_weight']
kl_weight = vae_config['loss']['kl_weight']
scaler_g = torch.cuda.amp.GradScaler()
scaler_d = torch.cuda.amp.GradScaler()
n_epochs = vae_config['training']['n_epochs']
val_interval = vae_config['training']['val_interval']
autoencoder_warm_up_n_epochs = vae_config['training']['autoencoder_warm_up_n_epochs']
num_example_images = vae_config['training']['num_example_images']


print_with_timestamp("Start setting")
for epoch in range(start_epoch, n_epochs):
    autoencoderkl.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch.to(device)
        optimizer_g.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = autoencoderkl(images)

            recons_loss = F.l1_loss(reconstruction.float(), images.float())
            if reconstruction.size(1) == 2:
                p_loss_1 = perceptual_loss(reconstruction[:, 0, :, :].unsqueeze(1).float(), images[:, 0, :, :].unsqueeze(1).float())
                p_loss_2 = perceptual_loss(reconstruction[:, 1, :, :].unsqueeze(1).float(), images[:, 1, :, :].unsqueeze(1).float())
                p_loss = (p_loss_1 + p_loss_2) / 2
            else :
                p_loss = perceptual_loss(reconstruction.float(), images.float())
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()
        
        if epoch > autoencoder_warm_up_n_epochs:
            with autocast(enabled=True):
                optimizer_d.zero_grad(set_to_none=True)

                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()
            
        epoch_loss += recons_loss.item()
        if epoch > autoencoder_warm_up_n_epochs:
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )
    epoch_recon_losses.append(epoch_loss / (step + 1))
    epoch_gen_losses.append(gen_epoch_loss / (step + 1))
    epoch_disc_losses.append(disc_epoch_loss / (step + 1))

    if epoch % val_interval == 0 and epoch > 0:
        val_epochs.append(epoch)
        autoencoderkl.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch.to(device)
                with autocast(enabled=True):
                    reconstruction, z_mu, z_sigma = autoencoderkl(images)
                    # Get the first reconstruction from the first validation batch for visualisation purposes
                    if val_step == 1:
                        intermediary_images.append(reconstruction[:num_example_images, 0])
                    recons_loss = F.l1_loss(images.float(), reconstruction.float())
                val_loss += recons_loss.item()
        val_loss /= val_step
        scheduler_g.step(val_loss)  
        scheduler_d.step(val_loss)
        lr_g = optimizer_g.param_groups[0]['lr']
        lr_d = optimizer_d.param_groups[0]['lr']
        lr_rates_g.append(lr_g)
        lr_rates_d.append(lr_d)
        val_recon_losses.append(val_loss)
        
    if epoch % 5 == 0 and epoch > 0:
        save_checkpoint_vae(epoch, autoencoderkl, discriminator, optimizer_g, optimizer_d, scheduler_d, 
                            scheduler_g, val_recon_losses, epoch_recon_losses, epoch_gen_losses, epoch_disc_losses, 
                            intermediary_images, lr_rates_g, lr_rates_d, val_epochs, f'vae_checkpoint_epoch_{epoch}.pth')
        if val_loss < vae_best_val_loss:
            vae_best_val_loss = val_loss
            save_checkpoint_vae(epoch, autoencoderkl, discriminator, optimizer_g, optimizer_d, 
                                scheduler_d, scheduler_g, val_recon_losses, epoch_recon_losses, 
                                epoch_gen_losses, epoch_disc_losses, intermediary_images, lr_rates_g, lr_rates_d, val_epochs, 'vae_best_checkpoint.pth')
    
    if epoch > val_interval:
        fig, ax1 = plt.subplots(figsize=(10, 5))

        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax1.set_xticks(range(0, epoch + 1, 10))
        ax1.plot(range(epoch + 1), epoch_recon_losses, label='Training Reconstruction Loss', color='tab:blue')
        ax1.plot(val_epochs, val_recon_losses, label='Validation Reconstruction Loss', linestyle='dashed', color='tab:orange')
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.legend(loc='upper left')

        # Create the line that will go on the right y-axis
        ax2 = ax1.twinx()
        ax2.set_ylabel('Learning Rate', color='tab:green')
        ax2.plot(val_epochs, lr_rates_g, label='Generator Learning Rate', linestyle='dotted', color='tab:green')
        ax2.plot(val_epochs, lr_rates_d, label='Discriminator Learning Rate', linestyle='dotted', color='tab:red')
        ax2.tick_params(axis='y', labelcolor='tab:green')
        ax2.legend(loc='upper right')

        fig.tight_layout()
        plt.title('Learning Curves and Learning Rates')
        plt.savefig('VAE_learning_curves.png')
        plt.close()
    
progress_bar.close()

# Get current date and time
now = datetime.now()
# Format date and time
date_time = now.strftime("%Y%m%d_%H%M")  # I replaced ":" with "M" to avoid file naming issues
# Use date_time string in file name
save_checkpoint_vae(epoch, autoencoderkl, discriminator, optimizer_g, optimizer_d, scheduler_d, 
                            scheduler_g, val_recon_losses, epoch_recon_losses, epoch_gen_losses, epoch_disc_losses, 
                            intermediary_images, lr_rates_g, lr_rates_d, val_epochs, f'vae_model_{date_time}.pth')
del discriminator
del perceptual_loss
torch.cuda.empty_cache()
