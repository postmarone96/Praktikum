# --------------------------------------------------------------------------------
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified by Marouane Hajri on 08.04.2024
# --------------------------------------------------------------------------------

import os
import glob
import argparse
import h5py
import json
import signal
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from helper_functions import *
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, PatchDiscriminator

# clear CUDA
torch.cuda.empty_cache()

# Register the signal handler
signal.signal(signal.SIGTERM, cleanup)

# Parse parameters form json file
with open('params.json') as json_file:
    config = json.load(json_file)

# VAE Config
vae_config = config['VAE']

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_file", type=str, required=True)
args = parser.parse_args()

# Prepare Dataset
train_dataset, validation_dataset = setup_datasets( args.dataset_file, 
                                                    config["dataset"]['input_channels'], 
                                                    config["dataset"]["validation_split"])

train_loader = DataLoader(  train_dataset, 
                            batch_size=config["dataset"]["batch_size"], 
                            shuffle=config["dataset"]["shuffle"], 
                            num_workers=config["dataset"]["num_workers"], 
                            persistent_workers=config["dataset"]["persistent_workers"])

val_loader = DataLoader(    validation_dataset, 
                            batch_size=config["dataset"]["batch_size"], 
                            shuffle=config["dataset"]["shuffle"], 
                            num_workers=config["dataset"]["num_workers"], 
                            persistent_workers=config["dataset"]["persistent_workers"])

device = torch.device("cuda")

# Visual Auto Encoder
autoencoder_config = vae_config['autoencoder']
autoencoderkl = AutoencoderKL(**autoencoder_config).to(device)

# Discriminator
discriminator_config = vae_config['discriminator']
discriminator = PatchDiscriminator(**discriminator_config).to(device)

# DataParallel
autoencoderkl = torch.nn.DataParallel(autoencoderkl)
discriminator = torch.nn.DataParallel(discriminator)

# Optimizer
optimizer_g = torch.optim.Adam(autoencoderkl.parameters(), lr=vae_config['optimizer']['lr_g'])
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=vae_config['optimizer']['lr_d'])

# Scheduler
scheduler_g = ReduceLROnPlateau(optimizer_g, **vae_config['optimizer']['scheduler'])
scheduler_d = ReduceLROnPlateau(optimizer_d, **vae_config['optimizer']['scheduler'])

# Upload Parameters from Checkpoint
checkpoint_path = glob.glob('vae_checkpoint_epoch_*.pth')
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path[0])
    start_epoch = checkpoint['epoch'] + 1
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
else:
    start_epoch = 0
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
num_epochs_checkpoints = vae_config['training']['num_epochs_checkpoints']
autoencoder_warm_up_n_epochs = vae_config['training']['autoencoder_warm_up_n_epochs']
num_example_images = vae_config['training']['num_example_images']

# Start training
try:
    for epoch in range(start_epoch, n_epochs):
        # Set models to training mode
        autoencoderkl.train()
        discriminator.train()

        # Initialize epoch losses
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0

        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in progress_bar:

            images = batch.to(device)

            # Zero the gradients of the generator's optimizer
            optimizer_g.zero_grad(set_to_none=True)

            # Forward pass with gradient scaling
            with autocast(enabled=True):
                # Autoencoder forward pass.
                reconstruction, z_mu, z_sigma = autoencoderkl(images)

                # Calculate reconstruction loss 
                recons_loss = F.l1_loss(reconstruction.float(), images.float())

                # If there are 2 channels in the output, calculate perceptual loss for each and average
                if reconstruction.size(1) == 2:
                    p_loss_1 = perceptual_loss(reconstruction[:, 0, :, :].unsqueeze(1).float(), images[:, 0, :, :].unsqueeze(1).float())
                    p_loss_2 = perceptual_loss(reconstruction[:, 1, :, :].unsqueeze(1).float(), images[:, 1, :, :].unsqueeze(1).float())
                    p_loss = (p_loss_1 + p_loss_2) / 2
                else :
                    # For single-channel outputs, calculate perceptual loss directly
                    p_loss = perceptual_loss(reconstruction.float(), images.float())

                # Calculate KL divergence loss for the latent space regularization
                kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
                kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]

                # Combine losses with respective weights
                loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)
                
                # Add adversarial loss after warm up
                if epoch > autoencoder_warm_up_n_epochs:
                    logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                    generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                    loss_g += adv_weight * generator_loss
            
            # Backpropagate and update generator (autoencoder) parameters
            scaler_g.scale(loss_g).backward()
            scaler_g.step(optimizer_g)
            scaler_g.update()

            # Train discriminator if in the adversarial phase (post-warm-up)
            if epoch > autoencoder_warm_up_n_epochs:
                with autocast(enabled=True):
                    # Zero the gradients of the discriminator's optimizer
                    optimizer_d.zero_grad(set_to_none=True)

                    # Calculate discriminator loss for both fake and real samples
                    logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                    loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                    logits_real = discriminator(images.contiguous().detach())[-1]
                    loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                    discriminator_loss = (loss_d_fake + loss_d_real) * 0.5
                    loss_d = adv_weight * discriminator_loss
                
                # Backpropagate and update discriminator parameters
                scaler_d.scale(loss_d).backward()
                scaler_d.step(optimizer_d)
                scaler_d.update()

            # Update epoch losses for logging
            epoch_loss += recons_loss.item()
            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

            # Update progress bar with current loss values
            progress_bar.set_postfix({
                    "recons_loss": epoch_loss / (step + 1),
                    "gen_loss": gen_epoch_loss / (step + 1),
                    "disc_loss": disc_epoch_loss / (step + 1),
                })

        # Record average losses for the epoch
        epoch_recon_losses.append(epoch_loss / (step + 1))
        epoch_gen_losses.append(gen_epoch_loss / (step + 1))
        epoch_disc_losses.append(disc_epoch_loss / (step + 1))

        # Perform validation and learning rate adjustment
        if epoch % val_interval == 0 and epoch > 0:
            val_epochs.append(epoch)
            autoencoderkl.eval()
            val_loss = 0
            
            # Perform validation with disabled gradient for memory saving
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch.to(device)
                    with autocast(enabled=True):
                        reconstruction, z_mu, z_sigma = autoencoderkl(images)
                        if val_step == 1:
                            intermediary_images.append(reconstruction[:num_example_images, 0])
                        recons_loss = F.l1_loss(images.float(), reconstruction.float())
                    val_loss += recons_loss.item()
            
            # Compute the average validation loss
            val_loss /= val_step

            # Adjust the learning rate of both optimizers based on the average validation loss
            scheduler_g.step(val_loss)  
            scheduler_d.step(val_loss)

            # Retrieve and record the current learning rates and Validartion recon_loss
            lr_g = optimizer_g.param_groups[0]['lr']
            lr_d = optimizer_d.param_groups[0]['lr']
            lr_rates_g.append(lr_g)
            lr_rates_d.append(lr_d)
            val_recon_losses.append(val_loss)

        # Save checkpoints at specified epoch interval
        if epoch % num_epochs_checkpoints == 0 and epoch > 0:
            save_checkpoint(
                epoch, f'vae_checkpoint_epoch_{epoch}.pth',
                autoencoder_state_dict=autoencoder_model.module.state_dict(),
                discriminator_state_dict=discriminator_model.module.state_dict(),
                optimizer_g_state_dict=optimizer_g.state_dict(),
                optimizer_d_state_dict=optimizer_d.state_dict(),
                scheduler_d_state_dict=scheduler_d.state_dict(),
                scheduler_g_state_dict=scheduler_g.state_dict(),
                val_recon_losses=val_recon_losses,
                epoch_recon_losses=epoch_recon_losses,
                epoch_gen_losses=epoch_gen_losses,
                epoch_disc_losses=epoch_disc_losses,
                intermediary_images=intermediary_images,
                lr_rates_g=lr_rates_g,
                lr_rates_d=lr_rates_d,
                val_epochs=val_epochs
            )
        # Plot learning curves and learning rates
        if epoch > val_interval:
            plot_learning_curves(epoch_losses = epoch_recon_losses,
                                val_losses = val_recon_losses,
                                lr_rates = None, 
                                epoch = epoch, 
                                val_epochs = val_epochs,
                                lr_rates_g = lr_rates_g, 
                                lr_rates_d = lr_rates_d,
                                save_path = 'VAE_learning_curves.png')

except KeyboardInterrupt:
    cleanup(signal.SIGINT, None, train_dataset, validation_dataset)

progress_bar.close()

# Save the last epoch as final model.pth
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M")  
save_checkpoint(
    epoch, f'vae_model_{date_time}.pth',
    autoencoder_state_dict=autoencoder_model.module.state_dict(),
    discriminator_state_dict=discriminator_model.module.state_dict(),
    optimizer_g_state_dict=optimizer_g.state_dict(),
    optimizer_d_state_dict=optimizer_d.state_dict(),
    scheduler_d_state_dict=scheduler_d.state_dict(),
    scheduler_g_state_dict=scheduler_g.state_dict(),
    val_recon_losses=val_recon_losses,
    epoch_recon_losses=epoch_recon_losses,
    epoch_gen_losses=epoch_gen_losses,
    epoch_disc_losses=epoch_disc_losses,
    intermediary_images=intermediary_images,
    lr_rates_g=lr_rates_g,
    lr_rates_d=lr_rates_d,
    val_epochs=val_epochs
)
# Clean up
del discriminator
del perceptual_loss
torch.cuda.empty_cache()
