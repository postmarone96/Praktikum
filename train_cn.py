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
import argparse
import glob
import json
from datetime import datetime
import numpy as np
import torch
import monai
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from helper_functions import *
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from generative.inferers import DiffusionInferer, ControlNetDiffusionInferer
from generative.networks.nets import DiffusionModelUNet, ControlNet, AutoencoderKL
from generative.networks.schedulers import DDPMScheduler

# clear CUDA
torch.cuda.empty_cache()

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_file", type=str, required=True)
args = parser.parse_args()

# parse parameters form json file
with open('params.json') as json_file:
    config = json.load(json_file)

vae_config = config['VAE']
ldm_config = config['LDM']
cn_config = = config['CN']

# Prepare Dataset
train_dataset, validation_dataset = setup_datasets( args.dataset_file, 
                                                    config['input_channels'],
                                                    config['dataset']['condition'],
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
autoencoderkl = load_model(  config = vae_config['autoencoder'],
                    model_class = AutoencoderKL, 
                    file_prefix = 'vae', 
                    model_prefix = 'autoencoder',
                    device = device)

# Latent Diffsuion UNet
unet = load_model(  config = ldm_config['unet'],
                    model_class = DiffusionModelUNet, 
                    file_prefix = 'ldm', 
                    model_prefix = 'unet',
                    device = device)
unet = torch.nn.DataParallel(unet)

# ControlNet
controlnet = ControlNet(**cn_config['cn']).to(device)

# Copy weights from the DM to the controlnet
controlnet.load_state_dict(unet.module.state_dict(), strict=False)
controlnet = torch.nn.DataParallel(controlnet)

# DDPM Scheduler
scheduler = DDPMScheduler(**cn_config['ddpm_scheduler'])

# Scaler
scaler = GradScaler()

# Lock UNet weights
for p in unet.parameters():
    p.requires_grad = False

# Optimizer
optimizer = torch.optim.Adam(params=controlnet.parameters(), lr=cn_config['optimizer']['lr'])

#Learning Rate Scheduler
scheduler_lr = ReduceLROnPlateau(optimizer, **cn_config['optimizer']['scheduler'])

# Upload Parameters from Checkpoint
checkpoint_path = glob.glob('cn_checkpoint_epoch_*.pth')
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path[0])
    start_epoch = checkpoint['epoch'] + 1
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    controlnet.module.load_state_dict(checkpoint['cn_state_dict'])
    unet.module.load_state_dict(checkpoint['unet_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler_lr.load_state_dict(checkpoint['scheduler_lr_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch_losses = checkpoint['epoch_losses']
    val_losses = checkpoint['val_losses']
    val_epochs = checkpoint['val_epochs']
    lr_rates = checkpoint['lr_rates']
else:
    start_epoch = 0
    epoch_losses = []
    val_losses = []
    val_epochs = []
    lr_rates = []

# Inferer initialization
check_data = next(iter(train_loader))
with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(check_data['image'].to(device))
scale_factor = 1 / torch.std(z)
controlnet_inferer = ControlNetDiffusionInferer(scheduler)
inferer = DiffusionInferer(scheduler)

# Training loop
n_epochs = cn_config['training']['n_epochs']
val_interval = cn_config['training']['val_interval']
num_epochs_checkpoints = cn_config['training']['num_epochs_checkpoint']

# Start training
for epoch in range(start_epoch, n_epochs):
    # Set models to training mode
    controlnet.train()

    # Initialize epoch losses
    epoch_loss = 0

    # Display progress bar
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")

    for step, batch in progress_bar:

        images = batch["image"].to(device)
        masks = batch["cond"].to(device)

        #  Zero the gradients of the optimizer
        optimizer.zero_grad(set_to_none=True)

        # Forward pass with gradient scaling
        with autocast(enabled=True):
            # Input encoding
            with torch.no_grad():
                e = autoencoderkl.encode_stage_2_inputs(images) * scale_factor
            
            # Generate random noise
            noise = torch.randn_like(e).to(device)
            
            # Generate random timesteps for diffusion model conditioning
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (e.shape[0],), device=e.device
            ).long()

            # ControlNet foward pass
            noise_pred = controlnet_inferer(inputs=e,
                                    diffusion_model=unet,
                                    controlnet=controlnet,
                                    noise=noise,
                                    timesteps=timesteps,
                                    cn_cond=masks,
            )

            # Calculate MSE loss 
            loss = F.mse_loss(noise_pred.float(), noise.float())

        # Backpropagate and update weights
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # Append Losses
        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_losses.append(epoch_loss / (step + 1))

    # Validation loop
    if epoch % val_interval == 0 and epoch > 0:
        val_epochs.append(epoch)
        
        # Set Controlnet in eval mode
        controlnet.eval()

        val_epoch_loss = 0
        for step, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            masks = batch["cond"].to(device)
            with torch.no_grad():
                with autocast(enabled=True):
                    # Input encoding
                    e = autoencoderkl.encode_stage_2_inputs(images) * scale_factor

                    # Random noise generation
                    noise = torch.randn_like(e).to(device)

                    # Generate random timesteps for diffusion model conditioning
                    timesteps = torch.randint(
                        0, controlnet_inferer.scheduler.num_train_timesteps, (e.shape[0],), device=e.device
                    ).long()

                    # Controlnet foward pass
                    noise_pred = controlnet_inferer(inputs=e,
                                    diffusion_model=unet,
                                    controlnet=controlnet,
                                    noise=noise,
                                    timesteps=timesteps,
                                    cn_cond=masks,
                    )

                    # MSE loss calculation
                    val_loss = F.mse_loss(noise_pred.float(), noise.float())

            val_epoch_loss += val_loss.item()
            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
            print(val_loss)
            break # Break after first batch of validation data
        
        # Adjust learning rate 
        scheduler_lr.step(val_epoch_loss / (step + 1)) 

        # Record learning rate and validaion losses
        lr_unet = optimizer.param_groups[0]['lr']
        lr_rates.append(lr_unet)
        val_losses.append(val_epoch_loss / (step + 1))

    # Save checkpoints at specified epoch interval
    if (epoch % num_epochs_checkpoints == 0 and epoch > 0 and epoch < 100) or (epoch % 20 == 0 and epoch > 100 and num_epochs_checkpoints <= 20):
        save_checkpoint(
            epoch, f'cn_checkpoint_epoch_{epoch}.pth',
            cn_state_dict = controlnet.module.state_dict(),
            unet_state_dict = unet.module.state_dict(),
            optimizer_state_dict= optimizer.state_dict(),
            scaler_state_dict = scaler.state_dict(),
            scheduler_state_dict = scheduler.state_dict(),
            scheduler_lr_state_dict = scheduler_lr.state_dict(),
            epoch_losses = epoch_losses,
            val_losses = val_losses,
            val_epochs = val_epochs,
            lr_rates = lr_rates
        )
    # Plot learning curves and learning rates    
    if epoch > val_interval:
        plot_learning_curves(epoch_losses = epoch_losses,
                    val_losses = val_losses,
                    lr_rates = lr_rates, 
                    epoch = epoch, 
                    val_epochs = val_epochs,
                    lr_rates_g = None, 
                    lr_rates_d = None,
                    save_path = 'CN_learning_curves.png')
progress_bar.close()

# Save last checkpoint as model
now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M")
save_checkpoint(
    epoch, f'cn_model_{date_time}.pth',
    cn_state_dict = controlnet.module.state_dict(),
    unet_state_dict = unet.module.state_dict(),
    optimizer_state_dict= optimizer.state_dict(),
    scaler_state_dict = scaler.state_dict(),
    scheduler_state_dict = scheduler.state_dict(),
    scheduler_lr_state_dict = scheduler_lr.state_dict(),
    epoch_losses = epoch_losses,
    val_losses = val_losses,
    val_epochs = val_epochs,
    lr_rates = lr_rates
)

# clean up
torch.cuda.empty_cache()
