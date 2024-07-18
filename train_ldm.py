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
import pickle
import zipfile
import shutil
import json
import signal
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import monai
from helper_functions import *
from monai.utils import first
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

# clear CUDA
torch.cuda.empty_cache()

# Register the signal handler
signal.signal(signal.SIGTERM, cleanup)

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--dataset_file", type=str, required=True)
parser.add_argument("--job", type=str, required=True)
args = parser.parse_args()

# parse parameters form json file
with open('params.json') as json_file:
    config = json.load(json_file)

vae_config = config['VAE']
ldm_config = config['LDM']

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
autoencoderkl = load_model(  config = vae_config['autoencoder'],
                    model_class = AutoencoderKL, 
                    file_prefix = 'vae', 
                    model_prefix = 'autoencoder',
                    device = device)
autoencoderkl = autoencoderkl.half()

# Latent Diffsuion UNet
unet = DiffusionModelUNet(**ldm_config['unet']).to(device)
unet = torch.nn.DataParallel(unet)

# Optimizer
optimizer = torch.optim.Adam(unet.parameters(), lr=ldm_config['optimizer']['lr'])

# Learning Rate Scheduler
scheduler_lr = ReduceLROnPlateau(optimizer, **ldm_config['optimizer']['scheduler'])

# Scaler
scaler = GradScaler()

# DDPM Scheduler
scheduler = DDPMScheduler(**ldm_config['ddpm_scheduler'])

# Upload Parameters from Checkpoint
checkpoint_path = glob.glob('ldm_checkpoint_epoch_*.pth')

if checkpoint_path:
    checkpoint = torch.load(checkpoint_path[0])
    start_epoch = checkpoint['epoch'] + 1
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    unet.module.load_state_dict(checkpoint['unet_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scheduler_lr.load_state_dict(checkpoint['scheduler_lr_state_dict'])
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
        z = autoencoderkl.encode_stage_2_inputs(check_data.to(device))
scale_factor = 1 / torch.std(z)
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

# Other parameters
n_epochs = ldm_config['training']['n_epochs']
val_interval = ldm_config['training']['val_interval']
num_epochs_checkpoints = ldm_config['training']['num_epochs_checkpoints']

# Start training
try:
    for epoch in range(start_epoch, n_epochs):
        unet.train()
        autoencoderkl.eval()
        epoch_loss = 0
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
        progress_bar.set_description(f"Epoch {epoch}")
        for step, batch in progress_bar:
            images = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=True):
                z_mu, z_sigma = autoencoderkl.encode(images)
                z = autoencoderkl.sampling(z_mu, z_sigma)
                noise = torch.randn_like(z).to(device)
                timesteps = torch.randint(0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device).long()
                noise_pred = inferer(
                    inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps, autoencoder_model=autoencoderkl
                )
                loss = F.mse_loss(noise_pred.float(), noise.float())
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
        epoch_losses.append(epoch_loss / (step + 1))

        if epoch % val_interval == 0 and epoch > 0:
            val_epochs.append(epoch)
            unet.eval()
            val_loss = 0
            with torch.no_grad():
                for val_step, batch in enumerate(val_loader, start=1):
                    images = batch.to(device)
                    with autocast(enabled=True):
                        z_mu, z_sigma = autoencoderkl.encode(images)
                        z = autoencoderkl.sampling(z_mu, z_sigma)
                        noise = torch.randn_like(z).to(device)
                        timesteps = torch.randint(
                            0, inferer.scheduler.num_train_timesteps, (z.shape[0],), device=z.device
                        ).long()
                        noise_pred = inferer(
                            inputs=images,
                            diffusion_model=unet,
                            noise=noise,
                            timesteps=timesteps,
                            autoencoder_model=autoencoderkl,
                        )
                        loss = F.mse_loss(noise_pred.float(), noise.float())
                    val_loss += loss.item()
            val_loss /= val_step
            scheduler_lr.step(val_loss)
            lr_unet = optimizer.param_groups[0]['lr']
            lr_rates.append(lr_unet)
            val_losses.append(val_loss)
            print(f"Epoch {epoch} val loss: {val_loss:.4f}")

        if epoch % num_epochs_checkpoints == 0 and epoch > 0:
            save_checkpoint(
                epoch, f'ldm_checkpoint_epoch_{epoch}.pth',
                unet_state_dict = unet.module.state_dict(),
                optimizer_state_dict = optimizer.state_dict(),
                scaler_state_dict = scaler.state_dict(),
                scheduler_state_dict = scheduler.state_dict(),
                scheduler_lr_state_dict = scheduler_lr.state_dict(),
                epoch_losses = epoch_losses,
                val_losses = val_losses,
                scale_factor = scale_factor,
                val_epochs = val_epochs,
                lr_rates = lr_rates
            )

        if epoch > val_interval:
            plot_learning_curves(epoch_losses = epoch_losses,
                                val_losses = val_losses,
                                lr_rates = lr_rates, 
                                epoch = epoch, 
                                val_epochs = val_epochs,
                                lr_rates_g = None, 
                                lr_rates_d = None,
                                save_path = 'LDM_learning_curves.png')
except KeyboardInterrupt:
    cleanup(signal.SIGINT, None, train_dataset, validation_dataset)
    
progress_bar.close()


now = datetime.now()
date_time = now.strftime("%Y%m%d_%H%M")
save_checkpoint(
    epoch, f'ldm_model_{date_time}.pth',
    unet_state_dict = unet.module.state_dict(),
    optimizer_state_dict = optimizer.state_dict(),
    scaler_state_dict = scaler.state_dict(),
    scheduler_state_dict = scheduler.state_dict(),
    scheduler_lr_state_dict = scheduler_lr.state_dict(),
    epoch_losses = epoch_losses,
    val_losses = val_losses,
    scale_factor = scale_factor,
    val_epochs = val_epochs,
    lr_rates = lr_rates
)

# Store generated output  
autoencoderkl = autoencoderkl.to(device).float()
current_working_directory = os.getcwd()
pkl_dir = os.path.join(current_working_directory, 'pkl_dir')
os.makedirs(pkl_dir, exist_ok=True)

number_of_samples = ldm_config['sampling']['number_of_samples']
data_dict = {}
for i in range(number_of_samples):
    unet.eval()
    scheduler.set_timesteps(num_inference_steps=ldm_config['sampling']['num_inference_steps'])
    noise = torch.randn(ldm_config['sampling']['noise_shape'])
    noise = noise.to(device)
    
    with torch.no_grad():
        image, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=unet,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=ldm_config['sampling']['intermediate_steps'],
            autoencoder_model=autoencoderkl,
        )
    # Store intermediates and noise in the dictionary
    data_dict[i] = {'intermediates': intermediates, 'noise': noise}
    # Decode latent representation of the intermediary images
    with torch.no_grad():
        # Concatenating the images for each channel (first two channels in this case) and then all channels together
        concat_channels = [torch.cat([img[:, j, :, :].unsqueeze(1) for img in intermediates], dim=-1) for j in range(len(config['dataset']['input_channels']))]
        concat_all_channels = torch.cat(concat_channels, dim=-2)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, len(config['dataset']['input_channels'])))
    im = ax.imshow(concat_all_channels[0, 0].cpu(), cmap="jet", vmin=0, vmax=1)
    # Remove the axis
    ax.axis('off')
    
    # Add the colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.04)
    cbar.set_label('Intensity', rotation=270, labelpad=15,  verticalalignment='center')
    channel_height = concat_all_channels.size(2) // len(config['dataset']['input_channels'])
    channel_labels = config['dataset']['input_channels']
    for j in range(len(channel_labels)):
        ax.text(-150, channel_height * (0.5 + j), channel_labels[j], fontsize=12, va='center', ha='center')

    plt.savefig(os.path.join(pkl_dir, f'sample_{i}.png'), dpi=300)
    plt.close()

# Save the data dictionary as a pickle file
with open(os.path.join(pkl_dir, 'data_dict.pkl'), 'wb') as f:
    pickle.dump(data_dict, f)


output_filename = os.path.join(current_working_directory, f'samples_{args.job}')

shutil.make_archive(output_filename, 'zip', pkl_dir)

torch.cuda.empty_cache()


