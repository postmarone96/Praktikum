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



device = torch.device("cuda")

# Visual Auto Encoder
autoencoderkl = load_model( config = vae_config['autoencoder'],
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

# ControlNet
controlnet = load_model(config = **cn_config['cn'],
                        model_class = ControlNet, 
                        file_prefix = 'cn', 
                        model_prefix = 'controlnet',
                        device = device)

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
checkpoint_paths = glob.glob('cn_checkpoint_epoch_*.pth') + glob.glob('cn_model*.pth')
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path[0])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler_lr.load_state_dict(checkpoint['scheduler_lr_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

# Inferer initialization
check_data = next(iter(train_loader))
with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(check_data['image'].to(device))
scale_factor = 1 / torch.std(z)
controlnet_inferer = ControlNetDiffusionInferer(scheduler)
inferer = DiffusionInferer(scheduler)

# Path to the directory containing your NIfTI files
directory_path = '/home/marouanehajri/Downloads/bg'

# List all files in the directory
files = os.listdir(directory_path)

# Loop over each file in the directory
for file_name in files:
    # Prepare Dataset
    train_dataset, _ = setup_datasets(  args.dataset_file, 
                                    config["dataset"]['input_channels'])
        ##############################################################################
        ### Think about maybe having the feeding of the dataset in the slurm file ####
        ############################################################################## 
    train_loader = DataLoader(  train_dataset, 
                            batch_size=config["dataset"]["batch_size"], 
                            shuffle=config["dataset"]["shuffle"], 
                            num_workers=config["dataset"]["num_workers"], 
                            persistent_workers=config["dataset"]["persistent_workers"])

    original_nii = nib.load(os.path.join('/home/marouanehajri/Downloads/raw', file_name))
    original_affine = original_nii.affine
    output_file=f'predictions_340/{file_name}_dataset.hdf5'
    
    dataset = NiftiHDF5Dataset(output_file)
    data_loader = DataLoader(dataset, batch_size=4, shuffle=False, num_workers=16, persistent_workers=True)
    device = torch.device("cuda")
    
    total_volumes = len(dataset)
    #volume_shape = data_loader.dataset[0]['gt'].shape  # Assuming all volumes have the same shape
    aggregated_output = [] #np.empty((total_volumes, *volume_shape[1:]), dtype=np.float32)
    sample = torch.randn((4, 3, 80, 80)).to(device)
    # Process data through the model
    for batch_idx, val_bat in enumerate(tqdm(data_loader, desc="Processing", total=len(data_loader))):
        with torch.no_grad(), amp.autocast(enabled=True):
            z = autoencoderkl.encode_stage_2_inputs(val_bat['gt'][:, 0:1, :, :].to(device))
            scale_factor = 1 / torch.std(z)
            m = val_bat['gt'].to(device)
    
            # Assuming you have a scheduler for timesteps
            for t in scheduler.timesteps:
                down_block_res_samples, mid_block_res_sample = controlnet(
                    x=sample, timesteps=torch.Tensor([t]).to(device).long(), controlnet_cond=m
                )
                noise_pred = unet(
                    sample,
                    timesteps=torch.Tensor([t]).to(device),
                    down_block_additional_residuals=down_block_res_samples,
                    mid_block_additional_residual=mid_block_res_sample,
                )
                sample, _ = scheduler.step(model_output=noise_pred, timestep=t, sample=sample)
    
            output = autoencoderkl.decode(sample) / scale_factor
            output_numpy = output.squeeze(1).cpu().numpy()
            aggregated_output.append(output_numpy)
            
    # Concatenate all slices into a single array
    aggregated_output = np.concatenate(aggregated_output, axis=0)
    #padded_output = np.pad(aggregated_output, ((22, 22), (22, 22), (0, 0)), 'constant', constant_values=0)
    #cropped_output = aggregated_output[:, 10:-10, 10:-10]
    padded_output = aggregated_output.astype(np.float32)
    new_nifti = nib.Nifti1Image(padded_output, affine=original_affine)
    nib.save(new_nifti, f'predictions_340/{file_name}')
    del dataset

