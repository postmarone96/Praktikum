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
import nibabel as nib
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
cn_config = config['CN']

# set device
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
controlnet = load_model(config = cn_config['cn'],
                        model_class = ControlNet, 
                        file_prefix = 'cn', 
                        model_prefix = 'cn',
                        device = device)

# DDPM Scheduler
scheduler = DDPMScheduler(**cn_config['ddpm_scheduler'])

# Scaler
scaler = GradScaler()

# Lock UNet weights
for p in unet.parameters():
    p.requires_grad = False

# batch size
batch_size = 30

# Optimizer
optimizer = torch.optim.Adam(params=controlnet.parameters(), lr=cn_config['optimizer']['lr'])

#Learning Rate Scheduler
scheduler_lr = ReduceLROnPlateau(optimizer, **cn_config['optimizer']['scheduler'])

# Loop over each file in the directory
train_dataset, _ = setup_datasets(  args.dataset_file, 
                                    config["dataset"]['input_channels'],
                                    condition=config["dataset"]['condition'])

train_loader = DataLoader(  train_dataset, 
                            batch_size=batch_size, 
                            shuffle=False, 
                            num_workers=config["dataset"]["num_workers"], 
                            persistent_workers=config["dataset"]["persistent_workers"])

# Inferer initialization
check_data = next(iter(train_loader))
with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(check_data['image'].to(device))
scale_factor = 1 / torch.std(z)
controlnet_inferer = ControlNetDiffusionInferer(scheduler)
inferer = DiffusionInferer(scheduler)

raw_dir = os.path.join(os.getcwd(), 'raw')
nii_files = [f for f in os.listdir(raw_dir) if f.endswith('.nii.gz')]
nii_file_path = os.path.join(raw_dir, nii_files[0])
original_nii = nib.load(nii_file_path)
original_affine = original_nii.affine

reconstructed_volume = np.zeros((original_nii.shape), dtype=np.float32)
total_slices = original_nii.shape[-1]

#volume_shape = data_loader.dataset[0]['gt'].shape  # Assuming all volumes have the same shape
aggregated_output = [] #np.empty((total_volumes, *volume_shape[1:]), dtype=np.float32)
sample = torch.randn((batch_size, 3, 80, 80)).to(device)
slice_idx = 0
# Process data through the model
for batch_idx, batch in enumerate(tqdm(train_loader, desc="Processing", total=len(train_loader))):
    with torch.no_grad(), autocast(enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(batch['image'].to(device))
        scale_factor = 1 / torch.std(z)
        m = batch['cond'].to(device)

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
        output_numpy = output_numpy[:, 10:-10, 10:-10]
        # output_numpy = np.moveaxis(output_numpy, 0, -1)
        # aggregated_output.append(output_numpy)
        start_slice_idx = slice_idx
        end_slice_idx = slice_idx + batch_size
        if end_slice_idx > total_slices:
            end_slice_idx = total_slices
        reconstructed_volume[:, :, start_slice_idx:end_slice_idx] = output_numpy

        # Update the slice index for the next batch
        slice_idx += batch_size

# Concatenate all slices into a single array
reconstructed_volume = np.moveaxis(reconstructed_volume, 0, -1)
reconstructed_nii = nib.Nifti1Image(reconstructed_volume, original_nii.affine, original_nii.header)
nib.save(reconstructed_nii, f'synth_{os.path.basename(nii_file_path)}')
del train_dataset
del train_loader
del unet
del autoencoderkl
del cn
