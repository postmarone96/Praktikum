import os
import glob
import argparse
import h5py
import json
import signal
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.cuda.amp import autocast
from tqdm import tqdm
from helper_functions import *
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, ControlNet
from generative.inferers import ControlNetDiffusionInferer, DiffusionInferer, LatentDiffusionInferer
from generative.metrics import *
from torchvision import models
from monai.utils import set_determinism
from generative.networks.schedulers import DDPMScheduler
# set_determinism(5)

# Parse parameters form json file
with open('params.json') as json_file:
    config = json.load(json_file)

# VAE Config
vae_config = config['VAE']
ldm_config = config['LDM']
cn_config = config['CN']
metrics_config = config['Metrics']

# Prepare Dataset
train_dataset, _ = setup_datasets( 'test.hdf5',
                                    config["dataset"]['input_channels'], 
                                    condition=config["dataset"]['condition'])

train_loader = DataLoader(  train_dataset, 
                            batch_size=config["dataset"]["batch_size"], 
                            shuffle=config["dataset"]["shuffle"], 
                            num_workers=config["dataset"]["num_workers"], 
                            persistent_workers=config["dataset"]["persistent_workers"])

device = torch.device("cuda")

fid = FIDMetric()

radnet = torch.hub.load("Warvito/radimagenet-models", model="radimagenet_resnet50", verbose=True)
radnet.to(device)
radnet.eval()

if metrics_config['model'] == 'vae':
    with h5py.File('vae_metrics.h5', 'w') as f:
        file_path = 'vae_ldm.txt'
        vae_models = pd.read_csv(file_path, sep='\t')
        for index, row in vae_models.iterrows():
            vae = load_model(config=vae_config['autoencoder'], 
                                model_class = AutoencoderKL,
                                file_prefix = 'vae', 
                                model_prefix = 'autoencoder',
                                device = device, 
                                path = config['project_dir'] + row['vae'])
            vae.eval()
            synth_features = []
            real_features = []
            
            for step, batch in train_loader:
                images = batch["image"].to(device)
                with torch.no_grad(), autocast(enabled=True):
                    reconstruction, _, _ = vae(images)
                    
                    # Get the features for the real data
                    real_eval_feats = get_features(images, radnet)
                    real_features.append(real_eval_feats)
            
                    # Get the features for the synthetic data
                    synth_eval_feats = get_features(reconstruction, radnet)
                    synth_features.append(synth_eval_feats)
                        
            synth_features = torch.vstack(synth_features)
            real_features = torch.vstack(real_features)
            group = f.create_group('score_{index}')
            group.attr['vae'] = row['vae']
            group.attr['ldm'] = row['ldm']
            group.attr['fid_Score'] = fid(synth_features, real_features)
            del vae
    
elif metrics_config['model'] == 'ldm':
    with h5py.File('ldm_metrics.h5', 'w') as f:
        file_path = 'vae_ldm.txt'
        ldm_models = pd.read_csv(file_path, sep='\t')
        for index, row in ldm_models.iterrows():
            vae = load_model(config=vae_config['autoencoder'], 
                                            model_class = AutoencoderKL,
                                            file_prefix = 'vae', 
                                            model_prefix = 'autoencoder',
                                            device = device, 
                                            path = config['project_dir'] + row['vae'])

            ldm = load_model(config=ldm_config['unet'],
                                        model_class = DiffusionModelUNet,
                                        file_prefix = 'ldm', 
                                        model_prefix = 'unet',
                                        device = device, 
                                        path = config['project_dir'] + row['ldm'])
             
            vae = vae.half()
            scheduler = DDPMScheduler(**ldm_config['ddpm_scheduler'])
            check_data = next(iter(train_loader))

            with torch.no_grad(), autocast(enabled=True):
                z = vae.encode_stage_2_inputs(check_data.to(device))
            scale_factor = 1 / torch.std(z)

            inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

            vae.eval()
            ldm.eval()
            synth_features = []
            real_features = []
            
            for step, batch in train_loader:
                images = batch["image"].to(device)
                noise = torch.randn(ldm_config['sampling']['noise_shape'])
                noise = noise.to(device)
                with torch.no_grad():
                    predictions, _ = inferer.sample(
                                            input_noise=noise,
                                            diffusion_model=ldm,
                                            scheduler=scheduler,
                                            save_intermediates=True,
                                            intermediate_steps=ldm_config['sampling']['intermediate_steps'],
                                            autoencoder_model=vae)
                        
                    # Get the features for the real data
                    real_eval_feats = get_features(images, radnet)
                    real_features.append(real_eval_feats)
            
                    # Get the features for the synthetic data
                    synth_eval_feats = get_features(predictions, radnet)
                    synth_features.append(synth_eval_feats)
                        
            synth_features = torch.vstack(synth_features)
            real_features = torch.vstack(real_features)
            
            group = f.create_group('score_{index}')
            group.attr['vae'] = row['vae']
            group.attr['ldm'] = row['ldm']
            group.attr['fid_Score'] = fid(synth_features, real_features)
            del vae
            del ldm

elif metrics_config['model'] == 'cn':
    with h5py.File('vae_metrics.h5', 'w') as f:
        file_path = 'cn.txt'
        cn_models = pd.read_csv(file_path, sep='\t')
        for index, row in cn_models.iterrows():
            
            vae = load_model(config=vae_config['autoencoder'], 
                                            model_class = AutoencoderKL,
                                            file_prefix = 'vae', 
                                            model_prefix = 'autoencoder',
                                            device = device, 
                                            path = config['project_dir'] + row['vae'])

            ldm = load_model(config=ldm_config['unet'],
                                        model_class = DiffusionModelUNet,
                                        file_prefix = 'ldm', 
                                        model_prefix = 'unet',
                                        device = device, 
                                        path = config['project_dir'] + row['ldm'])

            cn = load_model(config=cn_config['cn'],
                                        model_class = ControlNet,
                                        file_prefix = 'cn', 
                                        model_prefix = 'cn',
                                        device = device, 
                                        path = config['project_dir'] + row['cn'])

            # Inferer initialization
            scheduler = DDPMScheduler(**ldm_config['ddpm_scheduler'])
            check_data = next(iter(train_loader))
            with torch.no_grad(), autocast(enabled=True):
                z = vae.encode_stage_2_inputs(check_data['image'].to(device))
            scale_factor = 1 / torch.std(z)

            controlnet_inferer = ControlNetDiffusionInferer(scheduler)
            inferer = DiffusionInferer(scheduler)

            synth_features = []
            real_features = []
            sample = torch.randn((config["dataset"]["batch_size"], 3, 80, 80)).to(device)

            for step, batch in enumerate(train_loader):
                images = batch["image"].to(device)
                masks = batch["cond"].to(device)
                with torch.no_grad(), autocast(enabled=True):
                    z = vae.encode_stage_2_inputs(images)
                    scale_factor = 1 / torch.std(z)
                    for t in scheduler.timesteps:
                        down_block_res_samples, mid_block_res_sample = cn(
                            x=sample, timesteps=torch.Tensor([t]).to(device).long(), controlnet_cond=masks
                        )
                        noise_pred = ldm(
                            sample,
                            timesteps=torch.Tensor([t]).to(device),
                            down_block_additional_residuals=down_block_res_samples,
                            mid_block_additional_residual=mid_block_res_sample,
                        )
                        sample, _ = scheduler.step(model_output=noise_pred, timestep=t, sample=sample)
                    output = vae.decode(sample) / scale_factor
                    
                    # Get the features for the real data
                    real_eval_feats = get_features(images)
                    real_features.append(real_eval_feats)

                    # Get the features for the synthetic data
                    synth_eval_feats = get_features(output)
                    synth_features.append(synth_eval_feats)
            synth_features = torch.vstack(synth_features)
            real_features = torch.vstack(real_features)

            group = f.create_group('score_{index}')
            group.attr['vae'] = row['vae']
            group.attr['ldm'] = row['ldm']
            group.attr['cn'] = row['cn']
            group.attr['fid_Score'] = fid(synth_features, real_features)
            del vae
            del ldm
            del cn
