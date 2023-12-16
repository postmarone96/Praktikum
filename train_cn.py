import os
import argparse
import h5py
import glob
from datetime import datetime
import matplotlib.pyplot
import matplotlib.pyplot as plt
import numpy as np
import torch
import monai
from monai.utils import first
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torchvision import transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from generative.inferers import DiffusionInferer, ControlNetDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import DiffusionModelUNet, ControlNet, AutoencoderKL
from generative.networks.schedulers import DDPMScheduler


import tempfile
import time


# clear CUDA
torch.cuda.empty_cache()

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--lr", type=str, default=1e-4)
args = parser.parse_args()

def print_with_timestamp(message):
    current_time = datetime.now()
    print(f"{current_time} - {message}")
print_with_timestamp("Starting the script")

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

print_with_timestamp("Defining NiftiDataset class")
class NiftiHDF5Dataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def __len__(self):
        with h5py.File(self.hdf5_file, 'r') as f:
            return len(f['bg'])

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            bg = f['bg'][idx]
            raw = f['raw'][idx]
            gt = f['gt'][idx]
            
        chann_1 = torch.tensor(bg)
        chann_2 = torch.tensor(raw)
        combined = {}
        combined['image'] = torch.stack([chann_1, chann_2], dim=0)
        combined['gt'] = torch.tensor(gt).unsqueeze(0)

        return combined

print_with_timestamp("Loading data")
dataset = NiftiHDF5Dataset(args.output_file)

cn_best_val_loss = float('inf')

validation_split = 0.2
dataset_size = len(dataset)
validation_size = int(validation_split * dataset_size)
training_size = dataset_size - validation_size
indices = torch.randperm(len(dataset))
train_indices = indices[:training_size]
val_indices = indices[training_size:]

train_dataset = Subset(dataset, train_indices)
validation_dataset = Subset(dataset, val_indices)

print_with_timestamp("Splitting data for training and validation")
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=16, persistent_workers=True)
val_loader = DataLoader(validation_dataset, batch_size=16, shuffle=False, num_workers=16, persistent_workers=True)

print_with_timestamp("Setting up device and models")
device = torch.device("cuda")

# AutoencoderKL
autoencoderkl = AutoencoderKL(spatial_dims=2, in_channels=2, out_channels=2, num_channels=(128, 128, 256), latent_channels=3, num_res_blocks=2, attention_levels=(False, False, False), with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False)
vae_path = glob.glob('vae_model_*.pth')
vae_model = torch.load(vae_path[0])
if list(vae_model['autoencoder_state_dict'].keys())[0].startswith('module.'):
    new_state_dict = {k[len("module."):]: v for k, v in vae_model['autoencoder_state_dict'].items()}
    autoencoderkl.load_state_dict(new_state_dict)
else:
    new_state_dict = vae_model['autoencoder_state_dict']
    autoencoderkl.load_state_dict(new_state_dict)
autoencoderkl = autoencoderkl.to(device)

# Mask Autoencoder
mask_autoencoderkl = AutoencoderKL(spatial_dims=2, in_channels=1, out_channels=1, num_channels=(128, 128, 256), latent_channels=3, num_res_blocks=2, attention_levels=(False, False, False), with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False)
mask_path = glob.glob('mask_model_*.pth')
mask_model = torch.load(mask_path[0])
if list(mask_model['autoencoder_state_dict'].keys())[0].startswith('module.'):
    new_state_dict = {k[len("module."):]: v for k, v in mask_model['autoencoder_state_dict'].items()}
    mask_autoencoderkl.load_state_dict(new_state_dict)
else:
    new_state_dict = mask_model['autoencoder_state_dict']
    mask_autoencoderkl.load_state_dict(new_state_dict)
mask_autoencoderkl = mask_autoencoderkl.to(device)

# UNET
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    num_res_blocks=2,
    num_channels=(128, 256, 512),
    attention_levels=(False, True, True),
    num_head_channels=(0, 256, 512),
)
ldm_path = glob.glob('ldm_model_*.pth')
ldm_model = torch.load(ldm_path[0])
if list(ldm_model['unet_state_dict'].keys())[0].startswith('module.'):
    new_state_dict = {k[len("module."):]: v for k, v in ldm_model['unet_state_dict'].items()}
    unet.load_state_dict(new_state_dict)
else:
    unet.load_state_dict(ldm_model['unet_state_dict'])
unet = torch.nn.DataParallel(unet)
unet = unet.to(device)

# ControlNet
controlnet = ControlNet(
    spatial_dims=2,
    in_channels=3,
    num_channels=(128, 256, 512),
    attention_levels=(False, True, True),
    num_res_blocks=2,
    num_head_channels=(0, 256, 512),
    conditioning_embedding_num_channels=(16,),
    conditioning_embedding_in_channels = 3,
)
# Copy weights from the DM to the controlnet
controlnet.load_state_dict(unet.module.state_dict(), strict=False)
controlnet = torch.nn.DataParallel(controlnet)
controlnet = controlnet.to(device)

# Other modules 
scheduler = DDPMScheduler(num_train_timesteps=1000)
scaler = GradScaler()
for p in unet.parameters():
    p.requires_grad = False
optimizer = torch.optim.Adam(params=controlnet.parameters(), lr=10**(-float(args.lr)))
scheduler_lr = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=3)

# Initialize from checkpoint
start_epoch = 0
checkpoint_path = glob.glob('cn_checkpoint_epoch_*.pth')
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path[0])
    start_epoch = checkpoint['epoch'] + 1
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    controlnet.module.load_state_dict(checkpoint['cn_state_dict'])
    unet.module.load_state_dict(checkpoint['unet_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler_lr = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=40)
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch_losses = checkpoint['epoch_losses']
    val_losses = checkpoint['val_losses']
    val_epochs = checkpoint['val_epochs']
    lr_rates = checkpoint['lr_rates']
    print_with_timestamp(f"Resuming from epoch {start_epoch}...")
else:
    epoch_losses = []
    val_losses = []
    val_epochs = []
    lr_rates = []

# Inferer
check_data = next(iter(train_loader))
with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(check_data['image'].to(device))
#print(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 / torch.std(z)
controlnet_inferer = ControlNetDiffusionInferer(scheduler)
inferer = DiffusionInferer(scheduler)

# Training loop
n_epochs = 150
val_interval = 5
for epoch in range(start_epoch, n_epochs):
    controlnet.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        masks = batch["gt"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=True):
            with torch.no_grad():
                e = autoencoderkl.encoder(images)
                m = mask_autoencoderkl.encoder(masks)
            # Generate random noise
            noise = torch.randn_like(m).to(device)
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (m.shape[0],), device=m.device
            ).long()
            noise_pred = controlnet_inferer(inputs=e,
                                    diffusion_model=unet,
                                    controlnet=controlnet,
                                    noise=noise,
                                    timesteps=timesteps,
                                    cn_cond=m,
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
        controlnet.eval()
        val_epoch_loss = 0
        for step, batch in enumerate(val_loader):
            images = batch["image"].to(device)
            masks = batch["gt"].to(device)
            with torch.no_grad():
                with autocast(enabled=True):
                    e = autoencoderkl.encoder(images)
                    m = mask_autoencoderkl.encoder(masks)
                    # noise generation
                    noise = torch.randn_like(m).to(device)
                    timesteps = torch.randint(
                        0, controlnet_inferer.scheduler.num_train_timesteps, (m.shape[0],), device=m.device
                    ).long()
            
                    noise_pred = controlnet_inferer(inputs=e,
                                    diffusion_model=unet,
                                    controlnet=controlnet,
                                    noise=noise,
                                    timesteps=timesteps,
                                    cn_cond=m,
                    )
                    val_loss = F.mse_loss(noise_pred.float(), noise.float())

            val_epoch_loss += val_loss.item()
            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
            print(val_loss)
            break
        scheduler_lr.step(val_epoch_loss / (step + 1)) 
        lr_unet = optimizer.param_groups[0]['lr']
        lr_rates.append(lr_unet)
        val_losses.append(val_epoch_loss / (step + 1))

    if epoch % 5 == 0 and epoch > 0:
        save_checkpoint_cn(epoch, controlnet, unet, optimizer, scaler, scheduler, scheduler_lr, epoch_losses, val_losses, val_epochs, lr_rates, f'cn_checkpoint_epoch_{epoch}.pth')


    if epoch > val_interval:
        fig, ax1 = plt.subplots(figsize=(10, 5))
        # Plot Losses
        color = 'tab:blue'
        ax1.set_title('Learning Curves and Learning Rate', fontsize=20)
        ax1.set_xlabel('Epochs', fontsize=16)
        ax1.set_xticks(range(0, epoch + 1, 10))
        ax1.set_ylabel('Loss', fontsize=16, color=color)
        ax1.plot(range(epoch + 1), epoch_losses, color=color, label="Train")
        ax1.plot(val_epochs, val_losses, 'b--', label="Validation")
        ax1.tick_params(axis='y', labelcolor=color)

        # Twin axis for learning rates
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Learning Rate', fontsize=16, color=color)
        ax2.plot(val_epochs, lr_rates, color=color, label='Learning Rate')
        ax2.tick_params(axis='y', labelcolor=color)

        fig.tight_layout()
        fig.legend(loc="upper right", bbox_to_anchor=(0.8,0.9))

        plt.savefig('CN_learning_curves.png')
        plt.close()
progress_bar.close()

# Get current date and time
now = datetime.now()
# Format date and time
date_time = now.strftime("%Y%m%d_%H%M")
# Use date_time string in file name
save_checkpoint_cn(epoch, controlnet, unet, optimizer, scaler, scheduler, scheduler_lr, epoch_losses, val_losses, val_epochs, lr_rates, f'cn_model_{date_time}.pth')

torch.cuda.empty_cache()
