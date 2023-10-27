import os
import argparse
import h5py
import glob
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import monai
from monai.utils import first
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from generative.inferers import DiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import DiffusionModelUNet, ControlNet
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

def save_checkpoint_cn(epoch, unet, optimizer, scaler, scheduler, epoch_losses, val_losses, val_epochs, filename):
    checkpoint = {
        'epoch': epoch,
        'unet_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict':scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch_losses': epoch_losses,
        'val_losses': val_losses,
        'val_epochs': val_epochs,
    }
    torch.save(checkpoint, filename)

print_with_timestamp("Defining NiftiDataset class")
class NiftiHDF5Dataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def __len__(self):
        with h5py.File(self.hdf5_file, 'r') as f:
            # Assuming image_slices and annotation_slices have the same length
            return len(f['bg'])

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            bg = f['bg'][idx]
            raw = f['raw'][idx]
            gt = f['gt'][idx]

        # Convert to PyTorch tensors
        chann_1 = torch.tensor(bg)
        chann_2 = torch.tensor(raw)
        chann_3 = chann_1
        # Stack the image and annotation along the channel dimension
        combined = {}
        combined['image'] = torch.stack([chann_1, chann_2, chann_3], dim=0)
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
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=16, persistent_workers=True)
val_loader = DataLoader(validation_dataset, batch_size=5, shuffle=False, num_workers=16, persistent_workers=True)

print_with_timestamp("Setting up device and models")
device = torch.device("cuda")

print_with_timestamp("Start setting")
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    num_res_blocks=2,
    num_channels=(128, 256, 512),
    attention_levels=(False, True, True),
    num_head_channels=(0, 256, 512),
)

scaler = GradScaler()

ldm_path = glob.glob('ldm_model_*.pth')
ldm_model = torch.load(ldm_path[0])
unet.load_state_dict(ldm_model['unet_state_dict'])

scheduler = DDPMScheduler(num_train_timesteps=1000)
unet = unet.to(device)
# Create control net
controlnet = ControlNet(
    spatial_dims=2,
    in_channels=3,
    num_channels=(128, 256, 512),
    attention_levels=(False, True, True),
    num_res_blocks=1,
    num_head_channels=128,
    conditioning_embedding_num_channels=(16,),
)
# Copy weights from the DM to the controlnet
controlnet.load_state_dict(unet.state_dict(), strict=False)
controlnet = controlnet.to(device)

# Now, we freeze the parameters of the diffusion model.
for p in unet.parameters():
    p.requires_grad = False
optimizer = torch.optim.Adam(params=controlnet.parameters(), lr=2.5*10**(-float(args.lr)))

start_epoch = 0
checkpoint_path = glob.glob('cn_checkpoint_epoch_*.pth')
if checkpoint_path:
    checkpoint = torch.load(checkpoint_path[0])
    start_epoch = checkpoint['epoch'] + 1
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    unet.load_state_dict(checkpoint['unet_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch_losses = checkpoint['epoch_losses']
    val_losses = checkpoint['val_losses']
    val_epochs = checkpoint['val_epochs']
    print_with_timestamp(f"Resuming from epoch {start_epoch}...")
else:
    epoch_losses = []
    val_losses = []
    val_epochs = []

inferer = DiffusionInferer(scheduler)

device = torch.device("cuda")

n_epochs = 150
val_interval = 5
epoch_losses = []
val_epoch_losses = []

for epoch in range(start_epoch, n_epochs):
    unet.train()
    epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=70)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch["image"].to(device)
        masks = batch["gt"].to(device)

        optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):

            # Generate random noise
            noise = torch.randn_like(images).to(device)

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            images_noised = scheduler.add_noise(images, noise=noise, timesteps=timesteps)

            # Get controlnet output
            down_block_res_samples, mid_block_res_sample = controlnet(
                x=images_noised, timesteps=timesteps, controlnet_cond=masks

            )
            # Get model prediction
            noise_pred = unet(
                x=images_noised,
                timesteps=timesteps,
                down_block_additional_residuals=down_block_res_samples,
                mid_block_additional_residual=mid_block_res_sample,
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        epoch_loss += loss.item()

        progress_bar.set_postfix({"loss": epoch_loss / (step + 1)})
    epoch_losses.append(epoch_loss / (step + 1))

    if epoch % 5 == 0 and epoch > 0:
        val_epochs.append(epoch)
        unet.eval()
        val_epoch_loss = 0
        for step, batch in enumerate(val_loader):
            images = batch["image"].to(device)

            with torch.no_grad():
                with autocast(enabled=True):
                    noise = torch.randn_like(images).to(device)
                    timesteps = torch.randint(
                        0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                    ).long()
                    noise_pred = inferer(inputs=images, diffusion_model=unet, noise=noise, timesteps=timesteps)
                    val_loss = F.mse_loss(noise_pred.float(), noise.float())

            val_epoch_loss += val_loss.item()
            progress_bar.set_postfix({"val_loss": val_epoch_loss / (step + 1)})
            break
        val_losses.append(val_epoch_loss / (step + 1))

        save_checkpoint_cn(epoch, unet, optimizer, scaler, scheduler, epoch_losses, val_losses, val_epochs, f'cn_checkpoint_epoch_{epoch}.pth')
        if val_loss < cn_best_val_loss:
            cn_best_val_loss = val_loss
            save_checkpoint_cn(epoch, unet, optimizer, scaler, scheduler, epoch_losses, val_losses, val_epochs, 'cn_best_checkpoint.pth')


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
save_checkpoint_cn(epoch, unet, optimizer, scaler, scheduler, epoch_losses, val_losses, val_epochs, f'cn_model_{date_time}.pth')

torch.cuda.empty_cache()
