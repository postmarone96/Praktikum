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
from generative.inferers import LatentDiffusionInferer
from generative.losses.adversarial_loss import PatchAdversarialLoss
from generative.losses.perceptual import PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

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

def save_checkpoint_ldm(epoch, unet, optimizer, scaler, scheduler, scheduler_lr, scale_factor, epoch_losses, val_losses, filename):
    checkpoint = {
        'epoch': epoch,
        'unet_state_dict': unet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict':scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scheduler_lr_state_dict': scheduler_lr.state_dict(),
        'epoch_losses': epoch_losses,
        'val_losses': val_losses,
        'scale_factor': scale_factor,
    }
    torch.save(checkpoint, filename)

print_with_timestamp("Defining NiftiDataset class")
class NiftiHDF5Dataset(Dataset):
    def __init__(self, hdf5_file):
        self.hdf5_file = hdf5_file

    def __len__(self):
        with h5py.File(self.hdf5_file, 'r') as f:
            return len(f['all_slices'])

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            image_data = f['all_slices'][idx]
            image_data = torch.tensor(image_data).unsqueeze(0)  # Adding a channel dimension
        return image_data

print_with_timestamp("Loading data")
dataset = NiftiHDF5Dataset(args.output_file)

ldm_best_val_loss = float('inf')

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
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=16, persistent_workers=True)
val_loader = DataLoader(validation_dataset, batch_size=10, shuffle=False, num_workers=16, persistent_workers=True)

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
optimizer = torch.optim.Adam(unet.parameters(), lr=10**(-float(args.lr)))
scheduler_lr = ReduceLROnPlateau(optimizer, 'min')
scaler = GradScaler()
autoencoderkl = AutoencoderKL(spatial_dims=2, in_channels=1, out_channels=1, num_channels=(128, 128, 256), latent_channels=3, num_res_blocks=2, attention_levels=(False, False, False), with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False)
vae_path = glob.glob('autoencoderkl_weights_*.pth')
vae_model = torch.load(vae_path[0])
autoencoderkl.load_state_dict(vae_model)
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)
unet = unet.to(device)
autoencoderkl = autoencoderkl.to(device).half()

start_epoch = 0
checkpoint_path = 'ldm_best_checkpoint.pth'
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch'] + 1
    scaler.load_state_dict(checkpoint['scaler_state_dict'])
    unet.load_state_dict(checkpoint['unet_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    scheduler_lr.load_state_dict(checkpoint['scheduler_lr_state_dict'])
    epoch_losses = checkpoint['epoch_losses']
    val_losses = checkpoint['val_losses']
    print_with_timestamp(f"Resuming from epoch {start_epoch}...")
else:
    epoch_losses = []
    val_losses = []

n_epochs = 200
val_interval = 2
check_data = first(train_loader)
with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(check_data.to(device))
print(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 / torch.std(z)

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

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
        val_losses.append(val_loss)
        # Update the plot after each epoch (or validation interval)
        plt.figure(figsize=(10, 5))
        plt.title("Learning Curves", fontsize=20)
        plt.plot(range(epoch + 1), epoch_losses, linewidth=2.0, label="Train")
        plt.plot(range(0, epoch + 1, val_interval)[:len(val_losses)], val_losses, linewidth=2.0, label="Validation")
        plt.yticks(fontsize=12)
        plt.xticks(fontsize=12)
        plt.xlabel("Epochs", fontsize=16)
        plt.ylabel("Loss", fontsize=16)
        plt.legend(prop={"size": 14})
        plt.savefig(f'LDM_learning_curves.png')
        plt.close()
        print(f"Epoch {epoch} val loss: {val_loss:.4f}")
    if epoch % 5 == 0 and epoch > 0:
        save_checkpoint_ldm(epoch, unet, optimizer, scaler, scheduler, scheduler_lr, scale_factor, epoch_losses, val_losses, f'ldm_checkpoint_epoch_{epoch}.pth')
        if val_loss < ldm_best_val_loss:
            ldm_best_val_loss = val_loss
            save_checkpoint_ldm(epoch, unet, optimizer, scaler, scheduler, scheduler_lr, scale_factor, epoch_losses, val_losses, 'ldm_best_checkpoint.pth')
    
progress_bar.close()

# Get current date and time
now = datetime.now()
# Format date and time
date_time = now.strftime("%Y%m%d_%H%M")
# Use date_time string in file name
torch.save(unet.state_dict(), f'unet_weights_{date_time}.pth')
torch.save(scheduler.state_dict(), f'scheduler_{date_time}.pth')
torch.save(scale_factor, f'scale_factor_{date_time}.pth')

torch.cuda.empty_cache()