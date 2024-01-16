import os
import argparse
import h5py
import glob
import pickle
import zipfile
import shutil
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
parser.add_argument("--data_size", type=str, required=True)
parser.add_argument("--job", type=str, required=True)
parser.add_argument("--lr", type=str, default=1e-4)
args = parser.parse_args()

def print_with_timestamp(message):
    current_time = datetime.now()
    print(f"{current_time} - {message}")
print_with_timestamp("Starting the script")

def save_checkpoint_ldm(epoch, unet, optimizer, scaler, scheduler, scheduler_lr, scale_factor, epoch_losses, val_losses, val_epochs, lr_rates, filename):
    checkpoint = {
        'epoch': epoch,
        'unet_state_dict': unet.module.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict':scaler.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scheduler_lr_state_dict': scheduler_lr.state_dict(),
        'epoch_losses': epoch_losses,
        'val_losses': val_losses,
        'scale_factor': scale_factor,
        'val_epochs': val_epochs,
        'lr_rates': lr_rates,
    }
    torch.save(checkpoint, filename)

number_of_channels = 2

print_with_timestamp("Defining NiftiDataset class")
class NiftiHDF5Dataset(Dataset):
    def __init__(self, hdf5_file, number_of_channels):
        self.hdf5_file = hdf5_file
        self.number_of_channels = number_of_channels
    def __len__(self):
        with h5py.File(self.hdf5_file, 'r') as f:
            # Assuming image_slices and annotation_slices have the same length
            return len(f['bg'])

    def __getitem__(self, idx):
        with h5py.File(self.hdf5_file, 'r') as f:
            bg_data = f['bg'][idx]
            raw_data = f['raw'][idx]
            if self.number_of_channels == 3:
                gt_data = f['gt'][idx]

        # Convert to PyTorch tensors
        chann_1 = torch.tensor(raw_data)
        chann_2 = torch.tensor(bg_data)
        if self.number_of_channels == 3:
            chann_3 = torch.tensor(gt_data)

        # Stack the image and annotation along the channel dimension
        if self.number_of_channels == 3:
            combined = torch.stack([chann_1, chann_2, chann_3], dim=0)
        else: 
            combined = chann_1.unsqueeze(0) #torch.stack([chann_1, chann_2], dim=0)

        return combined

vae_best_val_loss = float('inf')
ldm_best_val_loss = float('inf')

print_with_timestamp("Loading data")
dataset = NiftiHDF5Dataset(args.output_file, number_of_channels)

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
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=16, persistent_workers=True)
val_loader = DataLoader(validation_dataset, batch_size=8, shuffle=False, num_workers=16, persistent_workers=True)

print_with_timestamp("Setting up device and models")
device = torch.device("cuda")

print_with_timestamp("Start setting")
unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    num_res_blocks=2,
    num_channels=(128, 256, 512),
    attention_levels=(True, True, True),
    num_head_channels=(128, 256, 512),
)
optimizer = torch.optim.Adam(unet.parameters(), lr=10**(-float(args.lr)))
scheduler_lr = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=20)
scaler = GradScaler()

autoencoderkl = AutoencoderKL(spatial_dims=2, in_channels=1, out_channels=1, num_channels=(128, 128, 256), latent_channels=3, num_res_blocks=2, attention_levels=(False, False, False), with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False)
vae_path = glob.glob('vae_model_*.pth')
vae_model = torch.load(vae_path[0])
if list(vae_model['autoencoder_state_dict'].keys())[0].startswith('module.'):
    new_state_dict = {k[len("module."):]: v for k, v in vae_model['autoencoder_state_dict'].items()}
    autoencoderkl.load_state_dict(new_state_dict)
else:
    new_state_dict = vae_model['autoencoder_state_dict']
    autoencoderkl.load_state_dict(new_state_dict)

autoencoderkl = autoencoderkl.to(device).half()

scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)
unet = torch.nn.DataParallel(unet)
unet = unet.to(device)


start_epoch = 0
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
    print_with_timestamp(f"Resuming from epoch {start_epoch}...")
else:
    epoch_losses = []
    val_losses = []
    val_epochs = []
    lr_rates = []


n_epochs = 200
val_interval = 2
check_data = next(iter(train_loader))
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
    if epoch % 5 == 0 and epoch > 0:
        save_checkpoint_ldm(epoch, unet, optimizer, scaler, scheduler, scheduler_lr, scale_factor, epoch_losses, val_losses, val_epochs, lr_rates, f'ldm_checkpoint_epoch_{epoch}.pth')
        if val_loss < ldm_best_val_loss:
            ldm_best_val_loss = val_loss
            save_checkpoint_ldm(epoch, unet, optimizer, scaler, scheduler, scheduler_lr, scale_factor, epoch_losses, val_losses, val_epochs, lr_rates, 'ldm_best_checkpoint.pth')
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

        plt.savefig('LDM_learning_curves.png')
        plt.close()
progress_bar.close()

# Get current date and time
now = datetime.now()
# Format date and time
date_time = now.strftime("%Y%m%d_%H%M")
# Use date_time string in file name
save_checkpoint_ldm(epoch, unet, optimizer, scaler, scheduler, scheduler_lr, scale_factor, epoch_losses, val_losses, val_epochs, lr_rates, f'ldm_model_{date_time}.pth')

autoencoderkl = autoencoderkl.to(device).float()

current_working_directory = os.getcwd()  # Gets the directory where the script is being executed
pkl_dir = os.path.join(current_working_directory, 'pkl_dir')
os.makedirs(pkl_dir, exist_ok=True)

number_of_samples = 50
data_dict = {}
for i in range(number_of_samples):
    unet.eval()
    scheduler.set_timesteps(num_inference_steps=1000)
    noise = torch.randn((1, 3, 64, 64))
    noise = noise.to(device)
    
    with torch.no_grad():
        image, intermediates = inferer.sample(
            input_noise=noise,
            diffusion_model=unet,
            scheduler=scheduler,
            save_intermediates=True,
            intermediate_steps=100,
            autoencoder_model=autoencoderkl,
        )
    # Store intermediates and noise in the dictionary
    data_dict[i] = {'intermediates': intermediates, 'noise': noise}
    # Decode latent representation of the intermediary images
    with torch.no_grad():
        # Concatenating the images for each channel (first two channels in this case) and then all channels together
        concat_channels = [torch.cat([img[:, j, :, :].unsqueeze(1) for img in intermediates], dim=-1) for j in range(number_of_channels)]
        concat_all_channels = torch.cat(concat_channels, dim=-2)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, number_of_channels))
    im = ax.imshow(concat_all_channels[0, 0].cpu(), cmap="jet", vmin=0, vmax=1)
    # Remove the axis
    ax.axis('off')
    
    # Add the colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.015, pad=0.04)
    cbar.set_label('Intensity', rotation=270, labelpad=15,  verticalalignment='center')
    channel_height = concat_all_channels.size(2) // number_of_channels
    channel_labels = ['Bg', 'Raw', 'Gt']
    for j in range(number_of_channels):
        ax.text(-150, channel_height * (0.5 + j), channel_labels[j], fontsize=12, va='center', ha='center')

    plt.savefig(os.path.join(pkl_dir, f'sample_{i}.png'), dpi=300)
    plt.close()

# Save the data dictionary as a pickle file
with open(os.path.join(pkl_dir, 'data_dict.pkl'), 'wb') as f:
    pickle.dump(data_dict, f)


output_filename = os.path.join(current_working_directory, f'samples_{args.job}')  # Output file path

shutil.make_archive(output_filename, 'zip', pkl_dir)

torch.cuda.empty_cache()


