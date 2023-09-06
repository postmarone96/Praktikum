import os
import argparse
import h5py
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
import monai
import nibabel as nib
from torch.utils.data import Dataset, DataLoader, Subset
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
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
parser.add_argument("--batch_size", type=int, default=20)
parser.add_argument("--lr_step", type=float, default=10)
parser.add_argument("--lr_gamma", type=float, default=0.5)
args = parser.parse_args()

def print_with_timestamp(message):
    current_time = datetime.now()
    print(f"{current_time} - {message}")
print_with_timestamp("Starting the script")

def save_checkpoint_vae(epoch, autoencoder_model, discriminator_model, optimizer_g, optimizer_d, scheduler_d, scheduler_g, val_recon_losses, epoch_recon_losses, epoch_gen_losses, epoch_disc_losses, intermediary_images, filename):
    checkpoint = {
        'epoch': epoch,
        'autoencoder_state_dict': autoencoder_model.state_dict(),
        'discriminator_state_dict': discriminator_model.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
        'scheduler_d_state_dict': scheduler_d.state_dict(),
        'scheduler_g_state_dict': scheduler_g.state_dict(),
        'val_recon_losses': val_recon_losses,
        'epoch_recon_losses': epoch_recon_losses,
        'epoch_gen_losses': epoch_gen_losses,
        'epoch_disc_losses': epoch_disc_losses,
        'intermediary_images': intermediary_images
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

vae_best_val_loss = float('inf')
ldm_best_val_loss = float('inf')

print_with_timestamp("Loading data")
dataset = NiftiHDF5Dataset(hdf5_file=args.output_file)

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
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, persistent_workers=True)
val_loader = DataLoader(validation_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, persistent_workers=True)

print_with_timestamp("Setting up device and models")
device = torch.device("cuda")

print_with_timestamp("AutoEncoder setup")

# Before the training loop:
start_epoch = 0
checkpoint_path = 'vae_best_checkpoint.pth'
autoencoderkl = AutoencoderKL(spatial_dims=2, in_channels=1, out_channels=1, num_channels=(128, 128, 256), latent_channels=3, num_res_blocks=2, attention_levels=(False, False, False), with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False)
discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)
optimizer_g = torch.optim.Adam(autoencoderkl.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)
scheduler_g = torch.optim.lr_scheduler.StepLR(optimizer_g, step_size=args.lr_step, gamma=args.lr_gamma)
scheduler_d = torch.optim.lr_scheduler.StepLR(optimizer_d, step_size=args.lr_step, gamma=args.lr_gamma)
adv_loss = PatchAdversarialLoss(criterion="least_squares")
adv_weight = 0.01
perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
autoencoderkl = autoencoderkl.to(device)
discriminator = discriminator.to(device)
perceptual_loss= perceptual_loss.to(device)

if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    start_epoch = checkpoint['epoch'] + 1  # because we start the next epoch
    autoencoderkl.load_state_dict(checkpoint['autoencoder_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    scheduler_d.load_state_dict(checkpoint['optimizer_d_state_dict'])
    scheduler_g.load_state_dict(checkpoint['optimizer_g_state_dict'])
    val_recon_losses = checkpoint['val_recon_losses']
    epoch_recon_losses = checkpoint['epoch_recon_losses']
    epoch_gen_losses = checkpoint['epoch_gen_losses']
    epoch_disc_losses = checkpoint['epoch_disc_losses']
    intermediary_images = checkpoint['intermediary_images']
    print_with_timestamp(f"Resuming from epoch {start_epoch}...")
else:
    val_recon_losses = []
    epoch_recon_losses = []
    epoch_gen_losses = []
    epoch_disc_losses = []
    intermediary_images = []

perceptual_weight = 0.001
scaler_g = torch.cuda.amp.GradScaler()
scaler_d = torch.cuda.amp.GradScaler()
kl_weight = 1e-6
n_epochs = 100
val_interval = 5
autoencoder_warm_up_n_epochs = 10
num_example_images = 4

print_with_timestamp("Start setting")
for epoch in range(start_epoch, n_epochs):
    autoencoderkl.train()
    discriminator.train()
    epoch_loss = 0
    gen_epoch_loss = 0
    disc_epoch_loss = 0
    progress_bar = tqdm(enumerate(train_loader), total=len(train_loader), ncols=110)
    progress_bar.set_description(f"Epoch {epoch}")
    for step, batch in progress_bar:
        images = batch.to(device)
        optimizer_g.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            reconstruction, z_mu, z_sigma = autoencoderkl(images)

            recons_loss = F.l1_loss(reconstruction.float(), images.float())
            p_loss = perceptual_loss(reconstruction.float(), images.float())
            kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3])
            kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
            loss_g = recons_loss + (kl_weight * kl_loss) + (perceptual_weight * p_loss)

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

        scaler_g.scale(loss_g).backward()
        scaler_g.step(optimizer_g)
        scaler_g.update()
        scheduler_g.step()
        
        if epoch > autoencoder_warm_up_n_epochs:
            with autocast(enabled=True):
                optimizer_d.zero_grad(set_to_none=True)

                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

            scaler_d.scale(loss_d).backward()
            scaler_d.step(optimizer_d)
            scaler_d.update()
            scheduler_d.step()
            
        epoch_loss += recons_loss.item()
        if epoch > autoencoder_warm_up_n_epochs:
            gen_epoch_loss += generator_loss.item()
            disc_epoch_loss += discriminator_loss.item()

        progress_bar.set_postfix(
            {
                "recons_loss": epoch_loss / (step + 1),
                "gen_loss": gen_epoch_loss / (step + 1),
                "disc_loss": disc_epoch_loss / (step + 1),
            }
        )
    epoch_recon_losses.append(epoch_loss / (step + 1))
    epoch_gen_losses.append(gen_epoch_loss / (step + 1))
    epoch_disc_losses.append(disc_epoch_loss / (step + 1))

    if (epoch + 1) % val_interval == 0:
        autoencoderkl.eval()
        val_loss = 0
        with torch.no_grad():
            for val_step, batch in enumerate(val_loader, start=1):
                images = batch.to(device)

                with autocast(enabled=True):
                    reconstruction, z_mu, z_sigma = autoencoderkl(images)
                    # Get the first reconstruction from the first validation batch for visualisation purposes
                    if val_step == 1:
                        intermediary_images.append(reconstruction[:num_example_images, 0])

                    recons_loss = F.l1_loss(images.float(), reconstruction.float())

                val_loss += recons_loss.item()

        val_loss /= val_step
        val_recon_losses.append(val_loss)
        print(f"epoch {epoch + 1} val loss: {val_loss:.4f}")
        save_checkpoint_vae(epoch, autoencoderkl, discriminator, optimizer_g, optimizer_d, scheduler_d, scheduler_g, val_recon_losses, epoch_recon_losses, epoch_gen_losses, epoch_disc_losses, intermediary_images, f'vae_checkpoint_epoch_{epoch}.pth')
        if val_loss < vae_best_val_loss:
            vae_best_val_loss = val_loss
            save_checkpoint_vae(epoch, autoencoderkl, discriminator, optimizer_g, optimizer_d, scheduler_d, scheduler_g, val_recon_losses, epoch_recon_losses, epoch_gen_losses, epoch_disc_losses, intermediary_images, 'vae_best_checkpoint.pth')

progress_bar.close()

# Get current date and time
now = datetime.now()
# Format date and time
date_time = now.strftime("%Y%m%d_%H%M")  # I replaced ":" with "M" to avoid file naming issues
# Use date_time string in file name
torch.save(autoencoderkl.state_dict(), f'autoencoderkl_weights_{date_time}.pth')
torch.save(discriminator.state_dict(), f'discriminator_weights_{date_time}.pth')

plt.figure(figsize=(10, 5))
plt.plot(epoch_recon_losses, label='Training Reconstruction Loss')
plt.plot(range(0, n_epochs, val_interval), val_recon_losses, label='Validation Reconstruction Loss', linestyle='dashed')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Learning Curves')
plt.savefig('VAE_learning_curves.png')

del discriminator
del perceptual_loss
torch.cuda.empty_cache()
