import os
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
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

def print_with_timestamp(message):
    current_time = datetime.now()
    print(f"{current_time} - {message}")
print_with_timestamp("Starting the script")

def save_checkpoint_ldm(epoch, model, optimizer, filename):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)

def save_checkpoint_vae(epoch, autoencoder_model, discriminator_model, optimizer_g, optimizer_d, filename):
    checkpoint = {
        'epoch': epoch,
        'autoencoder_state_dict': autoencoder_model.state_dict(),
        'discriminator_state_dict': discriminator_model.state_dict(),
        'optimizer_g_state_dict': optimizer_g.state_dict(),
        'optimizer_d_state_dict': optimizer_d.state_dict(),
    }
    torch.save(checkpoint, filename)

print_with_timestamp("Defining NiftiDataset class")
class NiftiDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.nii_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.nii.gz')]
        self.total_slices = 0
        self.slice_indices = []
        
        for nii_path in self.nii_files:
            img = nib.load(nii_path)
            image_data = img.get_fdata()
            image_data = np.moveaxis(image_data, -1, 0)
            self.total_slices += image_data.shape[0]
            self.slice_indices.extend([(nii_path, i) for i in range(image_data.shape[0])])

    def __len__(self):
        return self.total_slices

    def __getitem__(self, idx):
        nii_path, slice_idx = self.slice_indices[idx]
        img = nib.load(nii_path)
        image_data = img.get_fdata()
        image_data = np.moveaxis(image_data, -1, 0)
        image_data = image_data.astype(np.float32)
        max_value = np.max(image_data)
        image_data /= max_value
        img_slice = image_data[slice_idx]
        start_x = (img_slice.shape[0] - 256) // 2
        start_y = (img_slice.shape[1] - 256) // 2
        img_cropped = img_slice[start_x:start_x + 256, start_y:start_y + 256]
        img_tensor = torch.from_numpy(img_cropped).unsqueeze(0)
        return img_tensor

vae_best_val_loss = float('inf')
ldm_best_val_loss = float('inf')

print_with_timestamp("Loading data")
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

data_path = args.data_path
dataset = NiftiDataset(root_dir=data_path)

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
train_loader = DataLoader(train_dataset, batch_size=10, shuffle=True, num_workers=8, persistent_workers=True)
val_loader = DataLoader(validation_dataset, batch_size=10, shuffle=False, num_workers=8, persistent_workers=True)

print_with_timestamp("Setting up device and models")
device = torch.device("cuda")

print_with_timestamp("AutoEncoder setup")
autoencoderkl = AutoencoderKL(
    spatial_dims=2,
    in_channels=1,
    out_channels=1,
    num_channels=(128, 128, 256),
    latent_channels=3,
    num_res_blocks=2,
    attention_levels=(False, False, False),
    with_encoder_nonlocal_attn=False,
    with_decoder_nonlocal_attn=False,
)
autoencoderkl = autoencoderkl.to(device)

perceptual_loss = PerceptualLoss(spatial_dims=2, network_type="alex")
perceptual_loss.to(device)
perceptual_weight = 0.001

discriminator = PatchDiscriminator(spatial_dims=2, num_layers_d=3, num_channels=64, in_channels=1, out_channels=1)
discriminator = discriminator.to(device)

adv_loss = PatchAdversarialLoss(criterion="least_squares")
adv_weight = 0.01

optimizer_g = torch.optim.Adam(autoencoderkl.parameters(), lr=1e-4)
optimizer_d = torch.optim.Adam(discriminator.parameters(), lr=5e-4)

# For mixed precision training
scaler_g = torch.cuda.amp.GradScaler()
scaler_d = torch.cuda.amp.GradScaler()

kl_weight = 1e-6
n_epochs = 100
val_interval = 10
autoencoder_warm_up_n_epochs = 10

epoch_recon_losses = []
epoch_gen_losses = []
epoch_disc_losses = []
val_recon_losses = []
intermediary_images = []
num_example_images = 4

print_with_timestamp("Start setting")
for epoch in range(n_epochs):
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

        # Save checkpoint every 10 epochs (or choose another interval)
        if (epoch + 1) % 10 == 0:
            save_checkpoint_vae(epoch, autoencoderkl, discriminator, optimizer_g, optimizer_d, f'vae_checkpoint_epoch_{epoch}.pth')

        # Save checkpoint if validation loss improves
        if (epoch + 1) % val_interval == 0:
            if val_loss < vae_best_val_loss:
                vae_best_val_loss = val_loss
                save_checkpoint_vae(epoch, autoencoderkl, discriminator, optimizer_g, optimizer_d, 'vae_best_checkpoint.pth')

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

unet = DiffusionModelUNet(
    spatial_dims=2,
    in_channels=3,
    out_channels=3,
    num_res_blocks=2,
    num_channels=(128, 256, 512),
    attention_levels=(False, True, True),
    num_head_channels=(0, 256, 512),
)

scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)

with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoderkl.encode_stage_2_inputs(check_data.to(device))

print(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 / torch.std(z)

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

optimizer = torch.optim.Adam(unet.parameters(), lr=1e-4)

unet = unet.to(device)
n_epochs = 200
val_interval = 40
epoch_losses = []
val_losses = []
scaler = GradScaler()

for epoch in range(n_epochs):
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

    if (epoch + 1) % val_interval == 0:
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
        val_losses.append(val_loss)
        print(f"Epoch {epoch} val loss: {val_loss:.4f}")
        # Save checkpoint every 10 epochs (or choose another interval)
        if (epoch + 1) % 10 == 0:
            save_checkpoint_ldm(epoch, unet, optimizer, f'checkpoint_epoch_{epoch}.pth')

        # Save checkpoint if validation loss improves
        if (epoch + 1) % val_interval == 0:
            if val_loss < ldm_best_val_loss:
                ldm_best_val_loss = val_loss
                save_checkpoint_ldm(epoch, unet, optimizer, 'best_checkpoint.pth')

progress_bar.close()

# Plotting the learning curves
plt.figure()
plt.title("Learning Curves", fontsize=20)
plt.plot(np.linspace(1, n_epochs, n_epochs), epoch_losses, linewidth=2.0, label="Train")
plt.plot(
    np.linspace(val_interval, n_epochs, int(n_epochs / val_interval)),
    val_losses,
    linewidth=2.0,
    label="Validation"
)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.xlabel("Epochs", fontsize=16)
plt.ylabel("Loss", fontsize=16)
plt.legend(prop={"size": 14})
plt.savefig('LDM_learning_curves.png')
