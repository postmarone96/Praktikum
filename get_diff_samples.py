
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler
import glob

# clear CUDA
torch.cuda.empty_cache()

device = torch.device("cuda")

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
autoencoderkl = AutoencoderKL(spatial_dims=2, in_channels=3, out_channels=3, num_channels=(128, 128, 256), latent_channels=3, num_res_blocks=2, attention_levels=(False, False, False), with_encoder_nonlocal_attn=False, with_decoder_nonlocal_attn=False)
vae_path = glob.glob('vae*.pth')
vae_model = torch.load(vae_path[0])
autoencoderkl.load_state_dict(vae_model['autoencoder_state_dict'])
val_recon_losses = vae_model['val_recon_losses']
epoch_recon_losses = vae_model['epoch_recon_losses']
epoch_gen_losses = vae_model['epoch_gen_losses']
epoch_disc_losses = vae_model['epoch_disc_losses']
intermediary_images = vae_model['intermediary_images']
val_epochs = vae_model['val_epochs']

scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="linear_beta", beta_start=0.0015, beta_end=0.0195)
unet = unet.to(device)
autoencoderkl = autoencoderkl.to(device)

checkpoint_path = glob.glob('ldm*.pth')
checkpoint = torch.load(checkpoint_path[0])
scaler.load_state_dict(checkpoint['scaler_state_dict'])
unet.load_state_dict(checkpoint['unet_state_dict'])
scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
scale_factor = checkpoint['scale_factor']
scale_factor= scale_factor.to(device)
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

import os
import matplotlib.pyplot as plt
import torch
import numpy as np

# Assuming the necessary initialization for unet, scheduler, inferer, and autoencoderkl

unet.eval()
scheduler.set_timesteps(num_inference_steps=1000)
noise = torch.randn((1, 3, 64, 64))

noise_np = noise[0, 0].cpu().numpy()
# Visualize the noise using matplotlib and save it as SVG
plt.figure(figsize=(5,5))
plt.imshow(noise_np, cmap='jet')
plt.axis('off')
plt.savefig("noise_image.png", format='png', bbox_inches='tight', pad_inches=0)
plt.close()

noise = noise.to(device)

with torch.no_grad():
    image, intermediates = inferer.sample(
        input_noise=noise,
        diffusion_model=unet,
        scheduler=scheduler,
        save_intermediates=True,
        intermediate_steps=20,
        autoencoder_model=autoencoderkl,
    )

# Extract channels from all images
channels_data = [
    ([image[0, 0] for image in intermediates], 'diff_raw', 'raw_output'),
    ([image[0, 1] for image in intermediates], 'diff_bg', 'bg_output')
]

for channel_list, folder_name, file_prefix in channels_data:
    os.makedirs(folder_name, exist_ok=True)
    
    for i, im in enumerate(channel_list):
        im = im.cpu().numpy()
        im = (im - im.min()) / (im.max() - im.min())

        # Apply the jet colormap
        colored_image = plt.cm.jet(im)

        # Convert the RGBA image to RGB
        colored_image_rgb = (colored_image[:, :, :3] * 255).astype(np.uint8)
        
        # Set up a plot without axes or frame
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.axis('off')
        ax.imshow(colored_image_rgb)
        
        # Save the image in SVG format without any frame
        filename = f'{folder_name}/{file_prefix}{i}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
