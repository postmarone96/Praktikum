# **Project Configuration File Description**

This file provides a detailed explanation of the parameters in the JSON configuration file used for managing data and training processes in our machine learning pipeline.

## **Configuration Parameters**

#### **`broken_ds`**
Indicates the status of the dataset after a training session.
- **`0`**: Dataset is intact and not corrupted.
- **`1`**: Dataset is corrupted, possibly because the HDF5 file was not closed correctly if training exceeded two days.

#### **`model`**
Specifies the type of model used in the training session.
- **`vae`**: Variational Autoencoder.
- **`ldm`**: Latent Diffusion Model.
- **`cn`**: ControlNet.

#### **`project_dir`**
Directory where the project files are located.

#### **`ids_file`**
Path to the file containing voxel IDs necessary for non-annotated datasets.

## **Data Configuration**

#### **`dim`**
Dimension of the image slices used in training and evaluation.
- **`320`**: for 320x320 slices.

#### **`pad`**
Padding applied to image slices to achieve the required dimensions.
- **`10`**: necessary padding to convert 300x300 slices to 320x320.

#### **`gt_th`**
Threshold value for determining ground truth.
- **`0.5`**: threshold value.

#### **`size`**
Specifies the dataset size variant.
- **`xs`**: Small annotated dataset.
- **`xl`**: Large non-annotated dataset.

#### **`xs`**
Paths to the small annotated datasets.
- **`bg`**: Background image directory.
- **`raw`**: Raw image directory.
- **`gt`**: Ground truth image directory.

#### **`xl`**
Paths to the large non-annotated datasets.
- **`bg`**: Background image directory.
- **`raw`**: Raw image directory.
- **`number_of_patches`**: Number of patches in the xl dataset.
## **Job Configuration**

Configuration details for jobs depending on the model type:

#### **`Jobs VAE`**
- **`cp`**: Checkpoint parameter to specify the training epoch to start from or to use an existing checkpoint. 
  - **training VAE**: 
    - **`0`**: Start a fresh training.
    - **`other`**: Continue a VAE training from this Checkpoint.
  - **training LDM or CN**: 
    - **`0`**: Use the parameters of the last training epoch of the chosen VAE training as the VAE model parameters.
    - **`other`**: Use the parameter of a specific training epoch of chosen VAE training the as the VAE model parameters.
- **`id`**: 
  - **training VAE**: No effect.
  - **training LDM or CN**: SLURM job ID of the Chosen VAE training.
- **`directory`**: 
  - **`train_xs`**
  - **`train_xl/train_#`**


#### **`Jobs LDM`**
- **`cp`**: Checkpoint parameter to specify the training epoch to start from or to use an existing checkpoint. 
  - **training VAE**: No effect.
  - **training LDM**: 
    - **`0`**: Start a fresh training.
    - **`other`**: Continue a LDM training from this Checkpoint.
  - **training CN**: 
    - **`0`**: Use the parameters of the last training epoch of the chosen LDM training as the LDM model parameters.
    - **`other`**: Use the parameter of a specific training epoch of the chosen LDM training as the LDM model parameters.
- **`id`**: 
  - **training VAE or LDM**: No effect.
  - **training CN**: SLURM job ID of the Chosen LDM training.
- **`directory`**: 
  - **`train_xs`**
  - **`train_xl/train_#`**

#### **`Jobs CN`**
- **`cp`**: Checkpoint parameter to specify the training epoch to start from or to use an existing checkpoint. 
  - **training VAE or LDM**: No effect.
  - **training CN**: 
    - **`0`**: Start a fresh training.
    - **`other`**: Continue a CN training from this Checkpoint.
  - **Inference**: 
    - **`0`**: Use the parameters of the last training epoch of the chosen CN training as the CN model parameters.
    - **`other`**: Use the parameter of a specific training epoch of chosen CN training the as the CN model parameters.
- **`id`**: 
  - **training VAE, LDM or CN**: No effect.
  - **Inference**: SLURM job ID of the Chosen CN training.
- **`directory`**: 
  - **`train_xs`**
  - **`train_xl/train_#`**

## **Dataset Configuration**

- **`input_channels`**: Determines the channels of the prediction.
  - **`["raw"]`**: Single channel training.  
  - **`["raw", "bg"]`**: Combined channel training.

- **`condition`**: Determines the embedding conditions for the CN:
  - **`["bg", "gt"]`**: Single channel training.  
  - **`["gt"]`**: Combined channel training.

## **VAE**

- Configuration parameters for the Variational Autoencoder model.

#### **`autoencoder`**
- **`spatial_dims`**: Dimensionality of the data (e.g., 2D or 3D).  
- **`in_channels`**: Number of input channels for the autoencoder:
  - **`1`**: Single channel training.  
  - **`2`**: Combined channel training.
- **`out_channels`**: Number of output channels for the autoencoder:
  - **`1`**: Single channel training.  
  - **`2`**: Combined channel training.
- **`num_channels`**: List of channels used in the layers of the autoencoder.  
- **`latent_channels`**: Number of channels in the latent space.  
- **`num_res_blocks`**: Number of residual blocks in each stage.  
- **`attention_levels`**: Indicates whether attention is applied at each level. 
- **`with_encoder_nonlocal_attn`**: Enables non-local attention in the encoder.  
- **`with_decoder_nonlocal_attn`**: Enables non-local attention in the decoder.  

#### **`discriminator`**
- **`spatial_dims`**: Dimensionality of the data.  
- **`num_layers_d`**: Number of layers in the discriminator.  
- **`num_channels`**: Number of channels in the discriminator layers.  
- **`in_channels`**: Number of input channels for the discriminator:
  - **`1`**: Single channel training.  
  - **`2`**: Combined channel training.
- **`out_channels`**: Number of output channels for the discriminator:
  - **`1`**: Single channel training.  
  - **`2`**: Combined channel training.


#### **`loss`**
- **`adv_loss`**: Adversarial loss type. 
- **`spatial_dims`**: Dimensionality of the data for the loss.  
- **`adv_weight`**: Weight for the adversarial loss component.  
- **`perceptual_loss`**: Type of perceptual loss. 
- **`perceptual_weight`**: Weight for the perceptual loss component.  
- **`kl_weight`**: Weight for the KL divergence loss.  

#### **`optimizer`**
- **`lr_g`**: Learning rate for the generator (autoencoder).  
- **`lr_d`**: Learning rate for the discriminator.  
- **`scheduler`**:  
  - **`mode`**: Mode for the learning rate scheduler. 
  - **`factor`**: Factor by which the learning rate is reduced.  
  - **`patience`**: Number of epochs to wait before reducing the learning rate.  

#### **`training`**
- **`n_epochs`**: Number of training epochs.
- **`val_interval`**: Validation interval.
- **`autoencoder_warm_up_n_epochs`**: -1
  - **`-1`**: No warm up
  - **`#`**: Number of warm up epochs
- **`num_example_images`**: Number of Images stored 
- **`num_epochs_checkpoint`**: Number of epochs to store checkpoints.

### **`LDM`**
Configuration parameters for the Latent Diffusion Model.
#### **`unet`**
- **`spatial_dims`**: The dimensionality of the data.
- **`in_channels`**: Number of latent channels.
- **`out_channels`**: Number of latent channels.
- **`num_res_blocks`**: Number of residual blocks in each U-Net layer. 
- **`num_channels`**: A list specifying the number of channels at each resolution level of the U-Net.
- **`attention_levels`**: Indicates whether attention is applied at each level.
- **`num_head_channels`**: Number of channels used in the attention mechanism at each level.
  
#### **`ddpm_scheduler`**
- **`num_train_timesteps`**: The total number of timesteps for the diffusion process.
- **`schedule`**: The type of beta schedule used for the diffusion process.
- **`beta_start`**: The initial value of the beta parameter, which controls how much noise is added in early timesteps.
- **`beta_end`**: The final value of the beta parameter, which controls the noise level in later timesteps.

#### **`optimizer`**
- **`lr`**: The learning rate for the optimizer, controlling how fast the model learns.
- **`scheduler`**:
  - **`mode`**: Mode for the learning rate scheduler. 
  - **`factor`**: Factor by which the learning rate is reduced.  
  - **`patience`**: Number of epochs to wait before reducing the learning rate. 

#### **`training`**
- **`n_epochs`**: Total number of epochs to train the model.
- **`val_interval`**: Number of epochs between validation checks to monitor performance.
- **`num_epochs_checkpoint`**: Number of epochs to store checkpoints.

#### **`sampling`**
- **`number_of_samples`**: The number of samples to generate during the sampling process.
- **`num_inference_steps`**: Number of steps in the denoising process during sampling.
- **`noise_shape`**: The shape of the noise tensor used for sampling.
- **`intermediate_steps`**: Number of intermediate results to save during the sampling process.

### **`CN`**
Configuration parameters for the Control Net model.

#### **`cn`**
- **`spatial_dims`**: Dimensionality of the data.
- **`in_channels`**: Number of latent channels.
- **`num_res_blocks`**: Number of residual blocks.
- **`num_channels`**: Channels at each resolution level.
- **`attention_levels`**: Apply attention at specific levels.
- **`num_head_channels`**: Channels for attention heads.
- **`conditioning_embedding_num_channels`**: Embedding channels.
- **`conditioning_embedding_in_channels`**: Input channels for conditioning embeddings:
  - **`1`**: Combined channel training.
  - **`2`**: Single channel training.

#### **`ddpm_scheduler`**
- **`num_train_timesteps`**: The total number of timesteps for the diffusion process.

#### **`optimizer`**
- **`lr`**: Learning rate (e.g., `0.0001`).
- **`scheduler`**:
  - **`mode`**: Mode for the learning rate scheduler. 
  - **`factor`**: Factor by which the learning rate is reduced.  
  - **`patience`**: Number of epochs to wait before reducing the learning rate. 

#### **`training`**
- **`n_epochs`**: Number of training epochs.
- **`val_interval`**: Validation interval.
- **`num_epochs_checkpoint`**: Number of epochs to store checkpoints.

### **`Metrics`**
- **`model`**:
  - **`vae`**: Variational Autoencoder.
  - **`ldm`**: Latent Diffusion Model.
  - **`cn`**: ControlNet.
- **`dataset`**: Path to testing dataset.
- **`index`**: Row index with information about the desired model.

### **`Inference`**
- **`method`**:
  - **`1`**: Fixed noise for all slices.
  - **`2`**: Averaged noise with previous slice.
  - **`3`**: Middle step noise of the previous slice.

