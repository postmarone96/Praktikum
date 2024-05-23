# **Project Configuration File Description**

This README provides a detailed explanation of the parameters in the JSON configuration file used for managing data and training processes in our machine learning pipeline.

## **Configuration Parameters**

### **`broken_ds`**
Indicates the status of the dataset after a training session.
- **`0`**: Dataset is intact and not corrupted.
- **`1`**: Dataset is corrupted, possibly because the HDF5 file was not closed correctly if training exceeded two days.

### **`model`**
Specifies the type of model used in the training session.
- **`vae`**: Visual Autoencoder.
- **`ldm`**: Latent Diffusion Model.
- **`cn`**: Control Net.

### **`project_dir`**
Directory where the project files are located.
- Example: **`$HOME/Praktikum/`**

### **`ids_file`**
Path to the file containing voxel IDs necessary for non-annotated datasets.
- Example: **`$HOME/Praktikum/ids.txt`**

## **Data Configuration**

### **`dim`**
Dimension of the image slices used in training and evaluation.
- **`320`**: for 320x320 slices.

### **`pad`**
Padding applied to image slices to achieve the required dimensions.
- **`10`**: necessary padding to convert 300x300 slices to 320x320.

### **`gt_th`**
Threshold value for determining ground truth.
- **`0.5`**: threshold value.

### **`size`**
Specifies the dataset size variant.
- **`xs`**: Small annotated dataset.
- **`xl`**: Large non-annotated dataset.

### **`xs`**
Paths to the small annotated datasets.
- **`bg`**: Background image directory.
- **`raw`**: Raw image directory.
- **`gt`**: Ground truth image directory.

### **`xl`**
Paths to the large non-annotated datasets.
- **`bg`**: Background image directory.
- **`raw`**: Raw image directory.
- **`number_of_patches`**: Number of patches in the xl dataset.
## **Job Configuration**

Configuration details for jobs depending on the model type:

### **`Jobs VAE`**
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
  - **training VAE**: No effect.
  - **training LDM or CN**: Directory of the Chosen VAE training (**`train_xs`** or **`train_xl`**).
    - **`0`**:  **`train_xs`**
    - **`#`**: **`train_xl/#_patches`**
  - **example for better results**: VAE from **`train_xs`**, LDM from **`train_xl`** and CN from **`train_xs`**.


### **`Jobs LDM`**
- **`cp`**: 
  - **training VAE**: No effect.
  - **training LDM**: 
    - **`0`**: Start a fresh training.
    - **`other`**: Continue a LDM training from this Checkpoint.
  - **training CN**: 
    - **`0`**: Use the parameters of the last training epoch of the chosen LDM training as the LDM model parameters.
    - **`other`**: Use the parameter of a specific training epoch of chosen LDM; training the as the LDM model parameters.
- **`id`**: 
  - **training VAE or LDM**: No effect.
  - **training CN**: SLURM job ID of the Chosen LDM training.
- **`directory`**: 
  - **training VAE or LDM**: No effect.
  - **training CN**: Directory of the Chosen LDM training.
    - **`0`**:  **`train_xs`**
    - **`#`**: **`train_xl/#_patches`**

### **`Jobs CN`**
- **`cp`**: 
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
  - **training VAE, LDM or CN**: No effect.
  - **Inference**: Directory of the Chosen LDM training.
    - **`0`**: **`train_xs`**
    - **`#`**: **`train_xl/#_patches`**

## **Dataset Configuration**

- **`input_channels`**: Determines the channels of the prediction. Use only for VAE and LDM training **eg. `bg`**. 

- **`condition`**: Determines the embedding conditions for the CN **eg. `["raw", "gt"]`**.

## **`VAE`**

- Configuration parameters for the Visual Autoencoder model.
- Use only for training VAE.

#### **`autoencoder`**
#### **`discriminator`**
#### **`loss`**
#### **`optimizer`**
#### **`training`**
- **`n_epochs`**: Number of training epochs.
- **`val_interval`**: Validation interval.
- **`autoencoder_warm_up_n_epochs`**: -1
  - **`-1`**: No warm up
  - **`#`**: Number of warm up epochs
- **`num_example_images`**: Number of Images stored 
- **`num_epochs_checkpoint`**: 

### **`LDM`**
Configuration parameters for the Latent Diffusion Model.
#### **`unet`**
- **`spatial_dims`**: 2
- **`in_channels`**: 3
- **`out_channels`**: 3
- **`num_res_blocks`**: 2
- **`num_channels`**: [128, 256, 512]
- **`attention_levels`**: [false, true, true]
- **`num_head_channels`**: [0, 256, 512]

#### **`ddpm_scheduler`**
- **`num_train_timesteps`**: 1000
- **`schedule`**: "linear_beta"
- **`beta_start`**: 0.0015
- **`beta_end`**: 0.0195

#### **`optimizer`**
- **`lr`**: 0.0001
- **`scheduler`**:
  - **`mode`**: "min"
  - **`factor`**: 0.5
  - **`patience`**: 20

#### **`training`**
- **`n_epochs`**: 100
- **`val_interval`**: 2
- **`num_epochs_checkpoint`**: 5

#### **`sampling`**
- **`number_of_samples`**: 50
- **`num_inference_steps`**: 1000
- **`noise_shape`**: [1, 3, 64, 64]
- **`intermediate_steps`**: 100

### **`CN`**
Configuration parameters for the Control Net model.
#### **`cn`**
- **`spatial_dims`**: 2
- **`in_channels`**: 3
- **`num_res_blocks`**: 2
- **`num_channels`**: [128, 256, 512]
- **`attention_levels`**: [false, true, true]
- **`num_head_channels`**: [0, 256, 512]
- **`conditioning_embedding_num_channels`**: [16, 32, 64]
- **`conditioning_embedding_in_channels`**: 2

#### **`ddpm_scheduler`**
- **`num_train_timesteps`**: 1000

#### **`optimizer`**
- **`lr`**: 0.0001
- **`scheduler`**:
  - **`mode`**: "min"
  - **`factor`**: 0.5
  - **`patience`**: 20

#### **`training`**
- **`n_epochs`**: 100
- **`val_interval`**: 2
- **`num_epochs_checkpoint`**: 5

### **`CN_Inf`**


