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

## **Job Configuration**

Configuration details for jobs depending on the model type:

### **`vae`, `ldm`, `cn`**
- **`cp`**: Checkpoint parameter to specify the training epoch to start from or to use an existing checkpoint.
  - **`0`**: Start fresh training or use the last training epoch checkpoint.
- **`id`**: SLURM job ID to identify specific checkpoints or configurations.
  - **`0`**: Start a new training session or no specific job ID.
- **`directory`**: Specifies the directory for training (`train_xs` or `train_xl`).
