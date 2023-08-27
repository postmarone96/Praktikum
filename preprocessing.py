import os
import numpy as np
import nibabel as nib
import h5py
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
args = parser.parse_args()

class NiftiPreprocessor:
    def __init__(self, root_dir, output_file):
        self.root_dir = root_dir
        self.nii_files = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if f.endswith('.nii.gz')]
        self.output_file = output_file

    def process_and_save(self):
        # Compute total slices for the progress bar
        total_slices = 0
        for nii_path in self.nii_files:
            img = nib.load(nii_path)
            total_slices += img.shape[2]  # Assuming images are HxWxD

        with tqdm(total=total_slices, desc="Processing", unit="slice") as pbar:
            with h5py.File(self.output_file, 'w') as f:
                # Create initial empty dataset
                dset = f.create_dataset('all_slices', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)

                for nii_path in self.nii_files:
                    img = nib.load(nii_path)
                    image_data = img.get_fdata()
                    image_data = np.moveaxis(image_data, -1, 0)
                    image_data = image_data.astype(np.float32)

                    for img in image_data:
                        max_value = np.max(img)
                        img /= max_value
                        img_cropped = img[0:256, 0:256]
                        
                        # Dynamically resizing the dataset for every slice
                        dset.resize((dset.shape[0] + 1, 256, 256))
                        dset[-1] = img_cropped

                        # Update progress bar
                        pbar.update(1)

preprocessor = NiftiPreprocessor(root_dir=args.data_path, output_file=args.output_file)
preprocessor.process_and_save()
