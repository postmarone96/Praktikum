import os
import numpy as np
import nibabel as nib
import h5py
import argparse
import psutil

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
args = parser.parse_args()

class NiftiPreprocessor:
    def __init__(self, bg_dir, raw_dir, output_file):
        self.bg_dir = bg_dir
        self.raw_dir = raw_dir
        self.bg_files = sorted([os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith('.nii.gz')])
        self.raw_files = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.nii.gz')])
        self.output_file = output_file

    def process_and_save(self):
        # Assuming equal number of image and annotation files
        assert len(self.bg_files) == len(self.raw_files), "Mismatch in number of image and annotation files"

        buffer_bg = []
        buffer_raw = []

        with h5py.File(self.output_file, 'w') as f:
            dset_bg = f.create_dataset('raw', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)
            dset_raw = f.create_dataset('bg', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)

            for bg_path, raw_path in zip(self.bg_files, self.raw_files):
                # Handle image
                buffer_bg.extend(self.process_single_nifti(bg_path))
                # Handle annotation
                buffer_raw.extend(self.process_single_nifti(raw_path))

                # Save buffer if it's big enough
                if len(buffer_bg) >= 30000:
                    self.save_buffer_to_dataset(dset_bg, buffer_bg)
                    self.save_buffer_to_dataset(dset_raw, buffer_raw)
                    buffer_bg.clear()
                    buffer_raw.clear()

            # If there's any remaining data in the buffers, save them
            if buffer_bg:
                self.save_buffer_to_dataset(dset_bg, buffer_bg)
            if buffer_raw:
                self.save_buffer_to_dataset(dset_raw, buffer_raw)

    def process_single_nifti(self, nii_path):
        img = nib.load(nii_path)
        image_data = img.get_fdata()
        image_data = np.moveaxis(image_data, -1, 0)
        image_data = image_data.astype(np.float32)
        buffer = []
        for img in image_data:
            max_value = np.max(img)
            img /= max_value
            img_cropped = img[0:256, 0:256]
            buffer.append(img_cropped)
        return buffer

    def save_buffer_to_dataset(self, dataset, buffer):
        current_length = dataset.shape[0]
        dataset.resize((current_length + len(buffer), 256, 256))
        dataset[current_length:current_length + len(buffer)] = np.array(buffer)

# Assuming data_path points to a parent directory that contains 'C00' and 'C02' subdirectories
bg_dir = os.path.join(args.data_path, 'bg')
raw_dir = os.path.join(args.data_path, 'raw')

preprocessor = NiftiPreprocessor(bg_dir=bg_dir, raw_dir=raw_dir, output_file=args.output_file)
preprocessor.process_and_save()