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
    def __init__(self, raw_dir, bg_dir, gt_dir, output_file):
        self.raw = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.nii.gz')])
        self.bg = sorted([os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith('.nii.gz')])
        self.gt = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.nii.gz')])
        self.output_file = output_file

    def process_and_save(self):
        # Assuming equal number of image and annotation files
        assert len(self.raw) == len(self.bg), "Mismatch in number files"
        assert len(self.raw) == len(self.gt), "Mismatch in number gt files"

        buffer_raw = []
        buffer_bg = []
        buffer_gt = []

        with h5py.File(self.output_file, 'w') as f:
            dset_raw = f.create_dataset('raw', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)
            dset_bg = f.create_dataset('bg', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)
            dset_gt = f.create_dataset('gt', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)

            for raw_path, bg_path, gt_path in zip(self.raw, self.bg, self.gt):
                # Handle raw
                buffer_raw.extend(self.process_single_nifti(raw_path))
                # Handle bg
                buffer_bg.extend(self.process_single_nifti(bg_path))
                # Handle gt
                buffer_gt.extend(self.process_single_nifti(gt_path))

                # Save buffer if it's big enough
                if len(buffer_raw) >= 30000:
                    self.save_buffer_to_dataset(dset_raw, buffer_raw)
                    self.save_buffer_to_dataset(dset_bg, buffer_bg)
                    self.save_buffer_to_dataset(dset_gt, buffer_gt)
                    buffer_raw.clear()
                    buffer_bg.clear()
                    buffer_gt.clear()

            # If there's any remaining data in the buffers, save them
            if buffer_raw:
                self.save_buffer_to_dataset(dset_raw, buffer_raw)
            if buffer_bg:
                self.save_buffer_to_dataset(dset_bg, buffer_bg)
            if buffer_gt:
                self.save_buffer_to_dataset(dset_gt, buffer_gt)

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
raw_dir = os.path.join(args.data_path, 'raw')
bg_dir = os.path.join(args.data_path, 'bg')
gt_dir = os.path.join(args.data_path, 'gt')

preprocessor = NiftiPreprocessor(raw_dir=raw_dir, bg_dir=bg_dir, gt_dir=gt_dir, output_file=args.output_file)
preprocessor.process_and_save()
