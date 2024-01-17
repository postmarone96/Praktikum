import os
import numpy as np
import nibabel as nib
import h5py
import argparse
import gc

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--data_size", type=str, required=True)
args = parser.parse_args()

class NiftiPreprocessor:
    def __init__(self, raw_dir, bg_dir, gt_dir, output_file, data_size):
        print("Initializing NiftiPreprocessor")
        self.data_size = data_size
        self.raw = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.nii.gz')])
        self.bg = sorted([os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith('.nii.gz')])
        if self.data_size == 'xs':
            self.gt = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.nii.gz')])
        self.output_file = output_file
        print(f"Found {len(self.raw)} raw files, {len(self.bg)} background files.")

    def calculate_percentile_threshold(self, data_paths):
        percentiles = []
        for nii_path in data_paths:
            img = nib.load(nii_path)
            image_data = img.get_fdata()
            percentile_25 = np.percentile(image_data, 25)
            percentiles.append(percentile_25)
            print(f"Calculated 25th percentile for {nii_path}: {percentile_25}")

        overall_percentile = np.median(percentiles)
        print(f"Overall 25th percentile threshold: {overall_percentile}")
        return overall_percentile

    def process_and_save(self):
        assert len(self.raw) == len(self.bg), "Mismatch in number files"
        if self.data_size == 'xs':
            assert len(self.raw) == len(self.gt), "Mismatch in number gt files "

        percentile_threshold = self.calculate_percentile_threshold(self.raw)

        with h5py.File(self.output_file, 'w') as f:
            dset_raw = f.create_dataset('raw', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)
            dset_bg = f.create_dataset('bg', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)
            if self.data_size == 'xs':
                dset_gt = f.create_dataset('gt', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)

            for idx, (raw_path, bg_path) in enumerate(zip(self.raw, self.bg)):
                print(f"Processing file {idx+1}/{len(self.raw)}: {os.path.basename(raw_path)}")
                raw_buffer, valid_indices = self.process_single_nifti(raw_path, percentile_threshold)
                self.save_slices_to_dataset(dset_raw, raw_buffer['xy'])
                self.save_slices_to_dataset(dset_raw, raw_buffer['xz'])
                self.save_slices_to_dataset(dset_raw, raw_buffer['yz'])
                gc.collect()  # Trigger garbage collection

                bg_buffer = self.process_single_nifti_using_indices(bg_path, valid_indices)
                self.save_slices_to_dataset(dset_bg, bg_buffer['xy'])
                self.save_slices_to_dataset(dset_bg, bg_buffer['xz'])
                self.save_slices_to_dataset(dset_bg, bg_buffer['yz'])
                gc.collect()  # Trigger garbage collection

                if self.data_size == 'xs':
                    gt_buffer = self.process_mask_nifti_using_indices(self.gt[self.raw.index(raw_path)], valid_indices)
                    self.save_slices_to_dataset(dset_gt, gt_buffer['xy'])
                    self.save_slices_to_dataset(dset_gt, gt_buffer['xz'])
                    self.save_slices_to_dataset(dset_gt, gt_buffer['yz'])
                    gc.collect()

            print("All files processed and saved.")

    def save_slices_to_dataset(self, dataset, slices):
        current_length = dataset.shape[0]
        dataset.resize((current_length + len(slices), 256, 256))
        dataset[current_length:] = np.array(slices)
        print(f"Total dataset size: {dataset.shape[0]} slices.")

    def process_single_nifti_using_indices(self, nii_path, valid_indices):
        img = nib.load(nii_path)
        image_data = img.get_fdata()
        buffer = {'xy': [], 'xz': [], 'yz': []}
        for plane in ['xy', 'xz', 'yz']:
            slices = self.get_slices(image_data, plane)
            buffer[plane].extend(self.process_slices(slices, valid_indices[plane], plane))
        return buffer

    def process_mask_nifti_using_indices(self, nii_path, valid_indices):
        img = nib.load(nii_path)
        image_data = img.get_fdata()
        buffer = {'xy': [], 'xz': [], 'yz': []}
        for plane in ['xy', 'xz', 'yz']:
            slices = self.get_slices(image_data, plane)
            buffer[plane].extend(self.process_slices_for_masks(slices, valid_indices[plane], plane))
        return buffer

    def process_slices(self, slices, idx, plane):
        buffer = []
        for i in idx:
            img = slices[i]
            max_value = np.max(img)
            img /= max_value
            img_cropped = img[:256, :256]  # Adjust based on your dimensions
            buffer.append(img_cropped)
        return buffer

    def process_slices_for_masks(self, slices, idx, plane):
        buffer = []
        for i in idx:
            img = slices[i]
            max_value = np.max(img)
            img /= max_value
            img_cropped = img[:256, :256]  # Adjust based on your dimensions
            img_cropped = (img_cropped > 0.2).astype(np.float32)
            buffer.append(img_cropped)
        return buffer

    def process_single_nifti(self, nii_path, percentile_threshold):
        img = nib.load(nii_path)
        image_data = img.get_fdata()

        valid_indices = {'xy': [], 'xz': [], 'yz': []}
        buffer = {'xy': [], 'xz': [], 'yz': []}

        for plane in ['xy', 'xz', 'yz']:
            slices = self.get_slices(image_data, plane)
            for i, img in enumerate(slices):
                avg_intensity = np.mean(img)
                if avg_intensity >= percentile_threshold:
                    valid_indices[plane].append(i)
                    max_value = np.max(img)
                    img /= max_value
                    img_cropped = img[:256, :256]  # Adjust based on your dimensions
                    buffer[plane].append(img_cropped)

        return buffer, valid_indices

    def get_slices(self, image_data, plane):
        if plane == 'xy':
            return np.moveaxis(image_data, -1, 0)
        elif plane == 'xz':
            return np.moveaxis(image_data, 0, 1)
        else:  # 'yz'
            return np.moveaxis(image_data, 0, 2)

preprocessor = NiftiPreprocessor(raw_dir=os.path.join(args.data_path, 'raw'),
                                bg_dir=os.path.join(args.data_path, 'bg'),
                                gt_dir=os.path.join(args.data_path, 'gt'), 
                                output_file=args.output_file,
                                data_size=args.data_size)
preprocessor.process_and_save()
