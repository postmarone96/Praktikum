import os
import numpy as np
import nibabel as nib
import h5py
import argparse
import gc

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--data_size", type=str, required=True)
args = parser.parse_args()

class NiftiPreprocessor:
    def __init__(self, raw_dir, bg_dir, gt_dir, output_file, data_size):
        self.data_size = data_size
        self.raw = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.nii.gz')])
        self.bg = sorted([os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith('.nii.gz')])
        if self.data_size == 'xs':
            self.gt = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.nii.gz')])
        self.output_file = output_file
        
    def calculate_percentile_threshold(self, data_paths):
        all_intensities = []
        for nii_path in data_paths:
            img = nib.load(nii_path)
            image_data = img.get_fdata()
            all_intensities.extend(image_data.flatten())
        return np.percentile(all_intensities, 25)

    def process_and_save(self):
        
        # Assuming equal number of image and annotation files
        assert len(self.raw) == len(self.bg), "Mismatch in number files"
        if self.data_size == 'xs':
            assert len(self.raw) == len(self.gt), "Mismatch in number gt files "

        percentile_threshold = self.calculate_percentile_threshold(self.raw)
        buffer_raw = []
        buffer_bg = []
        if self.data_size == 'xs':
            buffer_gt = []

        with h5py.File(self.output_file, 'w') as f:
            dset_raw = f.create_dataset('raw', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)
            dset_bg = f.create_dataset('bg', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)
            if self.data_size == 'xs':
                dset_gt = f.create_dataset('gt', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)

            for raw_path, bg_path in zip(self.raw, self.bg):
                raw_slices, valid_indices = self.process_single_nifti(raw_path, percentile_threshold)
                self.save_slices_to_dataset(dset_raw, raw_slices)
                gc.collect()  # Trigger garbage collection

                bg_slices = self.process_single_nifti_using_indices(bg_path, valid_indices)
                self.save_slices_to_dataset(dset_bg, bg_slices)
                gc.collect()  # Trigger garbage collection

                if self.data_size == 'xs':
                    gt_path = self.gt[self.raw.index(raw_path)]
                    gt_slices = self.process_mask_nifti_using_indices(gt_path, valid_indices)
                    self.save_slices_to_dataset(dset_gt, gt_slices)
                    gc.collect()

    def save_slices_to_dataset(self, dataset, slices):
        current_length = dataset.shape[0]
        dataset.resize((current_length + len(slices), 256, 256))
        dataset[current_length:] = np.array(slices)

    def process_single_nifti_using_indices(self, nii_path, valid_indices):
        img = nib.load(nii_path)
        image_data = img.get_fdata()
        slices_xy = np.moveaxis(image_data, -1, 0)
        # slices_zy = np.moveaxis(image_data, 0, 1)
        # slices_xz = np.moveaxis(image_data, 0, 2)
        return self.process_slices(slices_xy, valid_indices) # + self.process_slices(slices_zy) + self.process_slices(slices_xz)

    def process_mask_nifti_using_indices(self, nii_path, valid_indices):
        img = nib.load(nii_path)
        image_data = img.get_fdata()
        slices_xy = np.moveaxis(image_data, -1, 0)
        #slices_zy = np.moveaxis(image_data, 0, 1)
        #slices_xz = np.moveaxis(image_data, 0, 2)
        return self.process_slices_for_masks(slices_xy, valid_indices) #+ self.process_slices_for_masks(slices_zy) + self.process_slices_for_masks(slices_xz)

    def process_slices(self, slices, idx):
        buffer = []
        for i in idx:
            img = slices[i]
            max_value = np.max(img)
            img /= max_value
            img_cropped = img[0:256, 0:256]
            buffer.append(img_cropped)
        return buffer

    def process_slices_for_masks(self, slices, idx):
        buffer = []
        for i in idx:
            img = slices[i]
            max_value = np.max(img)
            img /= max_value
            img_cropped = img[0:256, 0:256]
            img_cropped = (img_cropped > 0.5).astype(np.float32)
            buffer.append(img_cropped)
        return buffer

    def process_single_nifti(self, nii_path, percentile_threshold):
        img = nib.load(nii_path)
        image_data = img.get_fdata()
        slices_xy = np.moveaxis(image_data, -1, 0)

        valid_indices = []
        buffer = []
        for i, img in enumerate(slices_xy):
            avg_intensity = np.mean(img)
            if avg_intensity >= percentile_threshold:
                valid_indices.append(i)
                max_value = np.max(img)
                img /= max_value
                img_cropped = img[0:256, 0:256]
                buffer.append(img_cropped)
        return buffer, valid_indices


preprocessor = NiftiPreprocessor(raw_dir=os.path.join(args.data_path, 'raw'),
                                bg_dir=os.path.join(args.data_path, 'bg'),
                                gt_dir=os.path.join(args.data_path, 'gt'), 
                                output_file=args.output_file,
                                data_size=args.data_size)
preprocessor.process_and_save()
