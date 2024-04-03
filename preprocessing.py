import os
import numpy as np
import nibabel as nib
import h5py
import argparse

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--output_file", type=str, required=True)
parser.add_argument("--data_size", type=str, required=True)
parser.add_argument("--gt_th", type=str, required=True)
args = parser.parse_args()

class NiftiPreprocessor:
    def __init__(self, raw_dir, bg_dir, gt_dir, output_file, data_size):
        self.data_size = data_size
        self.raw = sorted([os.path.join(raw_dir, f) for f in os.listdir(raw_dir) if f.endswith('.nii.gz')])
        self.bg = sorted([os.path.join(bg_dir, f) for f in os.listdir(bg_dir) if f.endswith('.nii.gz')])
        if self.data_size == 'xs':
            self.gt = sorted([os.path.join(gt_dir, f) for f in os.listdir(gt_dir) if f.endswith('.nii.gz')])
        self.output_file = output_file

    def process_and_save(self):
        # Assuming equal number of image and annotation files
        assert len(self.raw) == len(self.bg), "Mismatch in number files"
        if self.data_size == 'xs':
            assert len(self.raw) == len(self.gt), "Mismatch in number gt files "
            
        buffer_raw = []
        buffer_bg = []
        if self.data_size == 'xs':
            buffer_gt = []

        with h5py.File(self.output_file, 'w') as f:
            dset_raw = f.create_dataset('raw', (0, 320, 320), maxshape=(None, 320, 320), chunks=True, compression="gzip", compression_opts=9)
            dset_bg = f.create_dataset('bg', (0, 320, 320), maxshape=(None, 320, 320), chunks=True, compression="gzip", compression_opts=9)
            if self.data_size == 'xs':
                dset_gt = f.create_dataset('gt', (0, 320, 320), maxshape=(None, 320, 320), chunks=True, compression="gzip", compression_opts=9)

            for raw_path, bg_path in zip(self.raw, self.bg):
                buffer_raw.extend(self.process_single_nifti(raw_path))
                buffer_bg.extend(self.process_single_nifti(bg_path))
                if self.data_size == 'xs':
                    gt_path = self.gt[self.raw.index(raw_path)]  # Match raw and gt files
                    buffer_gt.extend(self.process_single_nifti_for_masks(gt_path))

                # Save buffer if it's big enough
                if len(buffer_raw) >= 30000:
                    self.save_buffer_to_dataset(dset_raw, buffer_raw)
                    buffer_raw.clear()
                if len(buffer_bg) >= 30000:
                    self.save_buffer_to_dataset(dset_bg, buffer_bg)
                    buffer_bg.clear()
                if self.data_size == 'xs'and len(buffer_gt) >= 30000:
                    self.save_buffer_to_dataset(dset_gt, buffer_gt)
                    buffer_gt.clear()
                        

            # If there's any remaining data in the buffers, save them
            if buffer_raw:
                self.save_buffer_to_dataset(dset_raw, buffer_raw)
            if buffer_bg:
                self.save_buffer_to_dataset(dset_bg, buffer_bg)
            if self.data_size == 'xs' and buffer_gt:
                self.save_buffer_to_dataset(dset_gt, buffer_gt)

    def process_single_nifti(self, nii_path):
        img = nib.load(nii_path)
        image_data = img.get_fdata()
        slices_xy = np.moveaxis(image_data, -1, 0)
        return self.process_slices(slices_xy)

    def process_single_nifti_for_masks(self, nii_path):   
        img = nib.load(nii_path)
        image_data = img.get_fdata()
        slices_xy = np.moveaxis(image_data, -1, 0)
        return self.process_slices_for_masks(slices_xy)

    def process_slices(self, slices):
        buffer = []
        for img in slices:
            img = np.pad(img, pad_width=10, mode='reflect')
            max_value = np.max(img)
            img /= max_value
            buffer.append(img_cropped)
        return buffer

    def process_slices_for_masks(self, slices):
        buffer = []
        for img in slices:
            img = np.pad(img, pad_width=10, mode='reflect')
            max_value = np.max(img)
            img /= max_value
            img = (img > args.gt_th).astype(np.float32)
            buffer.append(img)
        return buffer

    def save_buffer_to_dataset(self, dataset, buffer):
        current_length = dataset.shape[0]
        dataset.resize((current_length + len(buffer), 320, 320))
        dataset[current_length:current_length + len(buffer)] = np.array(buffer)

preprocessor = NiftiPreprocessor(raw_dir=os.path.join(args.data_path, 'raw'),
                                bg_dir=os.path.join(args.data_path, 'bg'),
                                gt_dir=os.path.join(args.data_path, 'gt'), 
                                output_file=args.output_file,
                                data_size=args.data_size)
preprocessor.process_and_save()
