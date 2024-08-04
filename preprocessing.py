import os
import numpy as np
import nibabel as nib
import h5py
import argparse
import json

def get_file_paths(directory, suffix='.nii.gz'):
    return sorted([os.path.join(directory, f) for f in os.listdir(directory) if f.endswith(suffix)])

class NiftiPreprocessor:
    def __init__(self, args, config):
        self.args = args
        self.dim = config['data']['dim']
        self.pad = config['data']['pad']
        self.size = config['data']['size']
        self.gt_th = config['data']['gt_th']
        self.raw = get_file_paths(os.path.join(args.data_path, 'raw'))
        self.bg = get_file_paths(os.path.join(args.data_path, 'bg'))
        self.gt = get_file_paths(os.path.join(args.data_path, 'gt')) if self.size == 'xs' else []

    def load_and_preprocess_nifti(self, path, threshold=None):
        img = nib.load(path).get_fdata()
        img = np.moveaxis(img, -1, 0)  # Reorder to z, x, y
        if threshold is not None:  # Applying threshold for ground truth images
            img = np.array([self.threshold_gt(slice, threshold) for slice in img])
        else:  # Normalize non-GT images
            img = np.array([self.normalize_slice(slice) for slice in img])
        return img

    def normalize_slice(self, slice):
        slice = np.pad(slice, pad_width=self.pad, mode='reflect')
        return slice / np.max(slice)

    def threshold_gt(self, slice, threshold):
        slice = np.pad(slice, pad_width=self.pad, mode='reflect')
        return (slice > threshold).astype(np.float32)
        
    def process_and_save(self):
        assert len(self.raw) == len(self.bg), "Mismatch in number of files"
        if self.size == 'xs':
            assert len(self.raw) == len(self.gt), "Mismatch in number of gt files"
        
        with h5py.File(self.args.output_file, 'w') as f:
            dsets = {
                'raw': f.create_dataset('raw', (0, self.dim, self.dim), maxshape=(None, self.dim, self.dim), chunks=True, compression="gzip"),
                'bg': f.create_dataset('bg', (0, self.dim, self.dim), maxshape=(None, self.dim, self.dim), chunks=True, compression="gzip"),
                'gt': f.create_dataset('gt', (0, self.dim, self.dim), maxshape=(None, self.dim, self.dim), chunks=True, compression="gzip") if self.size == 'xs' else None
            }
            buffers = {key: [] for key in dsets if dsets[key]}

            for raw_path, bg_path in zip(self.raw, self.bg):
                buffers['raw'].extend(self.load_and_preprocess_nifti(raw_path))
                buffers['bg'].extend(self.load_and_preprocess_nifti(bg_path))
                if self.size == 'xs':
                    gt_path = self.gt[self.raw.index(raw_path)]
                    buffers['gt'].extend(self.load_and_preprocess_nifti(gt_path, threshold=self.gt_th))

                for key, buffer in buffers.items():
                    if len(buffer) >= 30000:
                        self.save_buffer_to_dataset(dsets[key], buffer)
                        buffer.clear()

            for key, buffer in buffers.items():
                if buffer:
                    self.save_buffer_to_dataset(dsets[key], buffer)

    def save_buffer_to_dataset(self, dataset, buffer):
        dataset.resize((dataset.shape[0] + len(buffer), self.dim, self.dim))
        dataset[-len(buffer):] = buffer

if __name__ == '__main__':
    # Args parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    args = parser.parse_args()

    # JSON parser
    with open('params.json') as json_file:
        config = json.load(json_file)

    # Run preprocessing and create dataset.hdf5 
    preprocessor = NiftiPreprocessor(args, config)
    preprocessor.process_and_save()
