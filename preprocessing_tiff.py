import os
import numpy as np
import h5py
from PIL import Image

def get_tiff_file_paths(directory):
    """Retrieve sorted .tif and .tiff file paths in a directory."""
    tiff_paths = []
    for root, _, files in os.walk(directory):
        for f in files:
            if f.lower().endswith('.tiff') or f.lower().endswith('.tif'):
                tiff_paths.append(os.path.join(root, f))
    return sorted(tiff_paths)

def map_file_paths(raw_paths, bg_paths, gt_paths):
    """Map file paths based on relative paths, keeping only those that have corresponding bg and gt files."""
    mapping = []
    for raw_path in raw_paths:
        relative_path = os.path.relpath(raw_path, start=os.path.dirname(raw_paths[0]))
        bg_path = os.path.join(os.path.dirname(bg_paths[0]), relative_path)
        gt_path = os.path.join(os.path.dirname(gt_paths[0]), relative_path)
        
        # Only add paths if corresponding bg and gt files exist
        if os.path.exists(bg_path) and os.path.exists(gt_path):
            mapping.append((raw_path, bg_path, gt_path))
    return mapping

class ImagePreprocessor:
    def __init__(self):
        self.data_path = '/lustre/groups/iterm/Rami/HFD/trigeminal_cuts_full'
        self.output_file = '/home/viro/marouane.hajri/Praktikum/metrics_cn/metrics_dataset.hdf5'
        self.dim = 320
        self.pad = 10
        self.gt_th = 0.5
        self.raw = get_tiff_file_paths(os.path.join(self.data_path, 'raw_cutouts'))
        self.bg = get_tiff_file_paths(os.path.join(self.data_path, 'bg_cutouts'))
        self.gt = get_tiff_file_paths(os.path.join(self.data_path, 'segmentation'))

        # Map files based on relative paths, keeping only complete sets
        self.file_mappings = map_file_paths(self.raw, self.bg, self.gt)

        # Assert only after filtering to ensure matching counts
        assert len(self.file_mappings) > 0, "No matching file sets found across raw, bg, and gt directories."

    def load_and_preprocess_image(self, path, threshold=None):
        img = Image.open(path)
        img_array = np.array(img)
        patches = self.cut_into_patches(img_array)
        if threshold is not None:
            patches = [self.threshold_patch(patch, threshold) for patch in patches]
        else:
            patches = [self.normalize_patch(patch) for patch in patches]
        return patches

    def cut_into_patches(self, img_array):
        patches = []
        h, w = img_array.shape
        h_steps = h // 300
        w_steps = w // 300
        for i in range(h_steps):
            for j in range(w_steps):
                patch = img_array[i*300:(i+1)*300, j*300:(j+1)*300]
                patches.append(patch)
        return patches

    def pad_patch(self, patch):
        return np.pad(patch, pad_width=self.pad, mode='reflect')

    def normalize_patch(self, patch):
        patch = self.pad_patch(patch)
        max_val = np.max(patch)
        if max_val > 0:
            patch = patch / max_val
        return patch.astype(np.float32)

    def threshold_patch(self, patch, threshold):
        patch = self.pad_patch(patch)
        return (patch > threshold).astype(np.float32)

    def process_and_save(self):
        with h5py.File(self.output_file, 'w') as f:
            dsets = {
                'raw': f.create_dataset('raw', (0, self.dim, self.dim), maxshape=(None, self.dim, self.dim),
                                        chunks=True, compression="gzip"),
                'bg': f.create_dataset('bg', (0, self.dim, self.dim), maxshape=(None, self.dim, self.dim),
                                       chunks=True, compression="gzip"),
                'gt': f.create_dataset('gt', (0, self.dim, self.dim), maxshape=(None, self.dim, self.dim),
                                       chunks=True, compression="gzip")
            }
            buffers = {key: [] for key in dsets}

            for mapping in self.file_mappings:
                raw_patches = self.load_and_preprocess_image(mapping[0])
                bg_patches = self.load_and_preprocess_image(mapping[1])
                gt_patches = self.load_and_preprocess_image(mapping[2], threshold=self.gt_th)

                buffers['raw'].extend(raw_patches)
                buffers['bg'].extend(bg_patches)
                buffers['gt'].extend(gt_patches)

                # Save buffers if they reach a certain size
                for key, buffer in buffers.items():
                    if len(buffer) >= 30000:
                        self.save_buffer_to_dataset(dsets[key], buffer)
                        buffer.clear()

            # Save any remaining data
            for key, buffer in buffers.items():
                if buffer:
                    self.save_buffer_to_dataset(dsets[key], buffer)

    def save_buffer_to_dataset(self, dataset, buffer):
        buffer = np.stack(buffer, axis=0)
        dataset.resize((dataset.shape[0] + buffer.shape[0], self.dim, self.dim))
        dataset[-buffer.shape[0]:] = buffer

if __name__ == '__main__':
    preprocessor = ImagePreprocessor()
    preprocessor.process_and_save()