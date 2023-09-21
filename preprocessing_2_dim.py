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
    def __init__(self, image_dir, annotation_dir, output_file):
        self.image_dir = image_dir
        self.annotation_dir = annotation_dir
        self.image_files = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.nii.gz')])
        self.annotation_files = sorted([os.path.join(annotation_dir, f) for f in os.listdir(annotation_dir) if f.endswith('.nii.gz')])
        self.output_file = output_file

    def process_and_save(self):
        # Assuming equal number of image and annotation files
        assert len(self.image_files) == len(self.annotation_files), "Mismatch in number of image and annotation files"

        buffer_image = []
        buffer_annotation = []

        with h5py.File(self.output_file, 'w') as f:
            dset_image = f.create_dataset('image_slices', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)
            dset_annotation = f.create_dataset('annotation_slices', (0, 256, 256), maxshape=(None, 256, 256), chunks=True, compression="gzip", compression_opts=9)

            for image_path, annotation_path in zip(self.image_files, self.annotation_files):
                # Handle image
                buffer_image.extend(self.process_single_nifti(image_path))
                # Handle annotation
                buffer_annotation.extend(self.process_single_nifti(annotation_path))

                # Save buffer if it's big enough
                if len(buffer_image) >= 30000:
                    self.save_buffer_to_dataset(dset_image, buffer_image)
                    self.save_buffer_to_dataset(dset_annotation, buffer_annotation)
                    buffer_image.clear()
                    buffer_annotation.clear()

            # If there's any remaining data in the buffers, save them
            if buffer_image:
                self.save_buffer_to_dataset(dset_image, buffer_image)
            if buffer_annotation:
                self.save_buffer_to_dataset(dset_annotation, buffer_annotation)

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
image_dir = os.path.join(args.data_path, 'images')
annotation_dir = os.path.join(args.data_path, 'annotation')

preprocessor = NiftiPreprocessor(image_dir=image_dir, annotation_dir=annotation_dir, output_file=args.output_file)
preprocessor.process_and_save()
