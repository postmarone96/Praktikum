import h5py
import numpy as np

class DebugFileCreator:
    def __init__(self, input_file, output_file, max_size_gb=1):
        self.input_file = input_file
        self.output_file = output_file
        self.max_size_bytes = max_size_gb * 1024**3  # Convert GB to bytes
        self.dim = 320

    def create_debug_file(self):
        with h5py.File(self.input_file, 'r') as fin, h5py.File(self.output_file, 'w') as fout:
            dsets = {}
            buffers = {key: [] for key in fin.keys()}
            total_size = 0

            # Create datasets in the output file with the same structure
            for key in fin.keys():
                dset_shape = (0, self.dim, self.dim)
                maxshape = (None, self.dim, self.dim)
                dsets[key] = fout.create_dataset(
                    key, shape=dset_shape, maxshape=maxshape,
                    chunks=True, compression="gzip"
                )

            # Sample patches from each dataset until we reach approximately 1GB
            i = 0
            while total_size < self.max_size_bytes:
                for key in fin.keys():
                    if i < len(fin[key]):
                        patch = fin[key][i]
                        buffers[key].append(patch)
                        total_size += patch.nbytes
                i += 1

            # Save the sampled data into the debug HDF5 file
            for key, buffer in buffers.items():
                buffer = np.stack(buffer, axis=0)
                dsets[key].resize((buffer.shape[0], self.dim, self.dim))
                dsets[key][:] = buffer

        print(f"Debug file created at {self.output_file}, size: {total_size / (1024**3):.2f} GB")

if __name__ == '__main__':
    input_file = '/home/viro/marouane.hajri/Praktikum/metrics_cn/metrics_dataset.hdf5'
    output_file = 'debug.hdf5'
    debug_creator = DebugFileCreator(input_file, output_file)
    debug_creator.create_debug_file()