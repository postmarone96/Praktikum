import h5py
import argparse
import os

# parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

def combine_hdf5_files(source_files, output_file):
    with h5py.File(output_file, 'w') as f_out:
        for ds_name, file_path in source_files.items():
            with h5py.File(file_path, 'r') as f_in:
                # Assuming the dataset inside each file has the same name as the file
                data = f_in[ds_name][:]
                f_out.create_dataset(ds_name, data=data)
                print(f"Dataset '{ds_name}' from '{file_path}' added to '{output_file}'.")

# Define your source files and corresponding dataset names
source_files = {
    'bg': os.path.join(args.data_path, 'datasetxy.hdf5'),
    'bg': os.path.join(args.data_path, 'datasetxz.hdf5'),
    'bg': os.path.join(args.data_path, 'datasetyz.hdf5'),
    'gt': os.path.join(args.data_path, 'datasetxy.hdf5'),
    'gt': os.path.join(args.data_path, 'datasetxz.hdf5'),
    'gt': os.path.join(args.data_path, 'datasetyz.hdf5'),
    'raw': os.path.join(args.data_path, 'datasetxy.hdf5'),
    'raw': os.path.join(args.data_path, 'datasetxz.hdf5'),
    'raw': os.path.join(args.data_path, 'datasetyz.hdf5')
}

# Output file
output_file = os.path.join(args.data_path, 'dataset.hdf5')

# Combine files
combine_hdf5_files(source_files, output_file)