import h5py
import argparse
import os
import numpy as np

# Parser
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

def append_and_combine_hdf5_files(dataset_name, file_paths, output_file):
    combined_data = []
    for file_path in file_paths:
        with h5py.File(file_path, 'r') as f_in:
            data = f_in[dataset_name][:]
            print(f"Reading from {file_path}, Dataset '{dataset_name}', Shape: {data.shape}, Dtype: {data.dtype}")
            combined_data.append(data)

    # Combine the data from all files
    combined_data = np.concatenate(combined_data, axis=0)
    print(f"Combined Data Shape: {combined_data.shape}, Dtype: {combined_data.dtype}")

    with h5py.File(output_file, 'a') as f_out:  # 'a' mode to append to existing file
        f_out.create_dataset(dataset_name, data=combined_data)
        print(f"Dataset '{dataset_name}' combined and added to '{output_file}'.")

# Paths to the datasets
datasets = {
    'bg': ['datasetxy.hdf5', 'datasetxz.hdf5', 'datasetyz.hdf5'],
    'gt': ['datasetxy.hdf5', 'datasetxz.hdf5', 'datasetyz.hdf5'],
    'raw': ['datasetxy.hdf5', 'datasetxz.hdf5', 'datasetyz.hdf5']
}

# Output file
output_file = os.path.join(args.data_path, 'dataset.hdf5')

# Combine datasets
for ds_name, files in datasets.items():
    file_paths = [os.path.join(args.data_path, f) for f in files]
    append_and_combine_hdf5_files(ds_name, file_paths, output_file)
