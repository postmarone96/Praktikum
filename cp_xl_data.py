import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--target_dir', type=str, required=True)
parser.add_argument('--bg', type=str, required=True)
parser.add_argument('--raw', type=str, required=True)
parser.add_argument('--ids', type=str, required=True)
parser.add_argument('--num_patches', type=str, required=True)
args = parser.parse_args()

ids_file_path = args.ids

def copy_files_from_directory(src_dir, dst_dir, ids, num_patches):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    used_ids = []
    for id in ids[:num_patches]:
        id = id.strip()
        src_path = os.path.join(src_dir, id)
        if os.path.exists(src_path):  # Check if the source file exists
            shutil.copy(src_path, dst_dir)
            used_ids.append(id)
    return used_ids

if __name__ == "__main__":
    # Read IDs from the file
    with open(args.ids, 'r') as ids_file:
        ids_list = ids_file.readlines()
    
    # Define target directories
    bg_dir = os.path.join(args.target_dir, 'bg')
    raw_dir = os.path.join(args.target_dir, 'raw')
    
    # Copy files and get the used IDs
    used_ids = copy_files_from_directory(args.bg, bg_dir, ids_list, args.num_patches)
    copy_files_from_directory(args.raw, raw_dir, ids_list, args.num_patches)
    
    # Save the used IDs to a JSON file
    with open(os.path.join(args.target_dir, 'ids.json'), 'w') as json_file:
        json.dump(used_ids, json_file, indent=4)

