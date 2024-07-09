import os
import shutil
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--target_dir', type=str, required=True)
parser.add_argument('--bg', type=str, required=True)
parser.add_argument('--raw', type=str, required=True)
parser.add_argument('--ids', type=str, required=True)
parser.add_argument('--num_patches', type=int, required=True)
args = parser.parse_args()

def copy_files(src_dir, dst_dir, ids):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for id in ids:
        src_path = os.path.join(src_dir, id)
        shutil.copy(src_path, dst_dir)

if __name__ == "__main__":
    # Read IDs from the file
    with open(args.ids, 'r') as ids_file:
        ids_list = [id.strip() for id in ids_file.readlines()]

    # Collect all IDs that are present in both bg and raw directories
    common_ids = []
    for id in ids_list:
        if os.path.exists(os.path.join(args.bg, id)) and os.path.exists(os.path.join(args.raw, id)):
            common_ids.append(id)

    # Limit to the required number of patches
    used_ids = common_ids[:args.num_patches]
    print(len(common_ids))

    # Define target directories
    bg_dir = os.path.join(args.target_dir, 'bg')
    raw_dir = os.path.join(args.target_dir, 'raw')

    # Copy files for valid IDs
    copy_files(args.bg, bg_dir, used_ids)
    copy_files(args.raw, raw_dir, used_ids)

    # Save the valid IDs to a JSON file
    with open(os.path.join(args.target_dir, 'ids.json'), 'w') as json_file:
        json.dump(used_ids, json_file, indent=4)
