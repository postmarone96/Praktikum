import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--target_dir', type=str, required=True)
parser.add_argument('--bg', type=str, required=True)
parser.add_argument('--raw', type=str, required=True)
parser.add_argument('--ids', type=str, required=True)
args = parser.parse_args()

ids_file_path = args.ids

def copy_files_from_directory(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    with open(ids_file_path, 'r') as ids_file:
        for id in ids_file:
            id = id.strip()
            src_path = os.path.join(src_dir, id)
            shutil.copy(src_path, dst_dir)

if __name__ == "__main__":
    
    bg_dir = os.path.join(args.target_dir, 'bg')
    raw_dir = os.path.join(args.target_dir, 'raw')
    
    copy_files_from_directory(args.bg, bg_dir)
    copy_files_from_directory(args.raw, raw_dir)
