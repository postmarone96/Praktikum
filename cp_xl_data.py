import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--bg', type=str, required=True)
parser.add_argument('--raw', type=str, required=True)
args = parser.parse_args()

ids_file_path = os.path.join(os.getcwd(), 'ids.txt')

def copy_directory(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    with open(ids_file_path, 'r') as ids_file:
        for id in ids_file:
            id = id.strip()
            src_path = os.path.join(src_dir, id)
            shutil.copy(src_path, dst_dir)

if __name__ == "__main__":
    
    bg_dir = os.path.join(os.getcwd(), 'train_xl/data/bg')
    raw_dir = os.path.join(os.getcwd(), 'train_xl/data/raw')
    
    copy_directory('', bg_dir)
    copy_directory('', raw_dir)
