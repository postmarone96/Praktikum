import os
import shutil

ids_file_path = os.path.join(os.environ['HOME'], 'Praktikum/ids.txt')

def copy_directory(src_dir, dst_dir):
    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)
    for id in open(ids_file_path, 'r'):
        src_path = os.path.join(src_dir, id.strip())
        dst_path = os.path.join(dst_dir, id.strip())
        shutil.copytree(src_path, dst_path)

if __name__ == "__main__":
    bg_dir = os.path.join(os.environ['HOME'], 'Praktikum/train_xl/data/bg')
    raw_dir = os.path.join(os.environ['HOME'], 'Praktikum/train_xl/data/raw')
    
    copy_directory('/lustre/groups/iterm/Rami/HFD/HFD_neurons/HFD/HFD_210320_UCHL1_755_HFD_DORSAL_l_1x_35o_4x8_GRB12-7-12_17-00-17/C00/', bg_dir)
    copy_directory('/lustre/groups/iterm/Rami/HFD/HFD_neurons/HFD/HFD_210320_UCHL1_755_HFD_DORSAL_l_1x_35o_4x8_GRB12-7-12_17-00-17/C02/', raw_dir)
