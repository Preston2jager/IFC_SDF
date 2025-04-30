import os
import yaml 
import torch
import shutil

from gcn_runner import GCN_Runner

import m01_Config_Files
import m02_Data_Files.d06_SDF_Ready
import m02_Data_Files.d07_GCN_Results

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def copy_json_files_from_folder(folder_path, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename in os.listdir(folder_path):
        if filename.endswith('.json'):
            source_file = os.path.join(folder_path, filename)
            if os.path.isfile(source_file):
                shutil.copy2(source_file, target_dir)
                print(f"Copied: {source_file}")

if __name__=='__main__':
    data_folder = os.path.dirname(m02_Data_Files.d06_SDF_Ready.__file__)
    target_folder = os.path.dirname(m02_Data_Files.d07_GCN_Results.__file__)
    train_cfg_path = os.path.join(os.path.dirname(m01_Config_Files.__file__), 'training.yaml')
    
    with open(train_cfg_path, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)
    
    Runner = GCN_Runner(train_cfg)
    copy_json_files_from_folder(data_folder, target_folder)
    Runner.train() 
