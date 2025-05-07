import torch
import os
import sys
import yaml
import argparse

import numpy as np
from tqdm import tqdm

from utils.utils_deepsdf import SDFLoss_multishape_full_exp
from utils import utils_deepsdf

from sdf_runner import SDF_Runner

import model_sdf as sdf_model
import dataset_sdf as dataset
import m01_Config_Files
import m02_Data_Files.d04_SDF_Converted
import m02_Data_Files.d05_SDF_Results.runs_sdf as runs
import m02_Data_Files.d08_Predict_Data.d01_Config
import m02_Data_Files.d08_Predict_Data.d04_SDF

def copy_training_idx_files(Source_idx_int2str_path, Target_idx_int2str_path):
    if os.path.exists(Source_idx_int2str_path):
        idx_int2str_dict = np.load(Source_idx_int2str_path, allow_pickle=True).item()
        np.save(Target_idx_int2str_path, idx_int2str_dict)
        print(f'Saved idx_int2str_dict to {Target_idx_int2str_path}')
    else:
        print(f'Warning: {Source_idx_int2str_path} not found! idx_str2int_dict not saved.')

if __name__=='__main__':

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", help="Mode of operation, train or pred", nargs='?', default="train")
    args = parser.parse_args()

    if args.mode == "train":
        print("Extract for training data")
        data_folder_path = os.path.dirname(m02_Data_Files.d04_SDF_Converted.__file__)
        train_cfg_path = os.path.dirname(m01_Config_Files.__file__)
        train_cfg_file = os.path.join(train_cfg_path, 'training.yaml')
        Target_folder_path = os.path.dirname(runs.__file__)
        Source_idx_int2str_path = os.path.join(os.path.dirname(m02_Data_Files.d04_SDF_Converted.__file__), 'idx_int2str_dict.npy')
        Target_idx_int2str_path = os.path.join(Target_folder_path, 'idx_int2str_dict.npy')
        copy_training_idx_files(Source_idx_int2str_path, Target_idx_int2str_path) # Only copy in training
    elif args.mode == "pred":
        print("Extract for prediction data")
        train_cfg_path = os.path.dirname(m02_Data_Files.d08_Predict_Data.d01_Config.__file__)
        train_cfg_file = os.path.join(train_cfg_path, 'training.yaml')
    else:
        print("Incorrect mode, train or pred?")
        sys.exit(1)

    with open(train_cfg_file, 'rb') as f:
        train_cfg = yaml.load(f, Loader=yaml.FullLoader)
    Runner = SDF_Runner(train_cfg, args)

    if args.mode == "train":
        Runner.train_standard(data_folder_path)
    elif args.mode == "pred":
        Runner()