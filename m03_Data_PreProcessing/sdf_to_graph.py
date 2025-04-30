import os
import datetime
import shutil
import json
import numpy as np

import m02_Data_Files.d06_SDF_selected
import m02_Data_Files.d04_SDF_Results.runs_sdf
import m02_Data_Files.d05_Graph

def parse_folder_time(folder_name):
    try:
        parts = folder_name.split('_')
        if len(parts) != 3:
            raise ValueError(f"文件夹名格式错误: {folder_name}")
        day = int(parts[0])
        month = int(parts[1])
        hour = int(parts[2][0:2])
        minute = int(parts[2][2:4])
        second = int(parts[2][4:6])
        year = datetime.datetime.now().year
        return datetime.datetime(year, month, day, hour, minute, second)
    except Exception as e:
        print(f"解析时间戳失败：{folder_name}，错误：{e}")
        return None
    
def find_nearest_folder(base_dir):
    """Find the lastest folder"""
    now = datetime.datetime.now()
    folders = []
    for name in os.listdir(base_dir):
        full_path = os.path.join(base_dir, name)
        if os.path.isdir(full_path):
            folder_time = parse_folder_time(name)
            if folder_time:
                folders.append((folder_time, full_path))
    if not folders:
        print("No valid folder found")
        return None

    folders.sort(key=lambda x: abs((x[0] - now).total_seconds()))
    return folders[0][1]  # return path

def copy_files_from_folder(folder_path, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)
    for filename in os.listdir(folder_path):
        source_file = os.path.join(folder_path, filename)
        if os.path.isfile(source_file):
            shutil.copy2(source_file, target_dir)
            print(f"Copyed：{source_file}")

def generate_graph():
    base_path = os.path.dirname(m02_Data_Files.d06_SDF_selected.__file__)
    graph_path = os.path.dirname(m02_Data_Files.d05_Graph.__file__)
    idx_file = os.path.join(base_path, "idx_int2str_dict.npy")
    latent_file = os.path.join(base_path, "results.npy") 

    idx2name = np.load(idx_file, allow_pickle=True).item()
    latent_codes_raw = np.load(latent_file, allow_pickle=True)
    latent_codes_dict = latent_codes_raw.item()  
    latent_codes = latent_codes_dict["best_latent_codes"]

    if len(idx2name) != len(latent_codes):
        raise ValueError(f"idx2name数量({len(idx2name)})和latent code数量({len(latent_codes)})不一致！")
    name_to_latent = {}

    for idx, full_name in idx2name.items():
        if '_' in full_name:
            parts = full_name.split('_', 2)  
            if len(parts) == 3:
                global_id = parts[2]  
                name_to_latent[global_id] = latent_codes[idx]
            else:
                print(f"[警告] 解析失败 {full_name}")

    for filename in os.listdir(graph_path):
        if filename.endswith(".json"):
            filepath = os.path.join(graph_path, filename)

            with open(filepath, "r") as f:
                data = json.load(f)

            for node in data.get("nodes", []):
                global_id = node.get("GlobalId")
                #type_ = node.get("type")
                key = f"{global_id}"

                if key in name_to_latent:
                    latent_vector = name_to_latent[key]
                    node["latent_code"] = latent_vector.tolist()  # 转成list，json不能存numpy array
                else:
                    print(f"[警告] 没找到 {key}，跳过")

            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            print(f"已更新：{filename}")

def Delete_files(folder_path, file_type):
    """
    elete all files of a type in a folder.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(file_type):
            target_file_path = os.path.join(folder_path, filename)
            os.remove(target_file_path)
            print(f"File Deleted: {target_file_path}")

def main():
    SDF_folder_path = os.path.dirname(m02_Data_Files.d06_SDF_selected.__file__)
    Results_folder_path = os.path.dirname(m02_Data_Files.d04_SDF_Results.runs_sdf.__file__)
    Graph_folder_path = os.path.dirname(m02_Data_Files.d05_Graph.__file__)
    file_types=[".pt",".0",".npy",".yaml",".json"]
    for type in file_types:
        Delete_files(SDF_folder_path, type)
    print("Old files deleted")
    nearest_folder = find_nearest_folder(Results_folder_path)
    if nearest_folder:
        print(f"Use：{nearest_folder}")
        copy_files_from_folder(nearest_folder, SDF_folder_path)
        #copy_files_from_folder(Graph_folder_path, SDF_folder_path)
    generate_graph()

if __name__=='__main__':
    main()
    #cfg_path = os.path.join(os.path.dirname(m01_Config_Files.__file__), 'extracting.yaml')
    #with open(cfg_path, 'rb') as f:
        #cfg = yaml.load(f, Loader=yaml.FullLoader)
    #main(cfg)
