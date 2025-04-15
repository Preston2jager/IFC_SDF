import ifcopenshell
import yaml
import os

from utils_data_preprocessing import *

import m01_Config_Files
import m02_Data_Files.d01_Raw_IFC
import m02_Data_Files.d01_Raw_IFC.d01_Expanded
import m02_Data_Files.d02_Object_Files
import m02_Data_Files.d05_Graph.json



import utils.utils_obj as object_randomise
import utils.utils_ifc as ifc_split

def delete_files(folder_path, file_type):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(file_type):
            target_file_path = os.path.join(folder_path, filename)
            os.remove(target_file_path)
            print(f"Deleted: {target_file_path}")

def main(cfg):

    # Define paths
    IFC_folder_path = os.path.dirname(m02_Data_Files.d01_Raw_IFC.__file__)
    Expanded_ifc_folder_path = os.path.dirname(m02_Data_Files.d01_Raw_IFC.d01_Expanded.__file__)
    Object_folder_path = os.path.dirname(m02_Data_Files.d02_Object_Files.__file__)
    Graph_folder_path = os.path.dirname(m02_Data_Files.d05_Graph.json.__file__)

    # Delete previous files
    delete_files(Expanded_ifc_folder_path, ".ifc")
    delete_files(Object_folder_path, ".obj")
    delete_files(Graph_folder_path, ".json")

    # Check for data expansion.
    if bool(cfg.get("data_expand")) is True:
        for filename in os.listdir(IFC_folder_path):
            if filename.lower().endswith('.ifc'):
                ifc_file_expand(cfg)
    else:
        for filename in os.listdir(IFC_folder_path):
            if filename.lower().endswith('.ifc'):
                source_file_path = os.path.join(IFC_folder_path, filename)
                target_file_path = os.path.join(Expanded_ifc_folder_path, filename)
                shutil.copy(source_file_path, target_file_path)
    
    ifc_split.regenerate_global_ids()

    index = 1
    for filename in os.listdir(ifc_file_path):
        if filename.lower().endswith('.ifc'):
            full_path = os.path.join(ifc_file_path, filename)
            ifc_file = ifcopenshell.open(full_path)
            settings = ifcopenshell.geom.settings() 
            settings.set(settings.USE_WORLD_COORDS, True)
            ifc_export_to_obj(ifc_file, settings, cfg, index) 
            extract_ifc_graph_json(index, full_path)
            index += 1
    
    if cfg.get("data_expand"):
        object_randomise.main(cfg)


if __name__=='__main__':
    cfg_path = os.path.join(os.path.dirname(m01_Config_Files.__file__), 'extracting.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)
