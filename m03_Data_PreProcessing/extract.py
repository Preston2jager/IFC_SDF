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

def main(cfg):

    # Extracting objects from IFC files

    # Define paths
    IFC_folder_path = os.path.dirname(m02_Data_Files.d01_Raw_IFC.__file__)
    Expanded_ifc_folder_path = os.path.dirname(m02_Data_Files.d01_Raw_IFC.d01_Expanded.__file__)
    Object_folder_path = os.path.dirname(m02_Data_Files.d02_Object_Files.__file__)
    Graph_folder_path = os.path.dirname(m02_Data_Files.d05_Graph.json.__file__)

    # Delete previous files
    Delete_files(Expanded_ifc_folder_path, ".ifc")
    Delete_files(Object_folder_path, ".obj")
    Delete_files(Graph_folder_path, ".json")

    # Copy raw IFC files into expanded folders.
    for filename in os.listdir(IFC_folder_path):
        if filename.lower().endswith('.ifc'):
            Source_file_path = os.path.join(IFC_folder_path, filename)
            Target_file_path = os.path.join(Expanded_ifc_folder_path, filename)
            shutil.copy(Source_file_path, Target_file_path)

    # Expand files if required
    if bool(cfg.get("data_expand")) is True:
        Num_of_copy = cfg.get("copies")
        for filename in os.listdir(IFC_folder_path):
            if filename.lower().endswith('.ifc'):
                IFC_file_expand(filename, Num_of_copy)

    # Generate new global id for all files
    for filename in os.listdir(Expanded_ifc_folder_path):
        if filename.lower().endswith('.ifc'):
            Regenerate_global_ids()
    
    # Export IFCs to .obj files 
    index = 1
    settings = ifcopenshell.geom.settings() 
    settings.set(settings.USE_WORLD_COORDS, True)
    IFC_classes = cfg.get("ifc_classes", [])
    for filename in os.listdir(Expanded_ifc_folder_path):
        if filename.lower().endswith('.ifc'):
            IFC_file_path = os.path.join(Expanded_ifc_folder_path, filename)
            IFC_file = ifcopenshell.open(IFC_file_path)
            IFC_to_obj(IFC_file, IFC_classes, index, settings) 
            Extract_graph(index, IFC_file)
            index += 1
    if cfg.get("data_expand"):
        object_randomise.main(cfg)


if __name__=='__main__':
    cfg_path = os.path.join(os.path.dirname(m01_Config_Files.__file__), 'extracting.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)
