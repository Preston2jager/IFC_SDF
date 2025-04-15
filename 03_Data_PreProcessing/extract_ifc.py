import ifcopenshell
import ifcopenshell.api
import ifcopenshell.geom
import yaml, os
from utils_ifc import *

import config_files
import data.raw_ifc.projects
import data.raw_ifc.expanded_projects
import data.objects
import data.converted_data
import data.converted_data.graph
import utils.utils_obj as object_randomise
import utils.utils_ifc as ifc_split

def main(cfg):
    reference_folder_pth = os.path.dirname(data.converted_data.__file__)
    ifc_file_path_raw = os.path.dirname(data.raw_ifc.projects.__file__)
    ifc_file_path = os.path.dirname(data.raw_ifc.expanded_projects.__file__)
    graph_path = os.path.dirname(data.converted_data.graph.__file__)
    output_dir = os.path.dirname(data.objects.__file__)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #Delete previous objs
    for filename in os.listdir(output_dir):
        if filename.lower().endswith('.obj'):
            old_file_path = os.path.join(output_dir, filename)
            os.remove(old_file_path)
            print(f"Deleted: {old_file_path}")
    #Delete previous reference json
    for filename in os.listdir(reference_folder_pth):
        if filename.lower().endswith('.json'):
            old_file_path = os.path.join(reference_folder_pth, filename)
            os.remove(old_file_path)
            print(f"Deleted: {old_file_path}")     
    #Delete previous ifcs
    for filename in os.listdir(ifc_file_path):
        if filename.lower().endswith('.ifc'):
            old_file_path = os.path.join(ifc_file_path, filename)
            os.remove(old_file_path)
            print(f"Deleted: {old_file_path}")
    #Delete previous json     
    for filename in os.listdir(graph_path):
        if filename.lower().endswith('.json'):
            old_file_path = os.path.join(graph_path, filename)
            os.remove(old_file_path)
            print(f"Deleted: {old_file_path}")

    if bool(cfg.get("data_expand")) is True:
        for filename in os.listdir(ifc_file_path_raw):
            if filename.lower().endswith('.ifc'):
                ifc_file_expand(cfg)
    else:
        for filename in os.listdir(ifc_file_path_raw):
            if filename.lower().endswith('.ifc'):
                source_file_path = os.path.join(ifc_file_path_raw, filename)
                new_file_path = os.path.join(ifc_file_path, filename)
                shutil.copy(source_file_path, new_file_path)
    
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
    cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'extract_ifc.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)
