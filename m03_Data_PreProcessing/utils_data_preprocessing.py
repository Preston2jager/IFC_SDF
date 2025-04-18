import ifcopenshell
import ifcopenshell.geom
import os
import re
import json
import shutil
import math
import tempfile
import random

import numpy as np
from collections import deque, defaultdict

import m01_Config_Files
import m02_Data_Files.d01_Raw_IFC
import m02_Data_Files.d01_Raw_IFC.d01_Expanded
import m02_Data_Files.d02_Object_Files
import m02_Data_Files.d05_Graph.json

def Delete_files(folder_path, file_type):
    """
    elete all files of a type in a folder.
    """
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(file_type):
            target_file_path = os.path.join(folder_path, filename)
            os.remove(target_file_path)
            print(f"File Deleted: {target_file_path}")

def IFC_file_expand(filename, copies):
    """
    Copy and rename IFC files.
    """
    Raw_IFC_dir = os.path.dirname(m02_Data_Files.d01_Raw_IFC.__file__)
    Raw_IFC_file_path = os.path.join(Raw_IFC_dir, filename)
    Expanded_IFC_dir = os.path.dirname(m02_Data_Files.d01_Raw_IFC.d01_Expanded.__file__)
    File_base, File_ext = os.path.splitext(filename)
    for i in range(1, copies + 1):
        Target_file_name = f"{File_base}_copy{i}{File_ext}"
        Target_file_path = os.path.join(Expanded_IFC_dir, Target_file_name)
        shutil.copy(Raw_IFC_file_path, Target_file_path)
        print(f"IFC Expansion: {Target_file_path} Created")

def Regenerate_global_ids(filename):
    """
    Generate new uids for all IFC elements.
    """
    IFC_dir = os.path.dirname(m02_Data_Files.d01_Raw_IFC.d01_Expanded.__file__)
    IFC_file_path = os.path.join(IFC_dir,filename)
    IFC_file = ifcopenshell.open(IFC_file_path)
    for entity in IFC_file.by_type("IfcRoot"):
        entity.GlobalId = ifcopenshell.guid.new()
    IFC_file.write(IFC_file_path) 
    print(f"New uid for {IFC_file_path} generated")

def Export_windows_and_doors(element, index, settings):
    """
    Replace windows and doors to a box and export as .obj.
    """
    Split=False
    Segments = None
    Vertices = []
    Triangles = []
    IFC_class = element.is_a()
    Vertex_offset = 0  
    try:
        shape = ifcopenshell.geom.create_shape(settings, element)
        verts = np.array(shape.geometry.verts).reshape(-1, 3)
        placement_matrix = np.array(shape.transformation.matrix).reshape(4, 4).T
        verts_homogeneous = np.hstack([verts, np.ones((verts.shape[0], 1))])
        local_verts = (np.linalg.inv(placement_matrix) @ verts_homogeneous.T).T[:, :3]
        min_local = local_verts.min(axis=0)
        max_local = local_verts.max(axis=0)
        bbox_local_vertices = np.array([
            [min_local[0], min_local[1], min_local[2]],
            [max_local[0], min_local[1], min_local[2]],
            [max_local[0], max_local[1], min_local[2]],
            [min_local[0], max_local[1], min_local[2]],
            [min_local[0], min_local[1], max_local[2]],
            [max_local[0], min_local[1], max_local[2]],
            [max_local[0], max_local[1], max_local[2]],
            [min_local[0], max_local[1], max_local[2]],
        ])
        bbox_local_vertices_h = np.hstack([bbox_local_vertices, np.ones((8, 1))])
        bbox_world_vertices = (placement_matrix @ bbox_local_vertices_h.T).T[:, :3]
        Vertices.extend(bbox_world_vertices)
        quad_faces = [
            [0, 1, 2, 3],  
            [4, 5, 6, 7],  
            [0, 1, 5, 4],  
            [1, 2, 6, 5],  
            [2, 3, 7, 6],  
            [3, 0, 4, 7],  
        ]
        # Split faces to triangles.
        for quad in quad_faces:
            idx0, idx1, idx2, idx3 = [Vertex_offset + i for i in quad]
            Triangles.append([idx0, idx1, idx2])
            Triangles.append([idx0, idx2, idx3])
        Vertex_offset += 8
    except Exception as e:
        print(f"Error with {element.GlobalId} info: {e}")
    return Vertices, Triangles, element.GlobalId, IFC_class, index, Split, Segments

def Export_general_elements(element, index, settings):
    """
    Export verts and faces for general elements, if an elements has separate parts, export separately.
    """
    Split = False
    IFC_class = element.is_a()
    Vertices, Triangles = Get_geometry(element, settings)
    face_graph = Build_face_graph(Triangles)
    Segments = Find_connected_components(Triangles, face_graph)
    if len(Segments) > 1:
        print(f"Element {element.GlobalId} has {len(Segments)} separate parts.")
        Split = True
    else:
        Segments = None
    return Vertices, Triangles, element.GlobalId, IFC_class, index, Split, Segments

def IFC_to_obj(IFC_file, IFC_classes, index, settings):
    """
    Export IFC elements to .obj files.
    """
    # Doors and windows are handled in different way.
    Special_classes = {"IfcWindow", "IfcDoor"}
    Window_and_door_elements = []
    General_elements = []

    # Extracting elements
    for class_name in IFC_classes:
        elements = IFC_file.by_type(class_name)
        if class_name in Special_classes:
            Window_and_door_elements += elements
        else:
            General_elements += elements

    # Process windows and doors
    for element in Window_and_door_elements:
        Vertices, Triangles, element.GlobalId, IFC_class, index, Split, Segments = Export_windows_and_doors(element, index, settings)
        Write_to_obj(Vertices, Triangles, element.GlobalId, IFC_class, index, Split, Segments)
    # Process other elements.
    for element in General_elements:
        Vertices, Triangles, element.GlobalId, IFC_class, index, Split, Segments = Export_general_elements(element, index, settings)
        Write_to_obj(Vertices, Triangles, element.GlobalId, IFC_class, index, Split, Segments)

def Get_geometry(element, settings):
    """
    Get verts and faces of general elements.
    """
    shape = ifcopenshell.geom.create_shape(settings, element)
    verts = np.array(shape.geometry.verts).reshape(-1, 3)  # 3D 坐标
    faces = np.array(shape.geometry.faces).reshape(-1, 3)  # 三角形索引
    return verts, faces

def Build_face_graph(faces):
    """
    Generate a graph of faces to determine separate elements.
    """
    edge_to_faces = defaultdict(set)
    face_graph = defaultdict(set)
    for i, face in enumerate(faces):
        edges = {
            (min(face[0], face[1]), max(face[0], face[1])),
            (min(face[1], face[2]), max(face[1], face[2])),
            (min(face[2], face[0]), max(face[2], face[0]))
        }
        for edge in edges:
            edge_to_faces[edge].add(i)
    for edge, face_set in edge_to_faces.items():
        face_list = list(face_set)
        for i in range(len(face_list)):
            for j in range(i + 1, len(face_list)):
                face_graph[face_list[i]].add(face_list[j])
                face_graph[face_list[j]].add(face_list[i])
    return face_graph

def Find_connected_components(faces, face_graph):
    """
    Identify separate geometry in an element.
    """
    visited = set()
    Segements = []
    def bfs(start_face):
        queue = deque([start_face])
        component = set()
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            component.add(current)
            queue.extend(face_graph[current] - visited)
        return component
    for i in range(len(faces)):
        if i not in visited:
            group = bfs(i)
            Segements.append(group)
    return Segements

def Write_to_obj(vertices, faces, uid, ifc_class, index, split=False, groups=None):
    """
    Export IFC elements as obj, also handle elements with multiple parts.
    """
    Objs_dir = os.path.dirname(m02_Data_Files.d02_Object_Files.__file__)
    if split:
        for split_num, group in enumerate(groups, start=1):
            used_vertex_indices = set()
            for face_index in group:
                used_vertex_indices.update(faces[face_index])
            used_vertex_indices = sorted(used_vertex_indices)
            index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertex_indices, start=1)}
            file_name = f"{index}_{ifc_class}_{uid}_split_{split_num}"
            output_file = os.path.join(Objs_dir, f"{file_name}.obj")
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# Split OBJ File\n")
                for old_idx in used_vertex_indices:
                    vertex = vertices[old_idx]
                    vertex_str = "v " + " ".join(map(str, vertex)) + "\n"
                    f.write(vertex_str)
                f.write(f"\no Object_{split_num}\n")
                for face_index in group:
                    remapped = [index_mapping[v] for v in faces[face_index]]
                    f.write("f " + " ".join(str(idx) for idx in remapped) + "\n")
            print(f"Element {uid}: Split {split_num} saved to {output_file}")
    else:
        file_name = f"{index}_{ifc_class}_{uid}.obj"
        output_file = os.path.join(Objs_dir, file_name)
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# OBJ File\n")
            for vertex in vertices:
                vertex_str = "v " + " ".join(map(str, vertex)) + "\n"
                f.write(vertex_str)
            f.write("\no Object\n")
            for face in faces:
                f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")
        print(f"Element {uid}: Saved to {output_file}")

def Extract_graph(project_index, ifc, output_prefix='ifc_graph'):
    """
    Generate ifc graphs for reference. To 
    """
    target_types = ["IfcWall", "IfcWindow", "IfcDoor", "IfcSlab"]
    elements = []
    id_to_index = {}
    index_to_info = {}
    # Extract elements of target types and record their information.
    for ifc_type in target_types:
        objs = ifc.by_type(ifc_type)
        for obj in objs:
            index = len(elements)
            elements.append(obj)
            global_id = obj.GlobalId
            id_to_index[global_id] = index
            # Use the element index as key so each element's info is stored separately.
            index_to_info[index] = {
                "index": index,
                "GlobalId": global_id,
                "type": ifc_type
            }
    # Build the graph structure with nodes and empty edges list.
    graph = {
        "project": project_index,
        "nodes": list(index_to_info.values()),
        "edges": []  
    }   
    main_wall = []
    all_wall = []
    globalid_to_index = {v["GlobalId"]: k for k, v in index_to_info.items()}
    slab_index = next(idx for idx, info in index_to_info.items() if info["type"] == "IfcSlab")
    # Process IfcWall elements and build edges based on relationships
    for wall in ifc.by_type("IfcWall"):
        wall_index = globalid_to_index[wall.GlobalId]
        graph["edges"].append((slab_index, wall_index))
        all_wall.append(wall_index)
        if hasattr(wall, "HasOpenings"):
            for rel in wall.HasOpenings:
                opening = rel.RelatedOpeningElement
                for fill in getattr(opening, "HasFillings", []):
                    filled = fill.RelatedBuildingElement
                    filled_index = globalid_to_index[filled.GlobalId]
                    graph["edges"].append((wall_index, filled_index))
                    main_wall.append(wall_index)
    side_wall = list(set(all_wall) - set(main_wall))
    for side_index in side_wall:
        for main_index in main_wall:
            graph["edges"].append((side_index, main_index))
    Output_dir = os.path.dirname(m02_Data_Files.d05_Graph.json.__file__)
    Output_file_name = os.path.join(Output_dir, f"{project_index}:{output_prefix}.json")
    with open(Output_file_name, "w") as f:
        json.dump(graph, f, indent=2)

def Generate_random_transform(cfg):
    """
    Generate random rotation and translation coordinates for an OBJ file.
    """
    angle = random.uniform(cfg["rotation"]["min"], cfg["rotation"]["max"])
    tx = random.uniform(cfg["translation"]["x_range"][0], cfg["translation"]["x_range"][1])
    ty = random.uniform(cfg["translation"]["y_range"][0], cfg["translation"]["y_range"][1])
    tz = 0
    return {
        "rotation_angle": angle,
        "translation": (tx, ty, tz),
        "axis": cfg.get("axis", "y")
    }

def Rotate_vertex(vertex, angle_degrees, axis='y'):
    """
    Get rotation coords for vertices.
    """
    angle_radians = math.radians(angle_degrees)
    x, y, z = vertex
    if axis == 'y':
        x_new = x * math.cos(angle_radians) + z * math.sin(angle_radians)
        y_new = y
        z_new = -x * math.sin(angle_radians) + z * math.cos(angle_radians)
    elif axis == 'x':
        x_new = x
        y_new = y * math.cos(angle_radians) - z * math.sin(angle_radians)
        z_new = y * math.sin(angle_radians) + z * math.cos(angle_radians)
    elif axis == 'z':
        x_new = x * math.cos(angle_radians) - y * math.sin(angle_radians)
        y_new = x * math.sin(angle_radians) + y * math.cos(angle_radians)
        z_new = z
    else:
        x_new, y_new, z_new = x, y, z
    return [x_new, y_new, z_new]

def Single_obj_transform(file_path, rotation_angle=0, translation=(0, 0, 0), axis='y'):
    """
    Rotate and move a single .obj file.
    """
    vertices = []
    faces = []
    others = []  
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                vertex = list(map(float, parts[1:4]))
                vertex = Rotate_vertex(vertex, rotation_angle, axis)
                vertex = [vertex[i] + translation[i] for i in range(3)]
                vertices.append(vertex)
            elif line.startswith('f '):
                faces.append(line.strip())
            else:
                others.append(line.rstrip())
    with tempfile.NamedTemporaryFile('w', delete=False) as tmp:
        for v in vertices:
            tmp.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for line in others:
            tmp.write(f"{line}\n")
        for face in faces:
            tmp.write(f"{face}\n")
        temp_path = tmp.name
    shutil.move(temp_path, file_path)

def Batch_obj_transform(directory, cfg):
    """
    Randomly rotate and move all expanded .obj files.
    Use prefix to identify file from same project model.
    """
    prefix_transform_map = {}
    for filename in os.listdir(directory):
        match = re.match(r'^(\d+)_.*\.obj$', filename, re.IGNORECASE)
        if match:
            prefix = match.group(1)
            if prefix not in prefix_transform_map:
                prefix_transform_map[prefix] = Generate_random_transform(cfg)
            params = prefix_transform_map[prefix]
            file_path = os.path.join(directory, filename)
            Single_obj_transform(
                file_path,
                rotation_angle=params["rotation_angle"],
                translation=params["translation"],
                axis=params["axis"]
            )
            print(f"{filename} Transformed")
        else:
            print(f"[!] Skipped {filename}")
