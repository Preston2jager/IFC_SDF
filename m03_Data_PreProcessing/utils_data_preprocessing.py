import ifcopenshell, os, json, shutil
import ifcopenshell.geom
import numpy as np
from collections import deque, defaultdict

import data.objects
import data.raw_ifc
import data.raw_ifc.projects
import data.raw_ifc.expanded_projects
import data.converted_data
import data.converted_data.graph

def get_geometry(beam, settings):
    shape = ifcopenshell.geom.create_shape(settings, beam)
    verts = np.array(shape.geometry.verts).reshape(-1, 3)  # 3D 坐标
    faces = np.array(shape.geometry.faces).reshape(-1, 3)  # 三角形索引
    return verts, faces

def build_face_graph(faces):
    edge_to_faces = defaultdict(set)
    face_graph = defaultdict(set)
    for i, face in enumerate(faces):
        # 使用无序对表示边，确保 (a,b) 和 (b,a) 视为相同边
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

def find_connected_components(faces, face_graph):
    """查找所有独立的连通组件"""
    visited = set()
    groups = []
    def bfs(start_face):
        """使用 BFS 进行连通性检测"""
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
            groups.append(group)
    return groups

def write_obj_file(vertices, faces, uid, ifc_class, index, groups=None, split=False):
    base_dir = os.path.dirname(data.objects.__file__)
    index_file = os.path.join(os.path.dirname(data.converted_data.__file__),"project_reference.json")
    if split:
        for split_num, group in enumerate(groups, start=1):
        # 收集当前组中使用到的所有顶点索引（原始索引）
            used_vertex_indices = set()
            for face_index in group:
                used_vertex_indices.update(faces[face_index])
            # 将顶点索引按顺序排列
            used_vertex_indices = sorted(used_vertex_indices)
            # 构建旧索引到新索引的映射，新索引从1开始
            index_mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(used_vertex_indices, start=1)}
            # 生成唯一的文件名，包含 beam 的 GlobalId 和分段编号
            file_name = f"{index}_{ifc_class}_{uid}_split_{split_num}"
            output_file = os.path.join(base_dir, f"{file_name}.obj")
            
            info=[]
            info.append({
                "project": index,
                "uid": uid,
                "ifcclass": ifc_class,
            })
            with open(index_file, "a") as f:
                json.dump(info, f, indent=4)

            with open(output_file, 'w', encoding='utf-8') as f:
                f.write("# Split OBJ File\n")
                # 写入仅当前组件使用到的顶点，格式化为 OBJ 文件中的顶点数据（v x y z）
                for old_idx in used_vertex_indices:
                    vertex = vertices[old_idx]
                    vertex_str = "v " + " ".join(map(str, vertex)) + "\n"
                    f.write(vertex_str)
                # 写入对象名
                f.write(f"\no Object_{split_num}\n")
                # 写入面数据，更新面中的顶点索引为新的索引
                for face_index in group:
                    remapped = [index_mapping[v] for v in faces[face_index]]
                    f.write("f " + " ".join(str(idx) for idx in remapped) + "\n")
            print(f"Element {uid}: Split {split_num} saved to {output_file}")
    else:
        file_name = f"{index}_{ifc_class}_{uid}.obj"
        output_file = os.path.join(base_dir, file_name)

        info=[]
        info.append({
                "project": index,
                "uid": uid,
                "ifcclass": ifc_class,
            })
        with open(index_file, "a") as f:
                json.dump(info, f, indent=4)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("# OBJ File\n")
            # 写入所有顶点
            for vertex in vertices:
                vertex_str = "v " + " ".join(map(str, vertex)) + "\n"
                f.write(vertex_str)
            # 写入对象名
            f.write("\no Object\n")
            # 写入所有面数据
            for face in faces:
                f.write("f " + " ".join(str(idx + 1) for idx in face) + "\n")
        print(f"Element {uid}: Saved to {output_file}")

def ifc_export_to_obj(ifc_file, settings, cfg, index):
    #Init 
    ifc_classes = cfg.get("ifc_classes", [])
    special_classes = {"IfcWindow", "IfcDoor"}
    window_and_door_elements = []
    general_elements = []
    #Add elements
    if "IfcWindow" in ifc_classes:
        window_and_door_elements += ifc_file.by_type("IfcWindow")
    if "IfcDoor" in ifc_classes:
        window_and_door_elements += ifc_file.by_type("IfcDoor")
    for class_name in ifc_classes:
        if class_name not in special_classes:
            general_elements += ifc_file.by_type(class_name)
    #Process windows and doors
    for element in window_and_door_elements:
        vertices = []
        triangles = []
        vertex_offset = 0  
        ifc_class = element.is_a()
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
            # 从局部坐标转换到世界坐标
            bbox_local_vertices_h = np.hstack([bbox_local_vertices, np.ones((8, 1))])
            bbox_world_vertices = (placement_matrix @ bbox_local_vertices_h.T).T[:, :3]
            vertices.extend(bbox_world_vertices)
            quad_faces = [
                [0, 1, 2, 3],  # 底面
                [4, 5, 6, 7],  # 顶面
                [0, 1, 5, 4],  # 前面
                [1, 2, 6, 5],  # 右面
                [2, 3, 7, 6],  # 后面
                [3, 0, 4, 7],  # 左面
            ]
            # 将每个四边形面拆分为两个三角面
            for quad in quad_faces:
                idx0, idx1, idx2, idx3 = [vertex_offset + i for i in quad]
                # 三角形1: (idx0, idx1, idx2)
                triangles.append([idx0, idx1, idx2])
                # 三角形2: (idx0, idx2, idx3)
                triangles.append([idx0, idx2, idx3])
            vertex_offset += 8
        except Exception as e:
            print(f"处理元素 {element.GlobalId} 出错：{e}")
            continue
        write_obj_file(vertices, triangles, element.GlobalId, ifc_class, index, groups=None, split=False)

    for element in general_elements:
        ifc_class = element.is_a()
        verts, faces = get_geometry(element, settings)
        face_graph = build_face_graph(faces)
        segments = find_connected_components(faces, face_graph)
        if len(segments) > 1:
            print(f"Element {element.GlobalId} has {len(segments)} separate parts.")
            # 将整个 beam 的 faces 与所有连通组件一起传入，生成多个 OBJ 文件
            write_obj_file(verts, faces, element.GlobalId, index, segments, split=True)
        else:
            #print(f"Element {element.GlobalId} is a single component; no split needed.")
            write_obj_file(verts, faces, element.GlobalId, ifc_class,index, groups=None, split=False)

def regenerate_global_ids():
    files_dir = os.path.dirname(data.raw_ifc.expanded_projects.__file__)
    for file_name in os.listdir(files_dir):
        if file_name.lower().endswith('.ifc'):
            file_path = os.path.join(os.path.dirname(data.raw_ifc.expanded_projects.__file__),file_name)
            ifc_model = ifcopenshell.open(file_path)
            for entity in ifc_model.by_type("IfcRoot"):
                entity.GlobalId = ifcopenshell.guid.new()
            ifc_model.write(file_path) 
    print("New uid generated")

def ifc_file_expand(cfg):
    copies = cfg.get("copies")+1
    source_dir = os.path.dirname(data.raw_ifc.projects.__file__)
    target_dir = os.path.dirname(data.raw_ifc.expanded_projects.__file__)
    if os.path.exists(target_dir):
        for filename in os.listdir(target_dir):
            if filename == "__init__.py":
                continue
            file_path = os.path.join(target_dir, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
    os.makedirs(target_dir, exist_ok=True)
    for file_name in os.listdir(source_dir):
        if file_name.lower().endswith('.ifc'):
            file_base, file_ext = os.path.splitext(file_name)
            source_file_path = os.path.join(source_dir, file_name)
            for i in range(1, copies + 1):
                new_file_name = f"{file_base}_copy{i}{file_ext}"
                new_file_path = os.path.join(target_dir, new_file_name)
                shutil.copy(source_file_path, new_file_path)
                print(f"{new_file_path} Created")


def extract_ifc_graph_json(project_index, ifc_file, output_prefix='ifc_graph'):
    ifc = ifcopenshell.open(ifc_file)
    target_types = ["IfcWall", "IfcWindow", "IfcDoor", "IfcSlab"]
    elements = []
    id_to_index = {}
    index_to_info = {}
    
    # Extract elements of target types and record their information
    for ifc_type in target_types:
        objs = ifc.by_type(ifc_type)
        for obj in objs:
            index = len(elements)
            elements.append(obj)
            global_id = obj.GlobalId
            id_to_index[global_id] = index
            # Use the element index as key so each element's info is stored separately
            index_to_info[index] = {
                "Project": project_index,
                "index": index,
                "GlobalId": global_id,
                "type": ifc_type
            }
            print(f"Relations for {index}:{global_id} ({ifc_type}):")

    # Build the graph structure with nodes and empty edges list
    graph = {
        "project": project_index,
        "nodes": list(index_to_info.values()),
        "edges": []  # Placeholder: populate with actual relationships as needed.
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
        # 检查墙体上是否有开口（门窗）
        if hasattr(wall, "HasOpenings"):
            for rel in wall.HasOpenings:
                opening = rel.RelatedOpeningElement
                for fill in getattr(opening, "HasFillings", []):
                    filled = fill.RelatedBuildingElement
                    filled_index = globalid_to_index[filled.GlobalId]
                    # 墙与门/窗建立连接
                    graph["edges"].append((wall_index, filled_index))
                    main_wall.append(wall_index)
    
    side_wall = list(set(all_wall) - set(main_wall))

    for side_index in side_wall:
        for main_index in main_wall:
            graph["edges"].append((side_index, main_index))
        
    # Save output files in the current working directory
    out_base_path = os.path.dirname(data.converted_data.graph.__file__)
    file_name = os.path.join(out_base_path, f"{project_index}:{output_prefix}.json")

    with open(file_name, "w") as f:
        #json.dump(list(graph["edges"]), f, indent=2)
        json.dump(graph, f, indent=2)

def main():
    path = os.path.join(os.path.dirname(data.raw_ifc.projects.__file__),"1.ifc")
    extract_ifc_graph_json(1, path)

if __name__=='__main__':
    main()

