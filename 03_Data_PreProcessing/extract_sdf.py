
import os, glob, yaml, trimesh
import numpy as np
from utils import utils_mesh
import point_cloud_utils as pcu
from glob import glob
from tqdm import tqdm

import config_files
import data.objects
import data.converted_data
"""
For each object, sample points and store their distance to the nearest triangle.
Sampling follows the approach used in the DeepSDF paper.
"""

def combine_sample_latent(samples, latent_class):
    """Combine each sample (x, y, z) with the latent code generated for this object.
    Args:
        samples: collected points, np.array of shape (N, 3)
        latent: randomly generated latent code, np.array of shape (1, args.latent_size)
    Returns:
        combined hstacked latent code and samples, np.array of shape (N, args.latent_size + 3)
    """
    latent_class_full = np.tile(latent_class, (samples.shape[0], 1))   
    # repeat the latent code N times for stacking
    return np.hstack((latent_class_full, samples))

def sample_on_sphere_surface(center, radius, num_points):
    directions = np.random.normal(size=(num_points, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)  # 单位向量
    # 将方向向量缩放到目标半径，再平移到中心
    points = center + radius * directions
    return points

def compute_triangle_areas(vertices, faces):
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross_prod = np.cross(v1 - v0, v2 - v0)
    return 0.5 * np.linalg.norm(cross_prod, axis=1)

def sample_points_and_compute_sdf(verts, faces, total_area, volume,cfg):

    surface_sample_num = int(float(total_area)*int(cfg['dense_of_samples_on_surface']))
    volume_sample_num = int(float(volume)*int(cfg['dense_of_samples_in_space']))
    far_field_sample_num = int(volume_sample_num * cfg['far_field_coefficient'])

    fid_surf, bc_surf = pcu.sample_mesh_random(verts, faces, surface_sample_num)
    p_surf = pcu.interpolate_barycentric_coords(faces, fid_surf, bc_surf, verts)

    triangles = faces[fid_surf]
    v0 = verts[triangles[:, 0]]
    v1 = verts[triangles[:, 1]]
    v2 = verts[triangles[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    face_normals /= np.linalg.norm(face_normals, axis=1, keepdims=True)

    offset_distance_1 = cfg['surface_offset_1']
    p_surf_out_1 = p_surf + offset_distance_1 * face_normals
    offset_distance_2 = cfg['surface_offset_2']
    p_surf_out_2 = p_surf +  offset_distance_2 * face_normals
    offset_distance_3 = offset_distance_1/2
    p_surf_in = p_surf - offset_distance_3 * face_normals

    centroid = np.mean(verts, axis=0)
    centered_verts = verts - centroid
    cov = np.cov(centered_verts, rowvar=False)
    eig_vals, eig_vecs = np.linalg.eigh(cov)
    local_verts = centered_verts @ eig_vecs
    local_min = np.min(local_verts, axis=0)
    local_max = np.max(local_verts, axis=0)
    local_center = (local_min + local_max) / 2.0
    half_range = (local_max - local_min) / 2.0
    new_half_range = half_range + 0.5
    new_local_min = local_center - new_half_range
    new_local_max = local_center + new_half_range
    p_vol_local = np.random.uniform(low=new_local_min, high=new_local_max, size=(volume_sample_num, 3))

    # 添加远场采样点（Far-field points）
    diag = np.linalg.norm(local_max - local_min)
    far_field_half_range_1 = diag + 5   # 扩大远场范围
    far_field_half_range_2 = diag + 10   # 扩大远场范围
    far_field_half_range_3 = diag + 20   # 扩大远场范围
    p_far_local_1 = sample_on_sphere_surface(local_center, far_field_half_range_1, far_field_sample_num)
    p_far_local_2 = sample_on_sphere_surface(local_center, far_field_half_range_2, far_field_sample_num)
    p_far_local_3 = sample_on_sphere_surface(local_center, far_field_half_range_3, far_field_sample_num)
    
    # 转换回世界坐标
    p_vol = p_vol_local @ eig_vecs.T + centroid
    p_far_1 = p_far_local_1 @ eig_vecs.T + centroid
    p_far_2 = p_far_local_2 @ eig_vecs.T + centroid
    p_far_3 = p_far_local_3 @ eig_vecs.T + centroid

    p_total = np.vstack((p_vol, p_surf_out_1, p_surf_out_2, p_surf_in, p_surf,p_far_1, p_far_2, p_far_3))
    sdf, _, _ = pcu.signed_distance_to_mesh(p_total, verts, faces)

    return p_total, sdf

def main(cfg):
    # Full paths to all .obj
    obj_paths = glob(os.path.join(os.path.dirname(data.objects.__file__), '*.obj'))
    # File to store the samples and SDFs
    samples_dict = dict()    
    # Store conversion between object index (int) and its folder name (str)
    idx_str2int_dict = dict()
    idx_int2str_dict = dict()

    for obj_idx, obj_path in enumerate(tqdm(obj_paths, desc="Processing OBJ files")):
        # Object unique index. Str to int by byte encoding
        obj_idx_str = os.path.splitext(os.path.basename(obj_path))[0]  # e.g., '1' 
        idx_str2int_dict[obj_idx_str] = obj_idx
        idx_int2str_dict[obj_idx] = obj_idx_str
        # Dictionary to store the samples and SDFs
        samples_dict[obj_idx] = dict()
        try:
            mesh_original = trimesh.load(obj_path, force='mesh')
            if not mesh_original.is_watertight:
                print(f"Mesh {obj_path} is not watertight, attempting to repair...")
                mesh_original.fill_holes()
                if not mesh_original.is_watertight:
                    print(f"Warning: Mesh {obj_path} could not be fully repaired.")
                else:
                    print(f"Mesh {obj_path} repaired successfully.")
            verts = np.array(mesh_original.vertices)
            faces = np.array(mesh_original.faces)
            total_area = compute_triangle_areas(verts, faces).sum()
            volume = mesh_original.volume
        except Exception as e:
            print(f"Error processing mesh {obj_path}: {e}")

        p_total, sdf = sample_points_and_compute_sdf(verts, faces, total_area, volume, cfg)

        samples_dict[obj_idx]['sdf'] = sdf
  
        # The samples are p_total, while the latent class is [obj_idx]
        samples_dict[obj_idx]['samples_latent_class'] = combine_sample_latent(p_total, np.array([obj_idx], dtype=np.int32))

    np.save(os.path.join(os.path.dirname(data.converted_data.__file__), f'samples_dict.npy'), samples_dict)
    np.save(os.path.join(os.path.dirname(data.converted_data.__file__), f'idx_str2int_dict.npy'), idx_str2int_dict)
    np.save(os.path.join(os.path.dirname(data.converted_data.__file__), f'idx_int2str_dict.npy'), idx_int2str_dict)
    print("Training data converted.")

if __name__=='__main__':
    cfg_path = os.path.join(os.path.dirname(config_files.__file__), 'extract_sdf.yaml')
    with open(cfg_path, 'rb') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)
    main(cfg)