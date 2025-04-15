import torch
import meshplot as mp
import skimage
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mp.offline()

def clamp(x, delta=torch.tensor([[0.1]]).to(device)):
    """Clamp function introduced in the paper DeepSDF.
    This returns a value in range [-delta, delta]. If x is within this range, it returns x, else one of the extremes.

    Args:
        x: prediction, torch tensor (batch_size, 1)
        delta: small value to control the distance from the surface over which we want to mantain metric SDF
    """
    maximum = torch.amax(torch.vstack((x, -delta)))
    minimum = torch.amin(torch.vstack((delta[0], maximum)))
    return minimum


#def SDFLoss_multishape(sdf, prediction, x_latent, sigma):
    #"""Loss function introduced in the paper DeepSDF for multiple shapes."""
    #l1 = torch.mean(torch.abs(prediction - sdf))
    #l2 = sigma**2 * torch.mean(torch.linalg.norm(x_latent, dim=1, ord=2))
    #loss = l1 + l2
    #print(f'Loss prediction: {l1:.3f}, Loss regulariser: {l2:.3f}')
    #return loss, l1, l2

def SDFLoss_multishape_full_exp(sdf, prediction, x_latent, sigma, alpha=0.5, w_max=1.2, w_min=0.05):
    """
    简单的全程指数衰减:
    weight = w_max * exp(-alpha * sdf)
    - sdf=0 => weight=w_max
    - sdf越来越大 => weight越来越接近0
    """
    dtype = sdf.dtype
    device = sdf.device

    weight = w_min + (w_max - w_min) * torch.exp(-alpha * torch.abs(sdf))

    l1 = torch.mean(weight * torch.abs(prediction - sdf))
    l2 = sigma**2 * torch.mean(torch.linalg.norm(x_latent, dim=1, ord=2))

    return l1 + l2, l1, l2


def generate_latent_codes(latent_size, samples_dict):
    """Generate a random latent codes for each shape form a Gaussian distribution
    Returns:
        - latent_codes: np.array, shape (num_shapes, latent_size)
        - dict_latent_codes: key: obj_index, value: corresponding idx in the latent_codes array. 
                                  e.g.  latent_codes = ([ [1, 2, 3], [7, 8, 9] ])
                                        dict_latent_codes[345] = 0, the obj that has index 345 refers to 
                                        the 0-th latent code.
    """
    latent_codes = torch.tensor([], dtype=torch.float32).reshape(0, latent_size).to(device)
    #dict_latent_codes = dict()
    for i, obj_idx in enumerate(list(samples_dict.keys())):
        #dict_latent_codes[obj_idx] = i
        latent_code = torch.normal(0, 0.01, size = (1, latent_size), dtype=torch.float32).to(device)
        latent_codes = torch.vstack((latent_codes, latent_code))
    latent_codes.requires_grad_(True)
    return latent_codes #, dict_latent_codes


def get_volume_coords(resolution = 128):
    """Get 3-dimensional vector (M, N, P) according to the desired resolutions."""
    x_min, x_max = -15, 15  # 梁的长度
    y_min, y_max = -15, 15   # 梁的高度
    z_min, z_max = -15, 15        # 梁的宽度
    x_values = torch.linspace(x_min, x_max, resolution, device=device)
    y_values = torch.linspace(y_min, y_max, resolution, device=device)
    z_values = torch.linspace(z_min, z_max, resolution, device=device)
    grid = torch.meshgrid(x_values, y_values, z_values, indexing='ij')
    grid_size_axis = (len(x_values), len(y_values), len(z_values))

    # Reshape grid to (M*N*P, 3)
    coords = torch.vstack((grid[0].ravel(), grid[1].ravel(), grid[2].ravel())).transpose(1, 0).to(device) 

    return coords, grid_size_axis


def save_meshplot(vertices, faces, path):
    mp.plot(vertices, faces, c=vertices[:, 2], filename=path)


def predict_sdf(latent, coords_batches, model):

    sdf = torch.tensor([], dtype=torch.float32).view(0, 1).to(device)

    model.eval()
    with torch.no_grad():
        for coords in coords_batches:
            latent_tile = torch.tile(latent, (coords.shape[0], 1))
            coords_latent = torch.hstack((latent_tile, coords))
            sdf_batch = model(coords_latent)
            sdf = torch.vstack((sdf, sdf_batch))        

    return sdf


def extract_mesh(grad_size_axis, sdf):
    # Extract zero-level set with marching cubes
    grid_sdf = sdf.view(grad_size_axis, grad_size_axis, grad_size_axis).detach().cpu().numpy()
    vertices, faces, normals, _ = skimage.measure.marching_cubes(grid_sdf, level=0.00)
    x_max = np.array([15, 15, 15])  # 目标最大范围
    x_min = np.array([-15, -15, -15])  # 目标最小范围
    vertices = vertices * ((x_max-x_min) / grad_size_axis) + x_min

    return vertices, faces