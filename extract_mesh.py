import os
import torch

import numpy as np

import subprocess

from scene import Scene
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import matplotlib.pyplot as plt
from utils.loss_utils import compute_scale_and_shift
from utils.mesh_utils import poisson_surface_reconstruction
from gaussian_renderer import generate_neural_gaussians, generate_neural_gaussians_SDF

# for marching cube
from skimage import measure
import plotly.graph_objects as go
import trimesh

# for tsdf fusion
import vdbfusion
from utils.graphics_utils import depth2point


from scipy.spatial import ckdtree

def get_grid_uniform(resolution, grid_boundary=[-2.0, 2.0]):
    x = np.linspace(grid_boundary[0], grid_boundary[1], resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points,
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}
    
@torch.no_grad()
def get_surface_trace(path, gs, resolution=100, grid_boundary=[-2.0, 2.0], return_mesh=False, level=0):
    grid = get_grid_uniform(resolution, grid_boundary)
    points = grid['grid_points']

    z = []
    for i, pnts in enumerate(torch.split(points, 100000, dim=0)):
        z.append(gs.get_sdf_value(pnts.cuda()).detach().cpu().numpy())
    z = np.concatenate(z, axis=0)

    if (not (np.min(z) > level or np.max(z) < level)):

        z = z.astype(np.float32)

        verts, faces, normals, values = measure.marching_cubes(
            volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
            level=level,
            spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1],
                     grid['xyz'][0][2] - grid['xyz'][0][1]))

        verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

        I, J, K = faces.transpose()

        traces = [go.Mesh3d(x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                            i=I, j=J, k=K, name='implicit_surface',
                            color='#ffffff', opacity=1.0, flatshading=False,
                            lighting=dict(diffuse=1, ambient=0, specular=0),
                            lightposition=dict(x=0, y=0, z=-1), showlegend=True)]

        meshexport = trimesh.Trimesh(verts, faces, normals)
        meshexport.export(path, 'ply')

        if return_mesh:
            return meshexport
        return traces
    return None

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, mesh_type="mcube"):
    if mesh_type == "poisson":        
        points, color, opaicity,scaling,rot, normal, _, _, _,_ = generate_neural_gaussians_SDF(views[0], gaussians, visible_mask=None)        
        
        points = points.cpu().detach().numpy()
        points_normals = torch.nn.functional.normalize(normal).cpu().detach().numpy()
        vertices, triangle, pcd = poisson_surface_reconstruction(points, points_normals, 9)
        # vertices, triangle, pcd = poisson_surface_reconstruction(points, None, 9)
        # use open3d to save the mesh
        import open3d as o3d
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangle)
        mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
        
        # pcd = pcd.voxel_down_sample(voxel_size=0.01)
        scale_matrix = np.diag([50, 50, 50])
        pcd.points = o3d.utility.Vector3dVector(np.matmul(scale_matrix, np.asarray(pcd.points).T).T)
        normals = np.asarray(pcd.normals)
        scaled_normals =normals * 0.1
        pcd.normals = o3d.utility.Vector3dVector(scaled_normals)
        # o3d.visualization.draw_geometries([pcd], point_show_normal=True)
        # mesh.compute_vertex_normals()
        o3d.io.write_triangle_mesh(os.path.join(model_path, "extracted_mesh_poisson_{}".format(iteration)+ ".ply"), mesh)        
    elif mesh_type == "mcube":
        _ = get_surface_trace(
            path = os.path.join(model_path, "extracted_mesh_marching_cube_{}".format(iteration)+".ply"),
            gs=gaussians,
            resolution=512,
            grid_boundary=[-1.0, 1.0],
            level=0.0
        )
    elif mesh_type == "tsdf":
        # use TSDF fusion with rendered mean to reconstruct surface
        # reference project:
        # https://github.com/GAP-LAB-CUHK-SZ/gaustudio
        # https://github.com/surfsplatting/surfsplatting.github.io/blob/main/assets/paper/paper.pdf
        vdb_volume = vdbfusion.VDBVolume(voxel_size=0.005, sdf_trunc=0.08, space_carving=True)
        for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
            torch.cuda.synchronize()
            voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
            render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
            torch.cuda.synchronize()
            
            render_depth = render_pkg["render_depth"].squeeze()
            
            
            render_pcd_cam, render_pcd_world = depth2point(render_depth, torch.inverse(view.inv_intrinsic).to(render_depth.device), view.world_view_transform.transpose(0, 1).to(render_depth.device))

            vdb_volume.integrate(render_pcd_world.view(-1, 3).double().cpu().numpy(), extrinsic=view.camera_center.double().cpu().numpy())
            
        # get mesh from vdb_volume
        vertices, faces = vdb_volume.extract_triangle_mesh(min_weight = 5)
        geo_mesh = trimesh.Trimesh(vertices, faces)
        geo_mesh.export(os.path.join(model_path, "extracted_mesh_tsdf.ply"), 'ply')
        
    else:
        raise NotImplementedError
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, mesh_type: str):
    with torch.no_grad():
        dataset.eval = True
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        # use training view to filter out the voxels    
        render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, mesh_type)

    
if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--mesh_type", default="mcube", type=str)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.mesh_type)
