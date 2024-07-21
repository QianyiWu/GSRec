#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import os
import torch

import numpy as np

import subprocess
# cmd = 'nvidia-smi -q -d Memory |grep -A4 GPU|grep Used'
# result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE).stdout.decode().split('\n')
# os.environ['CUDA_VISIBLE_DEVICES']=str(np.argmin([int(x.split()[2]) for x in result[:-1]]))

# os.system('echo $CUDA_VISIBLE_DEVICES')

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

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_renders")
    
    if not os.path.exists(depth_render_path):
        os.makedirs(depth_render_path, exist_ok=True)

    name_list = []
    per_view_dict = {}
    # debug = 0
    t_list = []
    
    render_pkg = render(views[0], gaussians, pipeline, background, visible_mask=None)
    points = render_pkg["points"].cpu().detach().numpy()
    points_normals = torch.nn.functional.normalize(render_pkg["normal"], dim=-1).cpu().detach().numpy()
    vertices, triangle, pcd = poisson_surface_reconstruction(points, points_normals, 9)
    # use open3d to save the mesh
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangle)
    mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
    
    # save the mls information
    torch.save(
        {'points': render_pkg["points"].cpu().detach(),
         'normals': render_pkg["normal"].cpu().detach(),
         'covs': render_pkg["cov3D"].cpu().detach(),
         'opacity': render_pkg["opacity"].cpu().detach()},
        os.path.join(model_path, name, "ours_{}_mls_info.pt".format(iteration))
    )
    
    
    # pcd = pcd.voxel_down_sample(voxel_size=0.01)
    scale_matrix = np.diag([50, 50, 50])
    pcd.points = o3d.utility.Vector3dVector(np.matmul(scale_matrix, np.asarray(pcd.points).T).T)
    normals = np.asarray(pcd.normals)
    scaled_normals = normals * 0.1
    pcd.normals = o3d.utility.Vector3dVector(scaled_normals)
    o3d.visualization.draw_geometries([pcd], point_show_normal=True)
    # mesh.compute_vertex_normals()
    o3d.io.write_triangle_mesh(os.path.join(model_path, name, "ours_{}".format(iteration), '{0:05d}'.format(0) + ".ply"), mesh)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):

     
        torch.cuda.synchronize(); t0 = time.time()
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize(); t1 = time.time()
        
        t_list.append(t1-t0)
        
        rendering = render_pkg["render"]
        gt = view.original_image[0:3, :, :]
        
        # render depth
        render_depth = render_pkg["render_depth"]
        gt_depth = view.depth
        scale, shift = compute_scale_and_shift(render_depth, gt_depth)
        depth = render_depth * scale + shift
        
        depth_concat = torch.cat((depth, gt_depth), dim=0).unsqueeze(1)
        tensor = torchvision.utils.make_grid(depth_concat, padding=0, normalize=False, scale_each=False).cpu().detach().numpy()
        plt.imsave(os.path.join(depth_render_path, '{0:05d}'.format(idx) + "_depth.png"), np.transpose(tensor, (1,2,0))[:,:,0], cmap="viridis")
                
        # add normal rendering
        render_normal = render_pkg["render_normal"]
        gt_normal = view.normal
        normal_concat = torch.stack((render_normal, gt_normal), dim=0)
        normal_concat = (normal_concat + 1)/2.0
        tensor = torchvision.utils.make_grid(normal_concat, padding=0, normalize=False, scale_each=False).cpu().detach().numpy()
        plt.imsave(os.path.join(depth_render_path, '{0:05d}'.format(idx) + "_normal.png"), (tensor.transpose((1,2,0))*255).astype(np.uint8))
              
        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
            
   
    
    
    t = np.array(t_list[5:])
    fps = 1.0 / t.mean()
    print(f'Test FPS: \033[1;35m{fps:.5f}\033[0m')

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)      
     
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        dataset.eval = True
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        
        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_false")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)
