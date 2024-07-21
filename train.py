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
import numpy as np

import torch
import torchvision
import torch.nn.functional as F
import json
import wandb
import time
from os import makedirs
import shutil, pathlib
from pathlib import Path
from PIL import Image
import torchvision.transforms.functional as tf
# from lpipsPyTorch import lpips
import lpips
from random import randint
from utils.loss_utils import l1_loss, ssim, ScaleAndShiftLoss, normal_loss, gradient_loss
from gaussian_renderer import prefilter_voxel, render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import matplotlib.pyplot as plt
from utils.loss_utils import compute_scale_and_shift
from utils.mesh_utils import poisson_surface_reconstruction
from extract_mesh import get_surface_trace

from scipy.spatial import ckdtree
from utils.general_utils import mls_sdf
from utils.graphics_utils import sample_pixels, unproject_3d

# torch.set_num_threads(32)
lpips_fn = lpips.LPIPS(net='vgg').to('cuda')

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = pathlib.Path(__file__).parent.resolve()


    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))
    back_dir = os.path.join(dst, 'code_backup')
    if not os.path.exists(back_dir):
        os.makedirs(back_dir, exist_ok=True)
    for item in os.listdir(dst):
        item_path = os.path.join(dst, item)
        if item_path != back_dir:
            shutil.move(item_path, back_dir)
    
    print('Backup Finished!')

def expand_indices(index_tensor, num_indices):
    # Create a tensor [0, 1, 2, 3, 4]
    sequence = torch.arange(num_indices, device=index_tensor.device)

    # Multiply index_tensor by 5 and reshape to (-1, 1) for broadcasting
    expanded = num_indices * index_tensor.unsqueeze(-1)

    # Add the sequence to each element and reshape
    result = (expanded + sequence).view(-1)

    return result

def training(dataset, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None, vis=False, k_near=50, sampling_numbers=8192, learn_sdf=False):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.implicit_sdf_divide_factor, dataset.sdf_inside_out, dataset.use_feat_bank)
    scene = Scene(dataset, gaussians, ply_path=ply_path)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):        
        # if network_gui.conn == None:
        #     network_gui.try_connect()
        # while network_gui.conn != None:
        #     try:
        #         net_image_bytes = None
        #         custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
        #         if custom_cam != None:
        #             net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
        #             net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
        #         network_gui.send(net_image_bytes, dataset.source_path)
        #         if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
        #             break
        #     except Exception as e:
        #         network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        
        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        
        voxel_visible_mask = prefilter_voxel(viewpoint_cam, gaussians, pipe,background)
        retain_grad = (iteration < opt.update_until and iteration >= 0)
        render_pkg = render(viewpoint_cam, gaussians, pipe, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad)
        
        image, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity =\
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"]

        render_depth, render_normal, render_median_depth = render_pkg["render_depth"], render_pkg["render_normal"], render_pkg["render_median_depth"]
        
        
        # monocular depth loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image*viewpoint_cam.mask.cuda(), gt_image) if opt.use_mask_for_rgb else l1_loss(image, gt_image)
        gt_depth = viewpoint_cam.depth.cuda() 
        depth_loss = ScaleAndShiftLoss(render_depth, gt_depth*50+0.5, mask=viewpoint_cam.mask.cuda())
        
        
        # monocular normal loss
        gt_normal = viewpoint_cam.normal.cuda()
        if opt.use_mask_for_normal:
            normal_l1_loss, normal_cos_loss, normal_grad_loss = normal_loss(render_normal, gt_normal, mask=viewpoint_cam.mask.cuda())
        else:
            normal_l1_loss, normal_cos_loss, normal_grad_loss = normal_loss(render_normal, gt_normal)
        
        # ssim loss        
        ssim_loss = (1.0 - ssim(image*viewpoint_cam.mask.cuda(), gt_image)) if opt.use_mask_for_rgb else (1.0 - ssim(image, gt_image))

        
        # scale regularization loss
        scaling_reg = scaling.min(dim=1)[0].mean() + ((scaling.topk(2)[0]**2).sum(1)/scaling.topk(2)[0].prod(1) -2).mean()
        
        if iteration > opt.sdf_start_iter:
            kdtree = ckdtree.cKDTree(render_pkg['points'].cpu().detach().numpy())
            random_coordinates = sample_pixels(viewpoint_cam.image_width, viewpoint_cam.image_height, sampling_numbers).cuda()

            # use median depth for sampling
            sample_depths = render_median_depth[0, random_coordinates[:, 0], random_coordinates[:, 1]].detach()
            
            unproject_points = unproject_3d(random_coordinates, viewpoint_cam.inv_intrinsic, sample_depths, viewpoint_cam.world_view_transform)
            
            # fine the nearest points around the unprojected points
            unproject_points_near_index = torch.from_numpy(kdtree.query(unproject_points.cpu().detach().numpy(), k=1)[1]).view(-1)# M

            sampling_points = render_pkg['points'] + torch.randn(render_pkg['points'].shape).cuda()*render_pkg['scaling']
                    
            # sampling points use depth and near-depth 
            near_depth_points = sampling_points[unproject_points_near_index]
            sampling_points = torch.cat([unproject_points, near_depth_points], dim=0)
            
            near_points_index = torch.from_numpy(kdtree.query(sampling_points.cpu().detach().numpy(), k=k_near)[1]).cuda().view(-1)# M
        
            if vis:
                import open3d as o3d
                scene_pcd = o3d.geometry.PointCloud()
                points_con = torch.cat([unproject_points, render_pkg['points']], dim=0)
                points_color = torch.cat([0.5*torch.ones_like(unproject_points), torch.zeros_like(render_pkg['points'])], dim=0)
                
                scene_pcd.points = o3d.utility.Vector3dVector(points_con.cpu().detach().numpy())
                scene_pcd.colors = o3d.utility.Vector3dVector(points_color.cpu().detach().numpy())
                o3d.visualization.draw_geometries([scene_pcd])
                    
            # data for MLS estimation
            selected_points = render_pkg['points'][near_points_index].view(-1, k_near, 3)
            distance = sampling_points.unsqueeze(1) - selected_points
            selected_cov3D = render_pkg['cov3D'][near_points_index].view(-1, k_near, 6)
            selected_normal = render_pkg['normal'][near_points_index].view(-1, k_near, 3)
            selected_opacity = render_pkg['opacity'][near_points_index].view(-1, k_near)
            
            
            points_num = render_pkg['points'].shape[0]
            sampling_points_num = sampling_points.shape[0]
            
            eiknoal_points = torch.empty(1024, 3).uniform_(-1, 1).cuda()
            normal_q = gaussians.get_gradient_value(torch.cat([render_pkg['points'], sampling_points, eiknoal_points], dim=0))
            
            normalized_normal_q = F.normalize(normal_q, dim=-1)
            
            
            weight, f_mls = mls_sdf(distance, selected_cov3D, selected_normal, 
                                    normalized_normal_q[points_num:points_num+sampling_points_num, :].unsqueeze(1), opacity=selected_opacity, weight_normal=opt.fmls_use_normal, offset_weight=opt.fmls_sdf_offset, normal_weight=opt.fmls_normal_weight)
                            
            f_mlp = gaussians.get_sdf_value(sampling_points)
            if (weight.sum(1)>1e-6).sum() == 0:
                sdf_loss = torch.tensor(0).float().cuda()
            else:
                sdf_loss = (f_mls[weight.sum(1) > 1e-6] - f_mlp[weight.sum(1) > 1e-6]).abs().mean()

            sdf_loss += opt.lambda_eki*(normal_q.norm(dim=-1) - 1.0).abs().mean() # eki loss
            
            sdf_loss += opt.lambda_normal_consistency * ((render_pkg['opacity'].squeeze()*(1-(normal_q[:points_num, :] * render_pkg['normal']).sum(-1)))).abs().mean()
        else:
            sdf_loss = torch.tensor(0).cuda()
            
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss + 0.01*scaling_reg 
        loss += opt.lambda_depth*depth_loss
        loss += opt.lambda_normal_cos*normal_cos_loss +opt.lambda_normal_l1* normal_l1_loss  # 
        loss += opt.lambda_render_norm_reg * normal_grad_loss 
        loss += sdf_loss
        
        if torch.isnan(loss):
            print(viewspace_point_tensor.grad)
            assert not torch.isnan(loss), 'Failed at iter: {}'.format(iteration)
        loss.backward()
        
        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, dataset_name, iteration, Ll1, loss, sdf_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background), wandb, logger)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
            
            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                # add statis
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)
                
                # densification
                if iteration > opt.update_from and iteration % opt.update_interval == 0:
                    gaussians.adjust_anchor(check_interval=opt.update_interval, success_threshold=opt.success_threshold, grad_threshold=opt.densify_grad_threshold, min_opacity=opt.min_opacity)
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
            
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, dataset_name, iteration, Ll1, loss, sdf_loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, wandb=None, logger=None):
    if tb_writer:
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar(f'{dataset_name}/iter_time', elapsed, iteration)


    if wandb is not None:
        wandb.log({"train_l1_loss":Ll1, 'train_total_loss':loss, 'sdf_loss': sdf_loss, 'anchor_number': scene.gaussians._anchor.shape[0], 'density': scene.gaussians.density.get_beta()})
    
    # Report test and samples of training set
    if iteration in testing_iterations:
        scene.gaussians.eval()
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                
                if wandb is not None:
                    gt_image_list = []
                    render_image_list = []
                    errormap_list = []

                for idx, viewpoint in enumerate(config['cameras']):
                    voxel_visible_mask = prefilter_voxel(viewpoint, scene.gaussians, *renderArgs)
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs, visible_mask=voxel_visible_mask)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 30):
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/errormap".format(viewpoint.image_name), (gt_image[None]-image[None]).abs(), global_step=iteration)

                        if wandb:
                            render_image_list.append(image[None])
                            errormap_list.append((gt_image[None]-image[None]).abs())
                            
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(f'{dataset_name}/'+config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            if wandb:
                                gt_image_list.append(gt_image[None])

                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()

                
                
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                logger.info("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                
                if tb_writer:
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(f'{dataset_name}/'+config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                if wandb is not None:
                    wandb.log({f"{config['name']}_loss_viewpoint_l1_loss":l1_test, f"{config['name']}_PSNR":psnr_test})

        if tb_writer:
            # tb_writer.add_histogram(f'{dataset_name}/'+"scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar(f'{dataset_name}/'+'total_points', scene.gaussians.get_anchor.shape[0], iteration)
        torch.cuda.empty_cache()

        scene.gaussians.train()

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, poisson_depth=8):
    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depth_renders")

    makedirs(render_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    
    t_list = []
    visible_count_list = []
    name_list = []
    per_view_dict = {}
    
    render_pkg = render(views[0], gaussians, pipeline, background, visible_mask=None)
    points = render_pkg["points"].cpu().detach().numpy()
    points_normals = torch.nn.functional.normalize(render_pkg["normal"], dim=-1).cpu().detach().numpy()
    vertices, triangle, pcd = poisson_surface_reconstruction(points, points_normals, poisson_depth)
    import open3d as o3d
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangle)
    mesh.vertex_normals = o3d.utility.Vector3dVector(points_normals)
    o3d.io.write_triangle_mesh(os.path.join(model_path, name, "ours_{}".format(iteration) + ".ply"), mesh)
    
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        
        torch.cuda.synchronize();t_start = time.time()
        
        voxel_visible_mask = prefilter_voxel(view, gaussians, pipeline, background)
        render_pkg = render(view, gaussians, pipeline, background, visible_mask=voxel_visible_mask)
        torch.cuda.synchronize();t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = (render_pkg["radii"] > 0).sum()
        visible_count_list.append(visible_count)
        
        # add depth rendering
        render_depth = render_pkg["render_depth"]
        gt_depth = view.depth
        scale, shift = compute_scale_and_shift(render_depth, gt_depth, gt_depth>0)
        depth = render_depth * scale + shift
        
        depth_concat = torch.cat((depth, gt_depth), dim=0).unsqueeze(1)
        tensor = torchvision.utils.make_grid(depth_concat, padding=0, normalize=False, scale_each=False).cpu().detach().numpy()
        plt.imsave(os.path.join(depth_path, '{0:05d}'.format(idx) + "_depth.png"), np.transpose(tensor, (1,2,0))[:,:,0], cmap="viridis")
        
        # add normal rendering
        render_normal = render_pkg["render_normal"]
        gt_normal = view.normal
        normal_concat = torch.stack((render_normal, gt_normal), dim=0)
        normal_concat = (normal_concat + 1)/2.0
        tensor = torchvision.utils.make_grid(normal_concat, padding=0, normalize=False, scale_each=False).cpu().detach().numpy()
        plt.imsave(os.path.join(depth_path, '{0:05d}'.format(idx) + "_normal.png"), (255*np.transpose(tensor, (1,2,0))).astype(np.uint8))
        
        # gts
        gt = view.original_image[0:3, :, :]
        
        # error maps
        errormap = (rendering - gt).abs()


        name_list.append('{0:05d}'.format(idx) + ".png")
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()
        
    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)
    
    return t_list, visible_count_list

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train=True, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None, poisson_depth=9):
    with torch.no_grad():
        # temoporally set the eval to True for visualization, if you are using the entire dataset for training, the numerical output is for subset of training set
        dataset.eval = True
        gaussians = GaussianModel(dataset.feat_dim, dataset.n_offsets, dataset.voxel_size, dataset.update_depth, dataset.update_init_factor, dataset.update_hierachy_factor, dataset.use_feat_bank)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        gaussians.eval()

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
            t_train_list, _  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, poisson_depth=poisson_depth)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            logger.info(f'Train FPS: \033[1;35m{train_fps.item():.5f}\033[0m')
            if wandb is not None:
                wandb.log({"train_fps":train_fps.item(), })

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, poisson_depth=poisson_depth)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            logger.info(f'Test FPS: \033[1;35m{test_fps.item():.5f}\033[0m')
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            if wandb is not None:
                wandb.log({"test_fps":test_fps, })
    
    return visible_count


def readImages(renders_dir, gt_dir):
    renders = []
    gts = []
    image_names = []
    for fname in os.listdir(renders_dir):
        if os.path.isdir(renders_dir / fname):
            continue
        render = Image.open(renders_dir / fname)
        gt = Image.open(gt_dir / fname)
        renders.append(tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda())
        gts.append(tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda())
        image_names.append(fname)
    return renders, gts, image_names


def evaluate(model_paths, visible_count=None, wandb=None, tb_writer=None, dataset_name=None, logger=None):

    full_dict = {}
    per_view_dict = {}
    full_dict_polytopeonly = {}
    per_view_dict_polytopeonly = {}
    print("")
    
    scene_dir = model_paths
    full_dict[scene_dir] = {}
    per_view_dict[scene_dir] = {}
    full_dict_polytopeonly[scene_dir] = {}
    per_view_dict_polytopeonly[scene_dir] = {}

    test_dir = Path(scene_dir) / "test"

    for method in os.listdir(test_dir):
        if not os.path.isdir(test_dir / method):
            continue
        full_dict[scene_dir][method] = {}
        per_view_dict[scene_dir][method] = {}
        full_dict_polytopeonly[scene_dir][method] = {}
        per_view_dict_polytopeonly[scene_dir][method] = {}

        method_dir = test_dir / method
        gt_dir = method_dir/ "gt"
        renders_dir = method_dir / "renders"
        renders, gts, image_names = readImages(renders_dir, gt_dir)

        ssims = []
        psnrs = []
        lpipss = []

        for idx in tqdm(range(len(renders)), desc="Metric evaluation progress"):
            ssims.append(ssim(renders[idx], gts[idx]))
            psnrs.append(psnr(renders[idx], gts[idx]))
            lpipss.append(lpips_fn(renders[idx], gts[idx]).detach())
        
        if wandb is not None:
            wandb.log({"test_SSIMS":torch.stack(ssims).mean().item(), })
            wandb.log({"test_PSNR_final":torch.stack(psnrs).mean().item(), })
            wandb.log({"test_LPIPS":torch.stack(lpipss).mean().item(), })

        logger.info(f"model_paths: \033[1;35m{model_paths}\033[0m")
        logger.info("  SSIM : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(ssims).mean(), ".5"))
        logger.info("  PSNR : \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(psnrs).mean(), ".5"))
        logger.info("  LPIPS: \033[1;35m{:>12.7f}\033[0m".format(torch.tensor(lpipss).mean(), ".5"))
        print("")


        if tb_writer:
            tb_writer.add_scalar(f'{dataset_name}/SSIM', torch.tensor(ssims).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/PSNR', torch.tensor(psnrs).mean().item(), 0)
            tb_writer.add_scalar(f'{dataset_name}/LPIPS', torch.tensor(lpipss).mean().item(), 0)
            
            tb_writer.add_scalar(f'{dataset_name}/VISIBLE_NUMS', torch.tensor(visible_count).mean().item(), 0)
        
        full_dict[scene_dir][method].update({"SSIM": torch.tensor(ssims).mean().item(),
                                                "PSNR": torch.tensor(psnrs).mean().item(),
                                                "LPIPS": torch.tensor(lpipss).mean().item()})
        per_view_dict[scene_dir][method].update({"SSIM": {name: ssim for ssim, name in zip(torch.tensor(ssims).tolist(), image_names)},
                                                    "PSNR": {name: psnr for psnr, name in zip(torch.tensor(psnrs).tolist(), image_names)},
                                                    "LPIPS": {name: lp for lp, name in zip(torch.tensor(lpipss).tolist(), image_names)},
                                                    "VISIBLE_COUNT": {name: vc for vc, name in zip(torch.tensor(visible_count).tolist(), image_names)}})

    with open(scene_dir + "/results.json", 'w') as fp:
        json.dump(full_dict[scene_dir], fp, indent=True)
    with open(scene_dir + "/per_view.json", 'w') as fp:
        json.dump(per_view_dict[scene_dir], fp, indent=True)
    
def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO) 
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO) 
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    # parser.add_argument('--ip', type=str, default="127.0.0.1")
    # parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 7_000, 15_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[3_000, 7_000, 15_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--knear", type=int, default=50)
    parser.add_argument("--ps_depth", type=int, default = 8)
    
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    try:
        saveRuntimeCode(os.path.join(args.model_path))
    except:
        print('save code failed')
    
    # enable logging
    
    model_path = args.model_path
    assert model_path.startswith('outputs'), "Model path should start with outputs prefix to prevent recursive code backup"
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)


    logger.info(f'args: {args}')

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        os.system("echo $CUDA_VISIBLE_DEVICES")
        logger.info(f'using GPU {args.gpu}')

        
    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]
    
    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"GSrec",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None
    
    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    
    # training

    training(lp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger, k_near=args.knear)

    # All done
    logger.info("\nTraining complete.")

    # rendering
    logger.info(f'\nStarting Rendering~')
    visible_count = render_sets(lp.extract(args), -1, pp.extract(args), wandb=wandb, logger=logger, poisson_depth=args.ps_depth)
    logger.info("\nRendering complete.")

    # calc metrics
    logger.info("\n Starting evaluation...")
    evaluate(args.model_path, visible_count=visible_count, wandb=wandb, logger=logger)
    logger.info("\nEvaluating complete.")
