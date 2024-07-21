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
import torch
from einops import repeat

import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
# from scene.gaussian_model import GaussianModel
from scene.gaussian_model_implicit import GaussianModel
from utils.general_utils import GetMinEigenVector, build_scaling_rotation, build_scaling_inv_rotation, mls_sdf, strip_symmetric

def generate_neural_gaussians_SDF(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    
    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3]

    # get offset's opacity
    # neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    neural_opacity = pc.get_opacity_mlp(feat) # [N, k], discard the view information in opacity

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)
    
    assert mask.sum() > 0, "no visible gaussians"

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    color = pc.get_color_mlp(cat_local_view)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    # scale_rot = pc.get_cov_mlp(cat_local_view)
    scale_rot = pc.get_cov_mlp(feat) # discard the view information in the cov mlp
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    assert torch.isnan(scale_rot).sum() == 0, "scale_rot has nan"
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    
    # normal: use learnable parameter
    # grid_normals = pc._normal[visible_mask]
    # normal = grid_normals.view([-1, 3]) # [mask]

    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    
    # calculate the IMLS, reference: "Provably Good Moving Least Squares" by Kolluri
    normal_unmask = GetMinEigenVector(concatenated_all[:, 3:6]*torch.sigmoid(scale_rot[:, :3]), pc.rotation_activation(scale_rot[:,3:7])).view(-1, 3) # [NK, 3]
    L = build_scaling_inv_rotation(concatenated_all[:, 3:6]*torch.sigmoid(scale_rot[:, :3]), pc.rotation_activation(scale_rot[:,3:7])) # [NK, 3, 3]
    actual_cov =  (L @ L.transpose(1, 2))# NK, 3, 3
    cov3D_unmask = strip_symmetric(actual_cov) # NK, 6
    
    
    offsets_unmask = (offsets * concatenated_all[:, :3]).view(-1, 3) # Nk, 3
    
    # use all offset3d
    # offset_3d_all = (offset_3d.unsqueeze(2) - offset_3d.unsqueeze(1)) # N, k, k, 3
    # offset_3d_all = offset_3d_all.view(-1, pc.n_offsets, 3) # NK, K, 3
    

    # results = offset_3d_all.unsqueeze(-2)@cov3D.unsqueeze(1)@offset_3d_all.unsqueeze(-1) # NK, K, 1, 1
    # density = torch.exp(-0.5* neural_opacity.unsqueeze(-1) * results.squeeze(-1)) # Nk, k, 1
    
    # estimated_imls = density * (offset_3d_all @ normal_unmask.unsqueeze(-1)) # Nk, k, 1
    # mask_point = (mask.view(-1, pc.n_offsets, 1) * mask.view(-1, 1, pc.n_offsets)).view(-1, pc.n_offsets, 1)
    # estimated_imls = estimated_imls * mask_point # mask out the negative opacity
    # density_denom = density * mask_point
    # estimated_imls = estimated_imls.sum(1) / (density_denom.sum(1)+1e-6)
    # estimated_imls = estimated_imls.view(-1, pc.n_offsets) * mask.view(-1, pc.n_offsets)
    # estimated_imls = estimated_imls.sum(1) / ((density.view(-1, pc.n_offsets) * mask.view(-1, pc.n_offsets)).sum(1)+1e-6)
    
    
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)
    
    
    # post-process cov
    scaling = torch.nan_to_num(scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])) # * (1+torch.sigmoid(repeat_dist))
    
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    
    # # normal: use the minimum egien vector corresponding to scaling to decide the normal vector
    # grid_normal = GetMinEigenVector(pc.rotation_activation(scale_rot[:,3:7]), scale_rot[:,:3])
    normal = GetMinEigenVector(scaling, rot) 
    
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets
    
    if is_training:
        return xyz, color, opacity, scaling, rot, normal, offsets_unmask, normal_unmask, cov3D_unmask, anchor, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot, normal, offsets_unmask, normal_unmask, cov3D_unmask, anchor

def generate_neural_gaussians(viewpoint_camera, pc : GaussianModel, visible_mask=None, is_training=False):
    ## view frustum filtering for acceleration    
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)
    
    feat = pc._anchor_feat[visible_mask]
    anchor = pc.get_anchor[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]
    


    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    
    ## view-adaptive feature
    if pc.use_feat_bank:
        cat_view = torch.cat([ob_view, ob_dist], dim=1)
        
        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1) # [n, 1, 3]

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1) # [n, c]


    cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1) # [N, c+3]

    # get offset's opacity
    # neural_opacity = pc.get_opacity_mlp(cat_local_view) # [N, k]
    neural_opacity = pc.get_opacity_mlp(feat) # [N, k], discard the view information in opacity

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)
    
    assert mask.sum() > 0, "no visible gaussians"

    # select opacity 
    opacity = neural_opacity[mask]

    # get offset's color
    color = pc.get_color_mlp(cat_local_view)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    # scale_rot = pc.get_cov_mlp(cat_local_view)
    scale_rot = pc.get_cov_mlp(feat) # discard the view information in the cov mlp
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]
    
    assert torch.isnan(scale_rot).sum() == 0, "scale_rot has nan"
    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]
    
    
    # normal: use learnable parameter
    # grid_normals = pc._normal[visible_mask]
    # normal = grid_normals.view([-1, 3]) # [mask]


    
    
    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets= masked.split([6, 3, 3, 7, 3], dim=-1)
    
    
    # post-process cov
    scaling = torch.nan_to_num(scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])) # * (1+torch.sigmoid(repeat_dist))
    
    rot = pc.rotation_activation(scale_rot[:,3:7])
    
    
    # # normal: use the minimum egien vector corresponding to scaling to decide the normal vector
    # grid_normal = GetMinEigenVector(pc.rotation_activation(scale_rot[:,3:7]), scale_rot[:,:3])
    normal = GetMinEigenVector(scaling, rot) 
    
    
    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets
    

    if is_training:
        return xyz, color, opacity, scaling, rot, normal, neural_opacity, mask
    else:
        return xyz, color, opacity, scaling, rot, normal

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, visible_mask=None, retain_grad=False, learn_SDF=True):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    is_training = pc.get_color_mlp.training
        
    if is_training:
        if learn_SDF:
            xyz, color, opacity, scaling, rot, normal, neural_opacity, mask = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
        else:
            xyz, color, opacity, scaling, rot, normal, offsets_unmask, normal_unmask, cov3D_unmask, visbile_anchor, neural_opacity, mask = generate_neural_gaussians_SDF(viewpoint_camera, pc, visible_mask, is_training=is_training)
    else:
        if learn_SDF:
            xyz, color, opacity, scaling, rot, normal = generate_neural_gaussians(viewpoint_camera, pc, visible_mask, is_training=is_training)
        else:
            xyz, color, opacity, scaling, rot, normal, offsets_unmask, normal_unmask, cov3D_unmask, visbile_anchor = generate_neural_gaussians_SDF(viewpoint_camera, pc, visible_mask, is_training=is_training)
    
    assert scaling is not None

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass


    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    
    # use the minimum egien vector corresponding to scaling to decide the normal vector
    
    
    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_depth, render_normal, render_median_depth, radii = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color,
        normal_precomp = pc.normal_activation(normal),
        # opacities = (opacity > 0).float().cuda(),
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None)
    
    # normalize the render normal
    render_normal = torch.nn.functional.normalize(render_normal, dim=0)
    render_normal = render_normal.contiguous()
    
    L = build_scaling_inv_rotation(scaling, rot)
    actu_cov3D = (L @ L.transpose(1, 2))
    cov3D = strip_symmetric(actu_cov3D)
    
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        return {"render": rendered_image,
                "render_depth": rendered_depth,
                "render_median_depth": render_median_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                "points": xyz,
                "render_normal": render_normal,
                "normal": normal,
                "cov3D": cov3D,
                "offsets_unmask": None if learn_SDF else offsets_unmask,
                "normal_unmask": None if learn_SDF else normal_unmask,
                "cov3D_unmask": None if learn_SDF else cov3D_unmask,
                "opacity": opacity,
                "visible_anchor": None if learn_SDF else visbile_anchor
                }
    else:
        return {"render": rendered_image,
                "render_depth": rendered_depth,
                "render_median_depth": render_median_depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "points": xyz,
                "render_normal": render_normal,
                "normal": normal,
                "cov3D": cov3D,
                "opacity": opacity,
                "offsets_unmask": None if learn_SDF else offsets_unmask,
                "normal_unmask": None if learn_SDF else normal_unmask,
                "cov3D_unmask": None if learn_SDF else cov3D_unmask,
                "visible_anchor": None if learn_SDF else visbile_anchor
                }


def prefilter_voxel(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_anchor, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=1,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_anchor


    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    radii_pure = rasterizer.visible_filter(means3D = means3D,
        scales = scales[:,:3],
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    return radii_pure > 0
