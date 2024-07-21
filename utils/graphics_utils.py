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
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def getinv_intrinsic(fovX, fovY, W, H):
    K = torch.zeros(3, 3)
    K[0, 0] = W/ (2 * math.tan(fovX * 0.5))
    K[1, 1] = H / (2 * math.tan(fovY * 0.5))
    K[0, 2] = W / 2.0
    K[1, 2] = H / 2.0
    K[2, 2] = 1.0
    
    return torch.inverse(K)

def get_intrinsic(fovX, fovY, W, H):
    K = torch.zeros(3, 3)
    K[0, 0] = W/ (2 * math.tan(fovX * 0.5))
    K[1, 1] = H / (2 * math.tan(fovY * 0.5))
    K[0, 2] = W / 2.0
    K[1, 2] = H / 2.0
    K[2, 2] = 1.0
    
    return K

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))

def sample_pixels(W, H, K, ignore_ratio=0.0):
    # Create a grid of pixel coordinates (H x W)
    # shrink the sample range to avoid sampling the border. The depth of GS near bonundary is not accurate
    offsets = int(ignore_ratio*H) # ignore ratio of the image size
    diff = (W-H)//2
    x_coords = torch.arange(diff+offsets, W-diff-offsets).repeat(H - 2*offsets, 1)
    y_coords = torch.arange(offsets, H-offsets).unsqueeze(1).repeat(1, W-2*diff-2*offsets)
    
    # x_coords = torch.arange(0, W).repeat(H, 1)
    # y_coords = torch.arange(0, H).unsqueeze(1).repeat(1, W)
    
    # Stack and reshape to get a list of coordinates
    # all_coords = torch.stack((x_coords, y_coords), dim=2).view(-1, 2)
    all_coords = torch.stack((y_coords, x_coords), dim=2).view(-1, 2) # [0~H, 0~W]
    
    # Randomly sample K pixels
    sampled_indices = torch.randperm(all_coords.size(0))[:K]
    # sampled_pixels = all_coords[sampled_indices]
    
    return all_coords[sampled_indices]

def unproject_3d(coordinates, inv_intrinsic, depth, c2w):
    # coordinates: B, 2: (x, y) in image space
    uv_coordinates = torch.cat([coordinates.flip(1), torch.ones_like(coordinates[:, :1])], dim=1)
    unproj = (inv_intrinsic @ uv_coordinates.transpose(1, 0).float()).transpose(0, 1)
    camera_coordinates = unproj * depth.unsqueeze(-1)
    # convert to world coordinates
    world_coordinates = (c2w[:3, :3] @ (camera_coordinates - c2w[3, :3]).transpose(0, 1)).transpose(1, 0)
    # world_coordinates = c2w[:3, :3] @ camera_coordinates + c2w[:3, 3].unsqueeze(dim=1)
    return world_coordinates


def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz

def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    xyz_world = xyz_world[...,:3]

    return xyz_world

def depth2point(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    xyz_world = xyz_world[...,:3]

    return xyz_cam.reshape(*depth_image.shape, 3), xyz_world.reshape(*depth_image.shape, 3)