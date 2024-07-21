import numpy as np
import matplotlib.pyplot as plt


# poisson surface reconstruction
def poisson_surface_reconstruction(points, normals=None, depth=9):
    # point/normals, N x 3
    
    import open3d as o3d
    
    ori_pcd = o3d.geometry.PointCloud()    
    ori_pcd.points = o3d.utility.Vector3dVector(points)
    # remove outliers
    # import pdb; pdb.set_trace()
    # pcd, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=10)

    
    if normals is None:
        ori_pcd.estimate_normals()
    else:
        ori_pcd.normals = o3d.utility.Vector3dVector(normals)
        
    # remove points outside of a cube
    R = np.identity(3)
    extent = np.ones(3)*2
    center = np.zeros(3)
    obb = o3d.geometry.OrientedBoundingBox(center, R, extent)
    pcd = ori_pcd.crop(obb)
        
    mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=depth,linear_fit=True)
    # o3d.visualization.draw_geometries([mesh])
    
    ## visualize density
    
    densities = np.asarray(densities)
    density_colors = plt.get_cmap('plasma')(
        (densities - densities.min()) / (densities.max() - densities.min()))
    density_colors = density_colors[:, :3]
    density_mesh = o3d.geometry.TriangleMesh()
    density_mesh.vertices = mesh.vertices
    density_mesh.triangles = mesh.triangles
    density_mesh.triangle_normals = mesh.triangle_normals
    density_mesh.vertex_colors = o3d.utility.Vector3dVector(density_colors)
    # o3d.visualization.draw_geometries([density_mesh])
    
    # vertices_to_remove = densities < np.quantile(densities, 0.05)
    # mesh.remove_vertices_by_mask(vertices_to_remove)
    
    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    
    return vertices, triangles, pcd
    
# marching cube 
