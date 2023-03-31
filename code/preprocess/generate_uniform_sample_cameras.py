from pytorch3d.renderer import (
    look_at_view_transform
)
import trimesh
import torch
import numpy as np
import open3d as o3d

def create_camera_actor(is_gt=False, scale=0.05):
    """ build open3d camera polydata """

    cam_points = scale * np.array([
        [ 0,   0,   0],
        [-1,  -1, 1.5],
        [ 1,  -1, 1.5],
        [ 1,   1, 1.5],
        [-1,   1, 1.5],
        [-0.5, 1, 1.5],
        [ 0.5, 1, 1.5],
        [ 0, 1.2, 1.5],
        [ 0,   0,   100]])

    cam_lines = np.array([[1, 2], [2, 3], [3, 4], [4, 1],
        [1, 0], [0, 2], [3, 0], [0, 4], [5, 7], [7, 6], [0, 8]])

    camera_actor = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(cam_points),
        lines=o3d.utility.Vector2iVector(cam_lines))

    color = (0.0, 0.0, 0.0) if is_gt else (0.0, 0.8, 0.8)
    camera_actor.paint_uniform_color(color)

    return camera_actor

if __name__ == '__main__':

    file_dir = '../object_gt/'
    # mesh = trimesh.load('../maya_data/mesh_0.obj', use_embree=True)
    mesh = trimesh.load(file_dir + 'cat_gt_with_base_poisson.ply', use_embree=True)
    median = np.mean(mesh.bounds[:, :], axis=0)
    size = np.max(mesh.bounds[1, :] - mesh.bounds[0, :], axis=0)
    t_offset = -median
    scale = 0.8 / size
    mesh.vertices += t_offset
    mesh.vertices *= scale
    mesh.export(file_dir + '/cat_gt_with_base_poisson_scale.ply')

    # mesh = trimesh.load(file_dir + 'cow_gt_with_base_poisson.ply', use_embree=True)
    # mesh.vertices += t_offset
    # mesh.vertices *= scale
    # mesh.export(file_dir + '/cow_gt_with_base_poisson_scale.ply')
    # mesh = trimesh.load(file_dir + 'dog_gt_with_base_poisson.ply', use_embree=True)
    # mesh.vertices += t_offset
    # mesh.vertices *= scale
    # mesh.export(file_dir + '/dog_gt_with_base_poisson_scale.ply')
    # exit(0)

    mesh = trimesh.load(file_dir + 'cat_gt_with_base_poisson_scale.ply', use_embree=True)
    median = np.mean(mesh.bounds[:, :], axis=0)
    size = np.max(mesh.bounds[1, :] - mesh.bounds[0, :], axis=0)

    num_views = 20
    # Get a batch of viewing angles.
    elev = torch.linspace(0, 30, num_views)
    azim = torch.linspace(-180, 180, num_views)
    R_1, T_1 = look_at_view_transform(dist=size * 5, elev=elev, azim=azim)
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-30, 30, num_views)
    R_2, T_2 = look_at_view_transform(dist=size * 5, elev=elev, azim=azim)
    
    num_views = 20
    elev = torch.linspace(0, 360, num_views)
    azim = torch.linspace(-180, 180, num_views)
    R_3, T_3 = look_at_view_transform(dist=size * 5, elev=elev, azim=azim)
    elev = torch.linspace(-150, 150, num_views)
    azim = torch.linspace(50, 310, num_views)
    R_4, T_4 = look_at_view_transform(dist=size * 5, elev=elev, azim=azim)
    R = torch.cat((R_1, R_2, R_3, R_4),dim=0)
    T = torch.cat((T_1, T_2, T_3, T_4),dim=0)

    # Initialize an OpenGL perspective camera that represents a batch of different 
    # viewing angles. All the cameras helper methods support mixed type inputs and 
    # broadcasting. So we can view the camera from the a distance of dist=2.7, and 
    # then specify elevation and azimuth angles for each viewpoint as tensors. 
    # R, T = look_at_view_transform(dist=size * 5, elev=elev, azim=azim)
    R = R.transpose(-1, -2)
    camera_pose = torch.zeros(R.shape[0], 4, 4)
    camera_pose[:, 0:3, 0:3] = R
    camera_pose[:, 0:3, 3] = T 
    camera_pose[:, 3, 3] = 1
    camera_pose = torch.linalg.inv(camera_pose)
    camera_pose[:, :, 0] = camera_pose[:, :, 0] * -1
    camera_pose[:, :, 1] = camera_pose[:, :, 1] * -1
    camera_pose[:, 0:3, 3] += median[None, None, :]
    print(camera_pose)

    all_camera_lines = []
    for i in range(camera_pose.shape[0]):
        camera_lines = create_camera_actor(scale = 0.2)
        all_camera_lines.append(camera_lines.transform(camera_pose[i, :, :]))
    mash_pcd = o3d.geometry.PointCloud()
    mash_pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    all_camera_lines.append(mash_pcd)
    # mesh_box = o3d.geometry.TriangleMesh.create_box(width=1,
    #                                                 height=1,
    #                                                 depth=1)
    # mesh_box.compute_vertex_normals()
    # mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    # all_camera_lines.append(mesh_box)

    o3d.visualization.draw_geometries(geometry_list=all_camera_lines, width = 640, height = 480, window_name = "mesh_with_hit_points")

    with open(file_dir + "camera_params_uniform_sample.log", 'w') as f:
        f.writelines("NO_CAPTURE_REALITY\n")
        for i in range(camera_pose.shape[0]):
            f.writelines(str(i))
            f.writelines("\n")
            f.writelines("2917.09632869, 2917.09632869, 511.5, 511.5")
            f.writelines("\n")
            f.writelines("1024, 1024, 0.1, 100.0")
            f.writelines("\n")
            for j in range(4):
                nparray_str = str(camera_pose[i, j, :].numpy())
                nparray_str = nparray_str[1:-1].strip()
                nparray_str = " ".join(nparray_str.split())
                f.writelines(nparray_str)
                f.writelines("\n")

  