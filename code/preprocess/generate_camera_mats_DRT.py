from __future__ import division

import PIL.Image

import trimesh
import numpy as np
import cv2
import trimesh
import torch
import open3d as o3d
import pytorch3d.io
import h5py
import os

    # self.resy=1080
    #     self.resx=1920
    #     self.num_view = HyperParams['num_view']
    #     self.name = HyperParams['name']
    #     scene = Render.Scene(f"{config.data_path}{name}_vh.ply")
    #     self.Views = []


def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

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

def create_point_cloud_actor(points, colors):
    """ open3d point cloud from numpy array """

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    return point_cloud

if __name__ == '__main__':
    
    root_dir = "/home/2TB/xjm/DRT/data/"
    output_dir = "../DRT_data_tiger/"
    mkdir_ifnotexists(output_dir)
    mkdir_ifnotexists(output_dir + "/mask/")
    mkdir_ifnotexists(output_dir + "/image/")
    mkdir_ifnotexists(output_dir + "/mask_loss/")
    mkdir_ifnotexists(output_dir + "/mask_loss/mask/")
    mkdir_ifnotexists(output_dir + "/corr/")
    mkdir_ifnotexists(output_dir + "/corr/vis/")
    mesh = trimesh.load(root_dir + 'tiger_vh.ply', force='mesh', use_embree=True, process=False)
    gt_mesh = trimesh.load(root_dir + 'tiger_scan.ply', force='mesh', use_embree=True, process=False)
    DRT_mesh_72view = trimesh.load(root_dir + '../result/tiger_recons.ply', force='mesh', use_embree=True, process=False)
    # DRT_mesh_72view = trimesh.load(output_dir + '/DRT/horse_recons_72view.ply', force='mesh', use_embree=True, process=False)
    median = np.mean(mesh.bounds[:, :], axis=0)
    size = np.max(mesh.bounds[1, :] - mesh.bounds[0, :], axis=0)
    print(size)
    t_offset = -median
    # old: scale = 1.2 / size
    # scale = 0.8 / size
    scale = 0.8 / size
    print(median)
    print(scale)
    input()

    h5data = h5py.File(root_dir + 'tiger.h5','r')

    print('loading data..............')
    Ks = []
    camera_poses = []
    for i in range(72):
        R = h5data['cam_proj'][i]
        K = h5data['cam_k'][:]
        R_inverse = np.linalg.inv(R)
        K_inverse = np.linalg.inv(K)
        screen_pixel = h5data['screen_position'][i].reshape([-1,3])
        target = screen_pixel
        mask = h5data['mask'][i]
        valid = screen_pixel[:,0] != 0
        Ks.append(K)
        camera_poses.append(R_inverse)
        print("{0}/mask/{1:0>5d}.png".format(output_dir, i))
        cv2.imwrite("{0}/mask/{1:0>5d}.png".format(output_dir, i), mask)
        cv2.imwrite("{0}/mask_loss/mask/{1:0>5d}.png".format(output_dir, i), mask)
        cv2.imwrite("{0}/image/{1:0>5d}.png".format(output_dir, i), np.zeros((mask.shape[0], mask.shape[1], 3)))
        
        print(target.shape)
        print(target.dtype)
        valid = (target[:,0] != 0)
        target[valid, :] += t_offset
        target[valid, :] *= scale
        target = target.reshape(1080, -1, 3).astype(np.float32)
        cv2.imwrite("{0}/corr/{1:0>5d}.exr".format(output_dir, i), target)
        target_vis = (target * 10).astype(np.uint8)
        print(target_vis.shape)
        print(target_vis.dtype)
        cv2.imwrite("{0}/corr/vis/{1:0>5d}_vis.png".format(output_dir, i), target_vis)
        # output_dir

    h5data.close()
    Ks = np.array(Ks)
    Ks = Ks.reshape(-1, 3, 3)
    # resolutions = np.array(resolutions)
    # resolutions = resolutions.reshape(-1, 2)
    camera_poses = np.array(camera_poses)
    camera_poses = camera_poses.reshape(-1, 4, 4)

    ori_camera_poses = camera_poses.copy() 

    camera_poses[:, :3, 3] += t_offset
    camera_poses[:, :3, 3] *= scale
    # print(camera_poses)
    Ps = torch.eye(4).unsqueeze(0).repeat(camera_poses.shape[0], 1, 1).numpy()
    Ps[:,:3,:] = np.matmul(Ks, np.linalg.inv(camera_poses)[:,:3,:])

    all_camera_lines = []
    for i in range(camera_poses.shape[0]):
        camera_lines = create_camera_actor(scale = 0.2)
        all_camera_lines.append(camera_lines.transform(camera_poses[i, :, :]))
    print('t_offset')
    print(t_offset)
    print('scale')
    print(scale)

    mesh.vertices += t_offset
    mesh.vertices *= scale
    gt_mesh.vertices += t_offset
    gt_mesh.vertices *= scale
    mesh.export(output_dir + '/idr_surface.ply')
    gt_mesh.export(output_dir + '/gt_mesh.ply')
    # DRT_mesh_18view.vertices += t_offset
    # DRT_mesh_18view.vertices *= scale
    # DRT_mesh_18view.export(output_dir + '/DRT/horse_recons_9view_scale_3.ply')
    DRT_mesh_72view.vertices += t_offset
    DRT_mesh_72view.vertices *= scale
    DRT_mesh_72view.export(root_dir + '../result/tiger_recons_scale.ply')

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mesh.vertices)
    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1,
                                                    height=1,
                                                    depth=1)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    all_camera_lines.append(pcd)

    o3d.visualization.draw_geometries(geometry_list=all_camera_lines, width = 640, height = 480, window_name = "mesh_with_hit_points")
    
    camera_dict = {f'scale_mat_{i}' : torch.eye(4).numpy() for i in np.arange(camera_poses.shape[0])}
    camera_dict.update({f'world_mat_{i}' : Ps[i, :, :] for i in np.arange(camera_poses.shape[0])})
    np.savez(output_dir + '/cameras', **camera_dict)

    # scene_mesh = trimesh.load(output_dir + '/idr_surface.ply', force='mesh')
    # scene_mesh.vertices /= scale
    # scene_mesh.vertices -= t_offset
    # scene_mesh.export(output_dir + '/idr_surface_DRT.obj')

    # for board_idx in range(30):
    #     print('board_idx: {0}'.format(board_idx))
    #     mesh = trimesh.load(root_dir + '/env_matting/board_' + str(board_idx) + '.obj', force='mesh')
    #     print(mesh.vertices.shape)
    #     mesh.vertices += t_offset
    #     mesh.vertices *= scale
    #     mesh.export(root_dir + '/env_matting/scaled_board_' + str(board_idx) + '.obj')

    # for board_scene_idx in range(2):
    #     print('board_scene_idx: {0}'.format(board_scene_idx))
    #     mesh = trimesh.load(root_dir + '/board_' + str(board_scene_idx) + '_mesh_with_texture_clip.obj', force='mesh')
    #     print(mesh.vertices.shape)
    #     mesh.vertices += t_offset
    #     mesh.vertices *= scale
    #     mesh.export(root_dir + '/scaled_board_' + str(board_scene_idx) + '_mesh_with_texture_clip.obj')

    with open(output_dir + "/camera_params.log", 'w') as f:
        f.writelines("NO_CAPTURE_REALITY\n")
        Ps = []
        for i in range(ori_camera_poses.shape[0]):
            f.writelines(str(i))
            f.writelines("\n")
            f.writelines(str(Ks[i, 0, 0]) + " " + str(Ks[i, 1, 1]) + " " + str(Ks[i, 0, 2]) + " " + str(Ks[i, 1, 2]))
            f.writelines("\n")
            f.writelines(str(1920) + " " + str(1080) + " 0.1 100.0")
            f.writelines("\n")
            for j in range(4):
                nparray_str = str(ori_camera_poses[i, j, :])
                nparray_str = nparray_str[1:-1].strip()
                nparray_str = " ".join(nparray_str.split())
                f.writelines(nparray_str)
                f.writelines("\n")
            P = Ks[i, :, :] @ (np.linalg.inv(ori_camera_poses[i, :, :])[:3, :4])
            print(Ks[i, :, :])
            Ps.append(P)
        np.save(output_dir + 'ori_camera_Ps.npy', Ps)

    with open(output_dir + "/scaled_camera_params.log", 'w') as f:
        f.writelines("NO_CAPTURE_REALITY\n")
        Ps = []
        for i in range(camera_poses.shape[0]):
            f.writelines(str(i))
            f.writelines("\n")
            f.writelines(str(Ks[i, 0, 0]) + " " + str(Ks[i, 1, 1]) + " " + str(Ks[i, 0, 2]) + " " + str(Ks[i, 1, 2]))
            f.writelines("\n")
            f.writelines(str(1920) + " " + str(1080) + " 0.1 100.0")
            f.writelines("\n")
            for j in range(4):
                nparray_str = str(camera_poses[i, j, :])
                nparray_str = nparray_str[1:-1].strip()
                nparray_str = " ".join(nparray_str.split())
                f.writelines(nparray_str)
                f.writelines("\n")
            P = Ks[i, :, :] @ (np.linalg.inv(camera_poses[i, :, :])[:3, :4])
            print(Ks[i, :, :])
            Ps.append(P)
        np.save(output_dir + 'camera_Ps.npy', Ps)
   