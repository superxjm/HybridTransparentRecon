from __future__ import division

import PIL.Image

import trimesh
import numpy as np
import cv2
import trimesh
import torch
import open3d as o3d
import pytorch3d.io

def refine_visual_hull(masks, Ps, scale, center):
    num_cam=masks.shape[0]
    GRID_SIZE=100
    MINIMAL_VIEWS=45 # Fitted for DTU, might need to change for different data.
    im_height=masks.shape[1]
    im_width = masks.shape[2]
    xx, yy, zz = np.meshgrid(np.linspace(-scale, scale, GRID_SIZE), np.linspace(-scale, scale, GRID_SIZE),
                             np.linspace(-scale, scale, GRID_SIZE))
    points = np.stack((xx.flatten(), yy.flatten(), zz.flatten()))
    points = points + center[:, np.newaxis]
    appears = np.zeros((GRID_SIZE*GRID_SIZE*GRID_SIZE, 1))
    for i in range(num_cam):
        proji = Ps[i] @ np.concatenate((points, np.ones((1, GRID_SIZE*GRID_SIZE*GRID_SIZE))), axis=0)
        depths = proji[2]
        proj_pixels = np.round(proji[:2] / depths).astype(np.long)
        relevant_inds = np.logical_and(proj_pixels[0] >= 0, proj_pixels[1] < im_height)
        relevant_inds = np.logical_and(relevant_inds, proj_pixels[0] < im_width)
        relevant_inds = np.logical_and(relevant_inds, proj_pixels[1] >= 0)
        relevant_inds = np.logical_and(relevant_inds, depths > 0)
        relevant_inds = np.where(relevant_inds)[0]

        cur_mask = masks[i] > 0.5
        relmask = cur_mask[proj_pixels[1, relevant_inds], proj_pixels[0, relevant_inds]]
        relevant_inds = relevant_inds[relmask]
        appears[relevant_inds] = appears[relevant_inds] + 1

    final_points = points[:, (appears >= MINIMAL_VIEWS).flatten()]
    centroid=final_points.mean(axis=1)
    normalize = final_points - centroid[:, np.newaxis]

    return centroid,np.sqrt((normalize ** 2).sum(axis=0)).mean() * 3,final_points.T

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

    root_dir = "../real_data_cow/"

    mesh = trimesh.load(root_dir + 'object_proxy.ply', force='mesh', use_embree=True)
    median = np.mean(mesh.bounds[:, :], axis=0)
    size = np.max(mesh.bounds[1, :] - mesh.bounds[0, :], axis=0)
    print(size)
    t_offset = -median
    # old: scale = 1.2 / size
    # scale = 0.8 / size
    scale = 0.8 / size
    print(scale)

    # with open("../maya_data/camera_params_uniform_sample_interpolated.log", 'r') as f:
    with open(root_dir + "camera_params.log", 'r') as f:
	    lines = f.readlines()
    Ks = []
    camera_poses = []
    resolutions = []
    lines = lines[1:]
    for line, i in zip(lines, np.arange(len(lines))):
        if i % 7 == 0:
            print("name: ", line)
        if i % 7 == 1:
            print("intrin: ", line)
            intrin = list(map(float, line.strip().split(' ')))
            # intrin = [2917.09632869, 2917.09632869, 511.5, 511.5]#[3889.03219785, 3889.03219785, 511.5, 511.5]#[2903.70082896, 2903.70082896, 511.5, 511.5]
            Ks.append([intrin[0], 0, intrin[2], 0, intrin[1], intrin[3], 0, 0, 1])
        if i % 7 == 2:
            resolution = list(map(float, line.strip().split()))
            # resolution = [1024, 1024]
            resolutions.append([resolution[0], resolution[1]])
        if i % 7 >= 3 and i % 7 <= 6:
            camera_poses.append(list(map(float, line.strip().split())))

    Ks = np.array(Ks)
    Ks = Ks.reshape(-1, 3, 3)
    resolutions = np.array(resolutions)
    resolutions = resolutions.reshape(-1, 2)
    camera_poses = np.array(camera_poses)
    camera_poses = camera_poses.reshape(-1, 4, 4)
    camera_poses[:, :3, 3] += t_offset
    camera_poses[:, :3, 3] *= scale
    print(camera_poses)
    Ps = torch.eye(4).unsqueeze(0).repeat(camera_poses.shape[0], 1, 1).numpy()
    Ps[:,:3,:] = np.matmul(Ks, np.linalg.inv(camera_poses)[:,:3,:])

    all_camera_lines = []
    for i in range(camera_poses.shape[0]):
        camera_lines = create_camera_actor(scale = 0.2)
        all_camera_lines.append(camera_lines.transform(camera_poses[i, :, :]))

    mesh_box = o3d.geometry.TriangleMesh.create_box(width=1,
                                                    height=1,
                                                    depth=1)
    mesh_box.compute_vertex_normals()
    mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
    all_camera_lines.append(mesh_box)

    o3d.visualization.draw_geometries(geometry_list=all_camera_lines, width = 640, height = 480, window_name = "mesh_with_hit_points")
    
    camera_dict = {f'scale_mat_{i}' : torch.eye(4).numpy() for i in np.arange(camera_poses.shape[0])}
    camera_dict.update({f'world_mat_{i}' : Ps[i, :, :] for i in np.arange(camera_poses.shape[0])})

    np.savez(root_dir + '/cameras', **camera_dict)
    mesh.vertices += t_offset
    mesh.vertices *= scale
    mesh.export(root_dir + '/gt_mesh.ply')

    scene_mesh = trimesh.load(root_dir + '/scene_mesh.obj', force='mesh')
    scene_mesh.vertices += t_offset
    scene_mesh.vertices *= scale
    scene_mesh.export(root_dir + '/scaled_scene_mesh.obj')

    for board_idx in range(30):
        print('board_idx: {0}'.format(board_idx))
        mesh = trimesh.load(root_dir + '/env_matting/board_' + str(board_idx) + '.obj', force='mesh')
        print(mesh.vertices.shape)
        mesh.vertices += t_offset
        mesh.vertices *= scale
        mesh.export(root_dir + '/env_matting/scaled_board_' + str(board_idx) + '.obj')

    for board_scene_idx in range(2):
        print('board_scene_idx: {0}'.format(board_scene_idx))
        mesh = trimesh.load(root_dir + '/board_' + str(board_scene_idx) + '_mesh_with_texture_clip.obj', force='mesh')
        print(mesh.vertices.shape)
        mesh.vertices += t_offset
        mesh.vertices *= scale
        mesh.export(root_dir + '/scaled_board_' + str(board_scene_idx) + '_mesh_with_texture_clip.obj')

    with open(root_dir + "/scaled_camera_params.log", 'w') as f:
        f.writelines("NO_CAPTURE_REALITY\n")
        for i in range(camera_poses.shape[0]):
            f.writelines(str(i))
            f.writelines("\n")
            f.writelines(str(Ks[i, 0, 0]) + " " + str(Ks[i, 1, 1]) + " " + str(Ks[i, 0, 2]) + " " + str(Ks[i, 1, 2]))
            f.writelines("\n")
            f.writelines(str(resolutions[i, 0]) + " " + str(resolutions[i, 1]) + " 0.1 100.0")
            f.writelines("\n")
            for j in range(4):
                nparray_str = str(camera_poses[i, j, :])
                nparray_str = nparray_str[1:-1].strip()
                nparray_str = " ".join(nparray_str.split())
                f.writelines(nparray_str)
                f.writelines("\n")
   