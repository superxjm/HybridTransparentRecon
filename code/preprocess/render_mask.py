# from __future__ import division

import PIL.Image

import trimesh
import numpy as np
import cv2
import torch
import open3d as o3d
import os
from glob import glob
import tqdm

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

# numpy  get rays
def get_rays_np(H, W, fx, fy, cx, cy, c2w):
    """Get ray origins, directions from a pinhole camera."""
    i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                       np.arange(H, dtype=np.float32), indexing='xy')
    # 也可以理解为摄像机坐标下z=1的平面上的点
    dirs = np.stack([(i-cx)/fx, (j-cy)/fy, np.ones_like(i)], -1) 
    rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
    rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
    return rays_o, rays_d

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG', '*.exr']:
        imgs.extend(glob(os.path.join(path, ext)))
    return imgs

if __name__ == '__main__':

    # with open("./trajectory.txt", 'r') as f:
	#     lines = f.readlines()
    # lines = [list(map(float, x.strip().split(' '))) for x in lines]
    # camera_poses = np.array(lines)
    # camera_poses = camera_poses.reshape(-1, 4, 4)
    # print(camera_poses.shape)
    # input()
    root_dir = "../DRT_data_tiger/"

    with open(root_dir + "scaled_camera_params.log", 'r') as f:
    # with open(root_dir + "camera_params_uniform_sample_interpolated.log", 'r') as f:
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
            # intrin = [2917.09632869, 2917.09632869, 511.5, 511.5]
            Ks.append([intrin[0], 0, intrin[2], 0, intrin[1], intrin[3], 0, 0, 1])
        if i % 7 == 2:
            print("resolution: ", line)
            resolution = list(map(float, line.strip().split(' ')))
            # resolution = [1024, 1024]
            resolutions.append([resolution[0], resolution[1]])
        if i % 7 >= 3 and i % 7 <= 6:
            print("camera_poses: ", line)
            camera_poses.append(list(map(float, line.strip().split())))
    camera_poses = np.array(camera_poses)
    camera_poses = camera_poses.reshape(-1, 4, 4)
    Ks = np.array(Ks)
    Ks = Ks.reshape(-1, 3, 3)
    resolutions = np.array(resolutions)
    resolutions = resolutions.reshape(-1, 2)
    # cam_file_path = '../real_data/cameras.npz'
    # n_images = 100
    # camera_dict = np.load(cam_file_path)
    # scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    # world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(n_images)]
    # print(scale_mats)
    # print(world_mats)
    # input()

    # total_intrinsics_all = []
    # total_pose_all = []
    # for scale_mat, world_mat in zip(scale_mats, world_mats):
    #     P = world_mat @ scale_mat
    #     P = P[:3, :4]
    #     intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
    #     self.total_intrinsics_all.append(torch.from_numpy(intrinsics).float())
    #     self.total_pose_all.append(torch.from_numpy(pose).float())

    # test on a simple mesh
    # mesh = trimesh.load(root_dir + 'gt_mesh.ply', use_embree=True)
    # mesh = trimesh.load(root_dir + 'cat_gt_with_base_poisson_scale.ply', use_embree=True)
    # mesh = trimesh.load(root_dir + 'dog.obj', use_embree=True)
    mesh = trimesh.load(root_dir + 'surface_2080.ply', use_embree=True)
    # mesh = trimesh.load('../real_data/surface_120.ply', use_embree=True)
    print(mesh)

    # scene will have automatically generated camera and lights
    scene = mesh.scene()

    # all_camera_lines = []
    # for i in range(camera_poses.shape[0]):
    #     camera_lines = create_camera_actor(scale = 0.2)
    #     all_camera_lines.append(camera_lines.transform(camera_poses[i, :, :]))

    # vis_mesh = o3d.geometry.TriangleMesh()
    # vis_mesh.triangles = o3d.utility.Vector3iVector(np.array(mesh.vertices))
    # vis_mesh.vertices = o3d.utility.Vector3dVector(np.array(mesh.faces))

    # vis_pcd = o3d.geometry.PointCloud()
    # vis_pcd.points = o3d.utility.Vector3dVector(np.array(mesh.vertices))
          
    # all_camera_lines.append(vis_pcd)

    # o3d.visualization.draw_geometries(geometry_list=all_camera_lines, width = 640, height = 480, window_name = "mesh_with_hit_points")

    # any of the automatically generated values can be overridden
    # set resolution, in pixels
    # scene.camera.resolution = [1024, 1024]
    # set field of view, in degrees
    # make it relative to resolution so pixels per degree is same
    # scene.camera.fov = 60 * (scene.camera.resolution /
    #                          scene.camera.resolution.max())
    # scene.camera.focal = 2917.09632869, 2917.09632869#3889.03219785, 3889.03219785#2903.70082896, 2903.70082896
    image_paths = sorted(glob_imgs(root_dir + 'image/'))

    scene.camera.z_near = 0.001
    for path in tqdm.tqdm(image_paths):
    # for i in range(camera_poses.shape[0]):
        image_idx, _ = path.split('/')[-1].split('.')
        image_idx = int(image_idx)

        resolution = resolutions[image_idx]
        width, height = int(resolution[0]), int(resolution[1])
        K = Ks[image_idx, :, :] 

        camera_pose = camera_poses[image_idx, :, :].reshape(4, 4)
        origins, vectors = get_rays_np(height, 
                                       width, 
                                       K[0, 0], K[1, 1],
                                       K[0, 2], K[1, 2], 
                                       camera_pose)
        origins = origins.reshape(-1, 3)
        vectors = vectors.reshape(-1, 3)

        # do the actual ray- mesh queries
        points, index_ray, index_tri = mesh.ray.intersects_location(
            origins, vectors, multiple_hits=False)

        pixels = np.zeros(width * height, dtype=np.uint8)
        pixels[index_ray] = 255
        pixels = pixels.reshape(height, width)

        print(image_idx)
        # cv2.imwrite(root_dir + '/project_mask/%05d.png' % image_idx, pixels)
        cv2.imwrite(root_dir + '/our_rendered_mask/%05d.png' % image_idx, pixels)
        # create a PIL image from the depth queries
        # img = PIL.Image.fromarray(a)

        # show the resulting image
        # img.show()

    # print(pixels.shape)
        # print(pixels.shape)
        # pixel_ray = pixels[index_ray]

        # create a numpy array we can turn into an image
        # doing it with uint8 creates an `L` mode greyscale image

        # scale depth against range (0.0 - 1.0)
        # depth_float = ((depth - depth.min()) / depth.ptp())

        # # convert depth into 0 - 255 uint8
        # depth_int = 255#(depth_float * 255).round().astype(np.uint8)
        # # assign depth to correct pixel locations
        # render_image[pixel_ray[:, 0], pixel_ray[:, 1]] = depth_int
        # a = cv2.GaussianBlur(a, (5,5), cv2.BORDER_DEFAULT)
        # cv2.imshow('render_image', pixels)
        # cv2.waitKey(1)

    # create a raster render of the same scene using OpenGL
    # rendered = PIL.Image.open(trimesh.util.wrap_as_stream(scene.save_image()))