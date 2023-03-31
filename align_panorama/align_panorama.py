from __future__ import division

import PIL.Image

import trimesh
import numpy as np
import cv2
import trimesh
import torch
import open3d as o3d
import pytorch3d.io
from sklearn.preprocessing import normalize
from scipy.spatial.transform import Rotation as R

def sampleEnvLight(l, image, width, height):

    l = torch.clamp(l, -0.999999, 0.999999)
    # Compute theta and phi
    x, y, z = torch.split(l, [1, 1, 1], dim=1 )
    theta = torch.acos(y )
    phi = torch.atan2( x, z )
    v = theta / np.pi * (height-1)
    u = (-phi / np.pi / 2.0 + 0.5) * (width-1)

    # bilinear interpolation
    u, v = torch.flatten(u), torch.flatten(v)
    u1 = torch.clamp(torch.floor(u).detach(), 0, width-1)
    v1 = torch.clamp(torch.floor(v).detach(), 0, height-1)
    u2 = torch.clamp(torch.ceil(u).detach(), 0, width-1)
    v2 = torch.clamp(torch.ceil(v).detach(), 0, height-1)

    w_r = (u - u1).unsqueeze(1)
    w_l = (1 - w_r )
    w_u = (v2 - v).unsqueeze(1)
    w_d = (1 - w_u )

    u1, v1 = u1.long(), v1.long()
    u2, v2 = u2.long(), v2.long()
    num_pixel = width * height

    image = image.reshape(num_pixel, -1)
    index = (v1 * width + u2) 
    envmap_ru = torch.index_select(image, 0, index )
    index = (v2 * width + u2) 
    envmap_rd = torch.index_select(image, 0, index )
    index = (v1 * width + u1) 
    envmap_lu = torch.index_select(image, 0, index )
    index = (v2 * width + u1) 
    envmap_ld = torch.index_select(image, 0, index )

    envmap_r = envmap_ru * w_u.expand_as(envmap_ru ) + \
            envmap_rd * w_d.expand_as(envmap_rd )
    envmap_l = envmap_lu * w_u.expand_as(envmap_lu ) + \
            envmap_ld * w_d.expand_as(envmap_ld )
    rendered_image = envmap_r * w_r.expand_as(envmap_r ) + \
            envmap_l * w_l.expand_as(envmap_l )

    return rendered_image

def sampleImage(self, uvs, image, width, height):

    uvs = torch.clamp(uvs, 0.0, 1.0)
    # Compute theta and phi
    u = uvs[..., 0] * (width - 1)
    v = uvs[..., 1] * (height - 1)
    # Bilinear interpolation to get the new image

    # bilinear interpolation
    u, v = torch.flatten(u), torch.flatten(v)
    u1 = torch.clamp(torch.floor(u).detach(), 0, width-1)
    v1 = torch.clamp(torch.floor(v).detach(), 0, height-1)
    u2 = torch.clamp(torch.ceil(u).detach(), 0, width-1)
    v2 = torch.clamp(torch.ceil(v).detach(), 0, height-1)

    w_r = (u - u1).unsqueeze(1)
    w_l = (1 - w_r )
    w_u = (v2 - v).unsqueeze(1)
    w_d = (1 - w_u )

    u1, v1 = u1.long(), v1.long()
    u2, v2 = u2.long(), v2.long()
    num_pixel = width * height

    image = image.reshape(num_pixel, -1)
    index = (v1 * width + u2) 
    envmap_ru = torch.index_select(image, 0, index )
    index = (v2 * width + u2) 
    envmap_rd = torch.index_select(image, 0, index )
    index = (v1 * width + u1) 
    envmap_lu = torch.index_select(image, 0, index )
    index = (v2 * width + u1) 
    envmap_ld = torch.index_select(image, 0, index )

    envmap_r = envmap_ru * w_u.expand_as(envmap_ru ) + \
            envmap_rd * w_d.expand_as(envmap_rd )
    envmap_l = envmap_lu * w_u.expand_as(envmap_lu ) + \
            envmap_ld * w_d.expand_as(envmap_ld )
    rendered_image = envmap_r * w_r.expand_as(envmap_r ) + \
            envmap_l * w_l.expand_as(envmap_l )

    print('rendered_image')
    print(rendered_image.shape)

    return rendered_image

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

def panorama_uv_to_xyz(uv, width, height):

    u = uv[:, 0]
    v = uv[:, 1]
    theta = v / (height - 1) * np.pi
    phi = -(u / (width - 1) - 0.5) * 2.0 * np.pi
    y = np.cos(theta)
    x = np.sin(phi)
    z = np.cos(phi)

    rays = np.concatenate((x, y, z), axis=0)
    rays = rays.reshape(3, -1)
    return rays

     # l = torch.clamp(l, -0.999999, 0.999999)
    # Compute theta and phi
    # x, y, z = torch.split(l, [1, 1, 1], dim=1 )
    # theta = torch.acos(y )
    # phi = torch.atan2( x, z )
    # v = theta / np.pi * (height-1)
    # u = (-phi / np.pi / 2.0 + 0.5) * (width-1)

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

if __name__ == '__main__':

    panorama_image_name = 'insta360.jpg'
    panorama_log_name = panorama_image_name.replace("jpg", "log", 3)
    image_name = '0.jpg'
    image_log_name = image_name.replace("jpg", "log", 3)

    with open("./camera_params.log", 'r') as f:
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
            Ks.append([intrin[0], 0, intrin[2], 0, intrin[1], intrin[3], 0, 0, 1])
        if i % 7 == 2:
            resolution = list(map(float, line.strip().split(' ')))
            resolutions.append([resolution[0], resolution[1]])
        if i % 7 >= 3 and i % 7 <= 6:
            camera_poses.append(list(map(float, line.strip().split())))

    Ks = np.array(Ks)
    Ks = Ks.reshape(-1, 3, 3)
    resolutions = np.array(resolutions)
    resolutions = resolutions.reshape(-1, 2)
    camera_poses = np.array(camera_poses)
    camera_poses = camera_poses.reshape(-1, 4, 4)
    Ps = torch.eye(4).unsqueeze(0).repeat(camera_poses.shape[0], 1, 1).numpy()
    Ps[:,:3,:] = np.matmul(Ks, np.linalg.inv(camera_poses)[:,:3,:])

    image_marked_points = []
    with open(image_log_name, 'r') as f:
        lines = f.readlines()
    for line, i in zip(lines, np.arange(len(lines))):
        point = list(map(float, line.strip().split(' ')))
        image_marked_points.append([point[0], point[1]]) 
    image_marked_points = np.array(image_marked_points)
    image_marked_points = image_marked_points.reshape(-1, 2)
    image = cv2.imread(image_name)
    # for i in range(image_marked_points.shape[0]):
    #     image = cv2.circle(image, (int(image_marked_points[i, 0]), int(image_marked_points[i, 1])), radius=5, color=(0, 0, 240), thickness=5) 
    #     cv2.imshow('image', image)
    #     cv2.waitKey(0)

    panorama_marked_points = []
    with open(panorama_log_name, 'r') as f:
        lines = f.readlines()
    for line, i in zip(lines, np.arange(len(lines))):
        point = list(map(float, line.strip().split(' ')))
        panorama_marked_points.append([point[0], point[1]]) 
    panorama_marked_points = np.array(panorama_marked_points)
    panorama_marked_points = panorama_marked_points.reshape(-1, 2)
    panorama_image = cv2.imread(panorama_image_name)
    panorama_image = panorama_image.astype(np.float32) / 255.0
    # for i in range(panorama_marked_points.shape[0]):
    #     panorama_image = cv2.circle(panorama_image, (int(panorama_marked_points[i, 0]), int(panorama_marked_points[i, 1])), radius=5, color=(0, 0, 240), thickness=5) 
    #     vis_panorama_image = cv2.resize(panorama_image, (panorama_image.shape[1] // 4, panorama_image.shape[0] // 4), interpolation=cv2.INTER_LINEAR)      
    #     cv2.imshow('vis_panorama_image', vis_panorama_image)
    #     cv2.waitKey(0)

    image_marked_points = np.column_stack((image_marked_points, np.ones(image_marked_points.shape[0])))
    image_marked_points = np.transpose(image_marked_points, (1, 0))
    K = Ks[0]
    inv_K = np.linalg.inv(K)
    resolution = resolutions[0]
    camera_pose = camera_poses[0]
    image_rays = inv_K @ image_marked_points
    image_rays = camera_pose[:3, :3] @ image_rays
    image_rays = normalize(image_rays, axis=0, norm='l2')
    print('image_rays')
    print(image_rays.T)

    panorama_width, panorama_height = 6080, 3040
    panorama_rays = panorama_uv_to_xyz(panorama_marked_points, panorama_width, panorama_height)
    panorama_rays = normalize(panorama_rays, axis=0, norm='l2')
    print('panorama_rays')
    print(panorama_rays.T)

    estimated_rotation, rmsd = R.align_vectors(panorama_rays.T, image_rays.T)
    estimated_rotation = estimated_rotation.as_matrix()
    print('estimated_rotation')
    print(estimated_rotation)
    print('rmsd')
    print(rmsd)    

    ray_origins, ray_vectors = get_rays_np(resolution[1], resolution[0], \
        K[0, 0], K[1, 1], K[0, 2], K[1, 2], camera_pose)
    ray_vectors = normalize(ray_vectors.reshape(-1, 3), axis=1, norm='l2')
    
    ray_vectors = ray_vectors @ estimated_rotation.T
    ray_origins = torch.from_numpy(ray_origins.copy().reshape(-1, 3)).cuda()
    ray_vectors = torch.from_numpy(ray_vectors.copy().reshape(-1, 3)).cuda() 
    env_width = panorama_image.shape[1]
    env_height =  panorama_image.shape[0]
    panorama_image = np.ascontiguousarray(panorama_image)
    env_image = torch.from_numpy(panorama_image).float().cuda()

    # line_points_1 = ray_origins
    # line_points_1 = line_points_1[::100, None, :]
    # line_points_2 = ray_origins + ray_vectors
    # line_points_2 = line_points_2[::100, None, :]
    # line_points = torch.cat((line_points_1, line_points_2), dim=1)
    # print('line_points.shape')
    # print(line_points.shape)
    # line_points = line_points.reshape(-1, 3)
    # lines =  np.arange(0, line_points.shape[0] * 2).reshape(-1, 2)
    # line_set = o3d.geometry.LineSet()
    # line_set.points = o3d.utility.Vector3dVector(line_points.cpu().detach().numpy())
    # line_set.lines = o3d.utility.Vector2iVector(lines)
    # o3d.visualization.draw_geometries(geometry_list = [line_set], width = 640, height = 480, window_name = "rays")

    sample_image = sampleEnvLight(ray_vectors, env_image.detach(), env_width, env_height)
    sample_image = sample_image.detach().cpu().numpy().reshape(int(resolution[1]), int(resolution[0]), 3)
    vis_sample_image = cv2.resize(sample_image, (sample_image.shape[1] // 2, sample_image.shape[0] // 2), interpolation=cv2.INTER_LINEAR)      
    gt_image = cv2.imread(image_name)
    vis_gt_image = cv2.resize(gt_image, (gt_image.shape[1] // 2, gt_image.shape[0] // 2), interpolation=cv2.INTER_LINEAR)      
    print(vis_sample_image.shape)
    print(vis_gt_image.shape)
    vis_sample_image = vis_sample_image * 255
    vis_sample_image = vis_sample_image.astype(np.uint8)
    # cv2.imshow('vis_sample_image', vis_sample_image)
    cv2.imwrite('vis_sample_image.png', vis_sample_image)
    # cv2.imshow('vis_gt_image', vis_gt_image)
    cv2.imwrite('vis_gt_image.png', vis_gt_image)
    # cv2.waitKey(0)

   