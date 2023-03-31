# import the necessary packages
import apriltag
import argparse
import cv2
import os
from glob import glob
import numpy as np
import trimesh
import torch
import open3d as o3d
from sklearn.neighbors import KDTree
from sklearn.cluster import MeanShift
import tqdm

def glob_imgs(path):
    imgs = []
    for ext in ['*.png', '*.jpg', '*.JPEG', '*.JPG']:
        imgs.extend(sorted(glob(os.path.join(path, ext))))
    return imgs

def mkdir_ifnotexists(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)

def load_camera_params(path):
    with open(path, 'r') as f:
	    lines = f.readlines()
    Ks = []
    camera_poses = []
    resolutions = []
    lines = lines[1:]
    for line, i in zip(lines, np.arange(len(lines))):
        # if i % 7 == 0:
            # print("name: ", line)
        if i % 7 == 1:
            # print("intrin: ", line)
            intrin = list(map(float, line.strip().split(' ')))
            Ks.append([intrin[0], 0, intrin[2], 0, intrin[1], intrin[3], 0, 0, 1])
        if i % 7 == 2:
            resolution = list(map(float, line.strip().split()))
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

    return Ps, resolutions, camera_poses, Ks

def hex_to_rgb(value):
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) for i in range(0, lv, lv // 3))

def hex_colors_to_rgb_colors(value):
    rgb_colors = np.zeros((len(value), 3), dtype = np.int32)
    for i in range(len(value)):
        rgb_colors[i, :] = hex_to_rgb(value[i])
    return rgb_colors

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

def load_poly_points(path):

    with open(path, 'r') as f:
	    lines = f.readlines()

    line_count = 0
    poly = []
    while True:
        line = lines[line_count]
        line_count += 1
        point_num = int(line)
        points = []
        for j in range(point_num):
            line = lines[line_count]
            line_count += 1
            xy = list(map(float, line.strip().split(' ')))
            points.append(xy)
        poly.append(points)
        if line_count >= (len(lines) - 1):
            break
    
    return np.array(poly) 

def line_plane_collision(plane_normal, plane_point, ray_directions, ray_origins, epsilon=1e-6):
    
    ndotu = torch.bmm(plane_normal[:, None, :], ray_directions[:, :, None]).squeeze(-1)
    # if abs(ndotu) < epsilon:
    #     raise RuntimeError("no intersection or line is within plane")

    w = ray_origins - plane_point
    # si = -plane_normal.dot(w) / ndotu
    si = -torch.bmm(plane_normal[:, None, :], w[:, :, None]).squeeze(-1) / ndotu
    psi = w + si * ray_directions + plane_point

    return psi

def calc_mean_var(narray):

    N = narray.shape[0]
    sum1 = narray.sum()
    narray2 = narray*narray
    sum2 = narray2.sum()
    mean = sum1/N
    var = sum2/N-mean**2
    var = np.max(np.abs(narray - mean))
    return mean, var

def load_board_texture_image_list(path):

    with open(path, 'r') as f:
	    lines = f.readlines()

    board_texture_image_list = []
    for line in lines:
        board_texture_image_list.append(int(line))
    
    return board_texture_image_list

def load_image_board_map(path):
    
    with open(path, 'r') as f:
	    lines = f.readlines()

    image_board_map = []
    for line in lines:
        images = list(map(int, line.strip().split(' ')))
        image_board_map.append(images)
    
    return image_board_map

def xyz_to_uv(points_3d, P): 
    v = points_3d.T
    v = np.concatenate((v, np.ones((1, v.shape[1]))), axis=0)
    v = np.matmul(P, v)
    v[0:2, :] = v[0:2, :] / v[2, :] 
    v = v[0:2, :].T
    return v

def calc_board_coord(dir, board_3d_idx, board_image_idx, 
                     Ks, resolutions, camera_poses, 
                     enable_debug):

    plane_mesh = trimesh.load(dir + '/board_plane_' + str(board_3d_idx) + '.ply', use_embree=True)
    v1v0 = plane_mesh.vertices[1] - plane_mesh.vertices[0]
    v2v0 = plane_mesh.vertices[2] - plane_mesh.vertices[0]
    plane_normal = np.cross(v1v0, v2v0)
    plane_normal = plane_normal / (np.linalg.norm(plane_normal, axis=0) + 1e-16)
    plane_point = plane_mesh.vertices[0]
    print("plane_point: {0}".format(plane_point))
    print("plane_normal: {0}".format(plane_normal))
    
    board_ouv = load_poly_points(dir + '/env_matting/' + 'ouv_' + str(board_3d_idx) + '.log')
    plane_o = board_ouv[0, 0, :] 
    plane_u = board_ouv[0, 1, :] 
    plane_v = board_ouv[0, 2, :] 
    plane_ou = plane_u - plane_o
    plane_ov = plane_v - plane_o
    print("plane ou ov: {0}, {1}, {2}".format(plane_o, plane_ou, plane_ov))

    resolution = resolutions[board_image_idx]
    width, height = int(resolution[0]), int(resolution[1])
    K = Ks[board_image_idx, :, :] 
    camera_pose = camera_poses[board_image_idx, :, :].reshape(4, 4)
    origins, directions = get_rays_np(height, 
                                      width, 
                                      K[0, 0], K[1, 1],
                                      K[0, 2], K[1, 2], 
                                      camera_pose)
    board_ouv = board_ouv.reshape(-1, 2).astype(np.int32)
    origins = origins[board_ouv[:, 1], board_ouv[:, 0]]
    directions = directions[board_ouv[:, 1], board_ouv[:, 0]]
    print("origins: {0}".format(origins))
    print("directions: {0}".format(directions))
    
    plane_normal = np.broadcast_to(plane_normal, np.shape(directions))
    plane_point = np.broadcast_to(plane_point, np.shape(directions))
    hit_points = line_plane_collision(torch.from_numpy(plane_normal), 
                                      torch.from_numpy(plane_point), 
                                      torch.from_numpy(directions), 
                                      torch.from_numpy(origins))
    board_ouv_3d = hit_points.detach().numpy()
    print("board_ouv_3d: {0}".format(board_ouv_3d))

    sample_resolution = 1000
    xx, yy = np.meshgrid(np.arange(sample_resolution, dtype=np.float32),
                         np.arange(sample_resolution, dtype=np.float32), indexing='xy')
    xx = xx / (sample_resolution - 1) 
    yy = yy / (sample_resolution - 1) 
    board_o_3d, board_u_3d, board_v_3d = board_ouv_3d[0, :], board_ouv_3d[1, :], board_ouv_3d[2, :]
    board_ou_3d = (board_u_3d - board_o_3d)
    board_ov_3d = (board_v_3d - board_o_3d)
    sample_points_3d = board_o_3d[np.newaxis, np.newaxis, :] + \
        board_ou_3d[np.newaxis, np.newaxis, :] * xx[:, :, np.newaxis]  + \
        board_ov_3d[np.newaxis, np.newaxis, :] * yy[:, :, np.newaxis] 
    sample_points_3d = sample_points_3d.reshape(-1, 3)

    if enable_debug:
        o3d_vis_objs = []
        for i in range(board_ouv_3d.shape[0]):
            coordinate= o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=board_ouv_3d[i])
            o3d_vis_objs.append(coordinate)
    
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(sample_points_3d)
        # vis_pcd.colors = o3d.utility.Vector3dVector(sample_colors)
        o3d_vis_objs.append(vis_pcd)
        o3d.visualization.draw_geometries(geometry_list = o3d_vis_objs, width = 640, height = 480, window_name = "mesh_with_hit_points", point_show_normal = False) 

    return board_ouv_3d, sample_points_3d, plane_normal, plane_point

def color_space_cluster(dir, board_idx, texture_image_idx, 
                        Ks, resolutions, camera_poses,
                        board_ouv_3d, sample_points_3d, 
                        enable_debug):

    print(dir + '/env_matting/' + str(texture_image_idx) + '.jpg')
    texture_image = cv2.imread(dir + '/env_matting/' + str(texture_image_idx) + '.jpg').astype(np.float32)[:, :, :]
    # texture_image = cv2.GaussianBlur(texture_image, (5,5), cv2.BORDER_DEFAULT)
    texture_image = cv2.bilateralFilter(texture_image, 9, 60, 60)
    texture_image = texture_image / 255.0
    original_texture_image = texture_image.copy()
    # cv2.imshow('texture_image', texture_image)
    # cv2.waitKey(0)

    P = Ps[texture_image_idx, :3, :]

    corner_3d_0 = board_ouv_3d[0, :]
    corner_3d_1 = board_ouv_3d[1, :]
    corner_3d_2 = board_ouv_3d[2, :]
    corner_3d_3 = corner_3d_1 + corner_3d_2 - corner_3d_0 
    board_points_3d = np.concatenate((corner_3d_0, corner_3d_1, corner_3d_2, corner_3d_3), axis=0).reshape(-1, 3)
    board_points_uv = xyz_to_uv(board_points_3d, P)
    # print(board_points_uv)
    # exit()

    pts_src = board_points_uv
    pts_dst = np.array([[0, 0],[1024, 0],[0, 1024],[1024, 1024]])
    # Calculate Homography
    h, status = cv2.findHomography(pts_src, pts_dst)
    # Warp source image to destination based on homography
    texture_image = cv2.warpPerspective(texture_image, h, (1024, 1024))
    board_points_uv = np.array([[0, 0],[1024, 0],[0, 1024],[1024, 1024]]).astype(np.float64) 
    # cv2.imshow('homo_texture_image', homo_texture_image)
    # cv2.waitKey(0)

    local_texture_image_path = 'board_' + str(board_idx) + '.png'
    cv2.imwrite(dir + '/env_matting/' + local_texture_image_path, (texture_image * 255.0).astype(np.uint8))
    export_board_as_obj(dir + '/env_matting/', board_idx, board_points_3d, board_points_uv, texture_image.shape[1], texture_image.shape[0], local_texture_image_path)

    sample_resolution = 1024
    xx, yy = np.meshgrid(np.arange(sample_resolution, dtype=np.float32),
                         np.arange(sample_resolution, dtype=np.float32), indexing='xy')
    sample_points_uv = np.vstack([xx.ravel(), yy.ravel()]).T
    # sample_points_uv = xyz_to_uv(sample_points_3d, P)
    uv = sample_points_uv.astype(np.int64)
    sample_colors_bgr = texture_image[uv[:, 1], uv[:, 0], :]

    # sample_colors = texture_image[uv[:, 1], uv[:, 0], :]
    # sample_colors_bgr = sample_colors_bgr.reshape(-1, 3)
    # sample_colors = sample_colors.reshape(-1, 3)
    # vis_pcd = o3d.geometry.PointCloud()
    # vis_pcd.points = o3d.utility.Vector3dVector(sample_points)
    # vis_pcd.colors = o3d.utility.Vector3dVector(sample_colors)
    # o3d_vis_objs.append(vis_pcd)
    # o3d.visualization.draw_geometries(geometry_list = o3d_vis_objs, width = 640, height = 480, window_name = "mesh_with_hit_points", point_show_normal=False) 

    base_color_points = load_poly_points(dir + '/env_matting/' + str(texture_image_idx) + '.log')
    base_colors_bgr = []
    for i in range(base_color_points.shape[1]):
        u = int(base_color_points[0, i, 0])
        v = int(base_color_points[0, i, 1])
        color = 0.25 * (original_texture_image[v, u, :] + 
                        original_texture_image[v + 1, u, :] +
                        original_texture_image[v, u + 1, :] +
                        original_texture_image[v + 1, u + 1, :])
        base_colors_bgr.append(color)
    base_colors_bgr = np.array(base_colors_bgr)
    # print(base_colors_bgr.shape[0])
    # input()

    K = 1
    kdt = KDTree(base_colors_bgr, leaf_size=40, metric='euclidean')
    dist, nn_idx = kdt.query(sample_colors_bgr, k=K, return_distance=True)
    dist = dist[:, 0]
    nn_idx = nn_idx[:, 0]
    threshold = 0.04
    diff = np.max(sample_colors_bgr, axis=1) - np.min(sample_colors_bgr, axis=1)
    valid_mask = (dist < threshold)
    sample_colors_bgr[~valid_mask] = np.array([0.3, 0.3, 0.3])

    # sample_colors_bgr = sample_colors_bgr.reshape(1024, 1024, 3)
    # cv2.imshow('sample_colors_bgr', sample_colors_bgr)
    # cv2.waitKey(0)
    # exit()

    sample_points_uv = sample_points_uv[valid_mask, :]
    nn_idx = nn_idx[valid_mask] 
    gaussians = []
    image_width = float(texture_image.shape[1])
    image_height = float(texture_image.shape[0])
    for i in range(base_colors_bgr.shape[0]):
        # print(i)
        mean_u, var_u = calc_mean_var(sample_points_uv[nn_idx == i, 0])
        mean_v, var_v = calc_mean_var(sample_points_uv[nn_idx == i, 1])
        gaussians.append([mean_u / (image_width - 1), mean_v / (image_height - 1), (var_u + var_v) / (image_width + image_height)])

        radius = int(0.5 * (var_u + var_v))
        texture_image = cv2.circle(texture_image, 
                                   (int(mean_u), int(mean_v)), 
                                   radius, (1.0, 0, 0), 8) 
    gaussians = np.array(gaussians)

    cv2.imwrite(dir + '/env_matting/' + str(texture_image_idx) + '_clusters.jpg', (texture_image * 255.0).astype(np.uint8))
    color_clusters = {'base_colors_bgr': base_colors_bgr, \
                      'base_colors_gaussians': gaussians}
    for key in color_clusters:
        print('------------------------------')
        print('{0}:\n {1}'.format(key, color_clusters[key]))

    # if enable_debug:
    #     vis_pcd = o3d.geometry.PointCloud()
    #     vis_pcd.points = o3d.utility.Vector3dVector(sample_points_3d)
    #     vis_pcd.colors = o3d.utility.Vector3dVector(sample_colors)
    #     o3d.visualization.draw_geometries(geometry_list = [vis_pcd], width = 640, height = 480, window_name = "mesh_with_hit_points", point_show_normal = False) 

    return color_clusters

def export_board_as_obj(dir, board_idx, board_points_3d, board_points_uv, texture_width, texture_height, texture_image_path):

    fo = open(dir + 'board_' + str(board_idx) + '.mtl', 'w')
    fo.write("newmtl material_0\n")
    fo.write("Ka 1 1 1 \nKd 1 1 1 \nd 1 \nNs 0 \nillum 1\n")
    fo.write("newmtl material_0_u_v\n")
    fo.write("Ka 1 1 1 \nKd 1 1 1 \nd 1 \nNs 0 \nillum 1\n")
    fo.write("map_Kd " + texture_image_path + "\n")
    fo.close()

    normal = np.cross(board_points_3d[1, :] - board_points_3d[0, :],
        board_points_3d[2, :] - board_points_3d[0, :])
    
    board_points_uv[:, 0] /= texture_width
    board_points_uv[:, 1] /= texture_height
    board_points_uv[:, 1] = 1.0 - board_points_uv[:, 1]

    fo = open(dir + 'board_' + str(board_idx) + '.obj', 'w')
    fo.write('g default')
    fo.write('mtllib ' + 'board_' + str(board_idx) + '.mtl\n')
    fo.write('usemtl material_0\n')
    for i in range(4):
        fo.write('v ' + str(board_points_3d[i, :])[1:-1] + '\n')
        fo.write('vt ' + str(board_points_uv[i, :])[1:-1] + '\n')
        fo.write('vn ' + str(normal)[1:-1] + '\n')
    # f Vertex1/Texture1/Normal1 Vertex2/Texture2/Normal2 Vertex3/Texture3/Normal3
    fo.write('g default')
    fo.write('usemtl material_0_u_v\n')
    fo.write('f 1/1/1 2/2/2 3/3/3 \n')
    fo.write('f 2/2/2 4/4/4 3/3/3 \n')
  
    fo.close()

if __name__ == '__main__':

    enable_debug = False

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dir", required=True,
        help="input folder")
    ap.add_argument("--debug", action='store_true',
         default=False, help="enable debug mode")
    args = vars(ap.parse_args())

    root_dir = args["dir"]
    enable_debug = args["debug"]

    board_texture_image_list = load_board_texture_image_list(root_dir + '/env_matting/board_texture_image_list.txt')
    image_board_map = load_image_board_map(root_dir + '/env_matting/image_board_map.txt')
    board_3d_to_2d_map = load_image_board_map(root_dir + '/env_matting/board_3d_to_2d_map.txt')
    # print('board_texture_image_list')
    # print(board_texture_image_list)
    # print('image_board_map')
    # print(image_board_map)
    # print('board_3d_to_2d_map')
    # print(board_3d_to_2d_map)
    # exit()

    Ps, resolutions, camera_poses, Ks = load_camera_params(root_dir + '/camera_params.log')
    color_clusters_list = []
    for board_3d_idx in range(len(board_3d_to_2d_map)):
        board_image_idx = board_texture_image_list[board_3d_to_2d_map[board_3d_idx][0]] 
        board_ouv_3d, sample_points_3d, \
            plane_normal, plane_point = calc_board_coord(root_dir, board_3d_idx, board_image_idx, \
                                                        Ks, resolutions, camera_poses, \
                                                        enable_debug)
    
        for board_idx in board_3d_to_2d_map[board_3d_idx]:
            texture_image_idx = board_texture_image_list[board_idx]

            color_clusters = color_space_cluster(root_dir, board_idx, texture_image_idx, \
                                                Ks, resolutions, camera_poses, \
                                                board_ouv_3d, sample_points_3d, \
                                                enable_debug)
            color_clusters_list.append(color_clusters)

    image_board_mapping = []
    pixel_cluster_mappings = {} 
    for image_list, board_idx in zip(image_board_map, range(len(image_board_map))):

        color_clusters = color_clusters_list[board_idx] 
        base_colors_bgr = color_clusters['base_colors_bgr']
        base_colors_gaussians = color_clusters['base_colors_gaussians']
        all_base_colors_bgr = base_colors_bgr.copy()
        base_colors_bgr = base_colors_bgr[2:, :]
        base_colors_gaussians = base_colors_gaussians[2:, :]
        # print('base_colors_bgr:')
        # print(base_colors_bgr.shape)                                 
        # vis_base_colors_bgr = np.broadcast_to(base_colors_bgr[:, np.newaxis, :], (5, 100, 3))
        # cv2.imwrite(root_dir + '/scaled_src_image.png', (vis_base_colors_bgr * 255.0).astype(np.uint8))
        # exit()
        for image_idx in image_list:
            
            print(image_idx)

            src_image = cv2.imread(root_dir + '/env_matting/' + str(image_idx) + '.jpg')
            src_image = cv2.bilateralFilter(src_image, 5, 20, 20)
            src_image = src_image.astype(np.float32) / 255.0

            # gamma = 1.7
            # src_image = np.power(src_image, gamma)
            # src_image = src_image / 1.4
            # cv2.imwrite(root_dir + '/scaled_src_image.png', (src_image * 255.0).astype(np.uint8))
            # exit()
            # cv2.imshow('src_image', src_image)
            # cv2.waitKey(0)
            
            src_image_pixels = src_image.reshape(-1, 3)
            diff = np.max(src_image_pixels, axis=1) - np.min(src_image_pixels, axis=1)

            K = 1
            kdt = KDTree(base_colors_bgr, leaf_size=40, metric='euclidean')
            dist, nn_idx = kdt.query(src_image_pixels, k=K, return_distance=True)
            dist = dist[:, 0]
            nn_idx = nn_idx[:, 0]
            vis_src_image_pixels = src_image_pixels.copy()
            thres_dist = 0.3
            thres_diff = 0.2
            corr_valid_mask = (dist < thres_dist) & (diff > thres_diff) 

            all_kdt = KDTree(all_base_colors_bgr, leaf_size=40, metric='euclidean')
            all_dist, all_nn_idx = all_kdt.query(src_image_pixels, k=K, return_distance=True)
            all_dist = all_dist[:, 0]
            all_nn_idx = all_nn_idx[:, 0]
            thres_dist = 0.4
            outside_mask = ~(all_dist < thres_dist)

            vis_src_image_pixels[~corr_valid_mask] = np.array([0.0, 0.0, 0.0])
            vis_src_image_pixels[outside_mask] = np.array([0.0, 0.5, 0.0])
            vis_src_image = vis_src_image_pixels.reshape(src_image.shape)
            # mkdir_ifnotexists(root_dir + '/env_matting/output/')
            mkdir_ifnotexists(root_dir + '/corr/')
            mkdir_ifnotexists(root_dir + '/corr/result/')

            # src_pixel_idx = np.arange(src_image_pixels.shape[0])[valid_mask]
            # target_cluster_idx = nn_idx[valid_mask]
            # base_colors_gaussians = color_clusters['base_colors_gaussians']
            # target_gaussian_cluster = base_colors_gaussians[target_cluster_idx, :]
            # src_pixel_idx = np.stack((src_pixel_idx % src_image.shape[0], src_pixel_idx // src_image.shape[0]), axis=1)
            # pixel_cluster_mapping = np.concatenate((src_pixel_idx, target_gaussian_cluster), axis=1)
            # pixel_cluster_mappings[str(image_idx)] = pixel_cluster_mapping 

            target_cluster_idx = nn_idx
            target_gaussian_cluster = base_colors_gaussians[target_cluster_idx, :]
            target_gaussian_cluster[~corr_valid_mask] = np.array([np.nan, np.nan, np.nan])
            target_gaussian_cluster[outside_mask] = np.array([np.inf, np.inf, np.inf])
            target_gaussian_cluster = target_gaussian_cluster.reshape(src_image.shape).astype(np.float32)
            
            cv2.imwrite(root_dir + '/corr/result/%05d_result.png' % image_idx, (vis_src_image * 255.0).astype(np.uint8))
            cv2.imwrite(root_dir + '/corr/%05d.exr' % image_idx, target_gaussian_cluster)

            image_board_mapping.append([image_idx, board_idx])
    
    image_board_mapping = np.array(image_board_mapping)
    print('save image_board_mapping.npy')
    np.save(root_dir + '/corr/image_board_mapping.npy', image_board_mapping)
    # print('save pixel_cluster_mappings.npz')
    # np.savez(root_dir + 'env_matting/pixel_cluster_mappings', **pixel_cluster_mappings)
    exit()

#   output = {'base_colors_bgr': base_colors_bgr, \
    #           'base_colors_gaussians': gaussians, \
    #           'board_plane_normal': plane_normal,
    #           'board_plane_point': plane_point,
    #           'board_plane_ouv': board_ouv}
    # for key in output:
    #     print('------------------------------')
    #     print('{0}:\n {1}'.format(key, output[key]))
    # np.savez(root_dir + '/color_clusters', **color_clusters)
    # base_colors_bgr = color_clusters['base_colors_bgr']

    # for src_image_idx in range(resolutions.shape[0]):
    #     src_image = cv2.imread(root_dir + '/images/' + str(src_image_idx) + '.jpg').astype(np.float32)
    #     src_image = cv2. cv2.bilateralFilter(src_image, 9, 80, 80)
    #     src_image_hsv = src_image#cv2.cvtColor(src_image, cv2.COLOR_BGR2HSV_FULL)
    #     src_image = src_image / 255.0
    #     src_image_hsv = src_image_hsv / 255.0
    #     # cv2.imshow('src_image', src_image)
    #     # cv2.waitKey(0)
    #     # print(src_image_hsv.shape)
    #     # print(base_colors_bgr)
    #     # exit()

    #     src_image_pixels = src_image.reshape(-1, 3)
    #     src_image_hsv_pixels = src_image_hsv.reshape(-1, 3)
    #     K = 1
    #     kdt = KDTree(base_colors_bgr, leaf_size=40, metric='euclidean')
    #     dist, nn_idx = kdt.query(src_image_hsv_pixels, k=K, return_distance=True)
    #     dist = dist[:, 0]
    #     nn_idx = nn_idx[:, 0]
    #     src_image_pixels[dist > 0.4, :] = np.array([0.3, 0.3, 0.3])
    #     src_image = src_image_pixels.reshape(src_image.shape)
    #     cv2.imwrite(root_dir + '/env_matting/output/' + str(src_image_idx) + '_result.png', (src_image * 255.0).astype(np.uint8))

    # src_image_pixels = src_image.reshape(-1, 3)
    # src_image_pixels[dist > 0.5, :] = np.array([0, 0, 1])
    # src_image = src_image_pixels.reshape(src_image.shape)
    # cv2.imwrite('../real_data/env_matting/' + str(i) + '_result.png', (src_image * 255.0).astype(np.uint8))

    # for i in range(resolutions.shape[0]):
    #     print("image: {0}".format(i))
    #     src_image = cv2.imread('../real_data/env_matting/' + str(i) + '.jpg').astype(np.float32)
    #     src_image = src_image / 255.0
    #     src_image = cv2.GaussianBlur(src_image, (5,5), cv2.BORDER_DEFAULT) 
    #     src_image_pixels = src_image.reshape(-1, 3)
    #     # # src_image_pixels = src_image_pixels / (np.linalg.norm(src_image_pixels, axis=1)[:, np.newaxis] + 1e-16)

    #     resolution = resolutions[i]
    #     width, height = int(resolution[0]), int(resolution[1])
    #     K = Ks[i, :, :] 
    #     camera_pose = camera_poses[i, :, :].reshape(4, 4)
    #     origins, vectors = get_rays_np(height, 
    #                                     width, 
    #                                     K[0, 0], K[1, 1],
    #                                     K[0, 2], K[1, 2], 
    #                                     camera_pose)
    #     origins = origins.reshape(-1, 3)
    #     vectors = vectors.reshape(-1, 3)
    #     points, index_ray, index_tri = background_mesh.ray.intersects_location(
    #         origins, vectors, multiple_hits=False)
    #     # print(points.shape)
    #     v = points.T
    #     v = np.concatenate((v, np.ones((1, v.shape[1]))), axis=0)
    #     v = np.matmul(P, v)
    #     v[0:2, :] = v[0:2, :] / v[2, :] 
    #     v = v[0:2, :].T.astype(np.int64)
    #     background_vertex_colors = texture_image[v[:, 1], v[:, 0], :]
    #     background_pixels = np.zeros((width * height, 3), dtype=np.float32)
    #     background_pixels[index_ray, :] = background_vertex_colors 
    #     background_pixels = background_pixels.reshape(height, width, 3)
    #     cv2.imwrite('../real_data/env_matting/' + str(i) + '_background.png', (background_pixels * 255.0).astype(np.uint8))
    #     # diff_image = np.abs(background_pixels - src_image) 
    #     # cv2.imwrite('../real_data/' + str(i) + '_diff.png', (diff_image * 255.0).astype(np.uint8))
        
    #     K = 1
    #     kdt = KDTree(vertex_colors, leaf_size=40, metric='euclidean')
    #     dist, ind = kdt.query(src_image_pixels, k=K, return_distance=True)
    #     dist = dist[:, 0]
    #     ind = ind[:, 0]
    #     src_image_pixels = src_image.reshape(-1, 3)
    #     src_image_pixels[dist > 0.5, :] = np.array([0, 0, 1])
    #     src_image = src_image_pixels.reshape(src_image.shape)
    #     cv2.imwrite('../real_data/env_matting/' + str(i) + '_result.png', (src_image * 255.0).astype(np.uint8))

    #     corr_vertices_image = mesh_vertices[ind, :]
    #     corr_vertices_image[dist > 0.5, :] = np.NaN
    #     corr_vertices_image = corr_vertices_image.reshape(src_image.shape)
    #     cv2.imwrite('../real_data/env_matting/' + str(i) + '_corr.exr', corr_vertices_image.astype(np.float32))
        
        # ---------------------------
        # background_pixels = cv2.resize(background_pixels, (background_pixels.shape[1] // 4, background_pixels.shape[0] // 4)) 
        # src_image = cv2.resize(src_image, (src_image.shape[1] // 4, src_image.shape[0] // 4)) 
        # # print(background_pixels.shape)
        # # print(src_image.shape)
        # # exit()
        # background_pixels = background_pixels.reshape(-1, 3)
        # image_kdt = KDTree(background_pixels, leaf_size=40, metric='euclidean')
        # src_image_pixels = src_image.reshape(-1, 3)
        # dist, ind = image_kdt.query(src_image_pixels, k=K, return_distance=True)
        # ind = ind[:, 0]
        # src_idx = np.arange(len(ind))
        # src_pixel_coord = np.stack((src_idx // src_image.shape[1], src_idx % src_image.shape[1]), axis=1)
        # print('src_pixel_coord')
        # print(src_pixel_coord)
        # target_idx = ind
        # target_pixel_coord = np.stack((target_idx // src_image.shape[1], target_idx % src_image.shape[1]), axis=1)
        # print('target_pixel_coord')
        # print(target_pixel_coord)
        # pixel_dist = np.linalg.norm(src_pixel_coord - target_pixel_coord, axis=1)
        # src_image_pixels = src_image.reshape(-1, 3)
        # src_image_pixels[pixel_dist > 100, :] = np.array([0, 0, 1])
        # src_image = src_image_pixels.reshape(src_image.shape)
        # cv2.imwrite('../real_data/env_matting/' + str(i) + '_matting.png', (src_image * 255.0).astype(np.uint8))

        # exit()

      
   