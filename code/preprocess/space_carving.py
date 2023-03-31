#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Matthieu Zins
"""

import scipy.io
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import tqdm
import open3d as o3d
import torch
from skimage import measure
import trimesh

def get_grid_uniform(resolution):
    x = np.linspace(-0.7, 0.7, resolution)
    y = x
    z = x

    xx, yy, zz = np.meshgrid(x, y, z)
    grid_points = torch.tensor(np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T, dtype=torch.float)

    return {"grid_points": grid_points,
            "shortest_axis_length": 2.0,
            "xyz": [x, y, z],
            "shortest_axis_index": 0}

if __name__ == '__main__':

    root_dir = '../DRT_data_horse/'

    # Load camera matrices
    projections = np.load(root_dir + "camera_Ps.npy")

    # load images
    # files = sorted(glob.glob(root_dir + "/project_mask/*.png"))
    files = sorted(glob.glob(root_dir + "/mask/*.png"))
    anno_mask_idx_list = []
    for f in tqdm.tqdm(files):
        image_idx, _ = f.split('/')[-1].split('.')
        image_idx = int(image_idx)
        anno_mask_idx_list.append(image_idx)
    print(anno_mask_idx_list)

    # files = sorted(glob.glob(root_dir + "/project_mask/*.png"))
    files = sorted(glob.glob(root_dir + "/mask/*.png"))
    silhouette = []
    for f in tqdm.tqdm(files):
        im = cv2.imread(f, cv2.IMREAD_GRAYSCALE).astype(float)
        im /= 255
        # cv2.imshow('silhouette', im)
        # cv2.waitKey(0)
        silhouette.append(im)

    resolution = 320
    grid = get_grid_uniform(resolution)
    points = grid['grid_points']
        
    # create voxel grid
    pts = grid['grid_points'].numpy()
    pts = np.vstack((pts.T, np.ones((1, pts.shape[0]))))

    filled = []
    count = 0
    projections = projections[::1]
    silhouette = silhouette[::1]
    for P, im in tqdm.tqdm(zip(projections, silhouette)):

        height, width = im.shape
        uvs = P @ pts
        uvs /= uvs[2, :]
        uvs = np.round(uvs).astype(int)
        x_good = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < width)
        y_good = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < height)
        good = np.logical_and(x_good, y_good)
        indices = np.where(good)[0]
        fill = np.zeros(uvs.shape[1])
        sub_uvs = uvs[:2, indices]
        res = im[sub_uvs[1, :], sub_uvs[0, :]]
        fill[indices] = res 
        filled.append(fill)
        # if count in anno_mask_idx_list:
        #     height, width = im.shape
        #     uvs = P @ pts
        #     uvs /= uvs[2, :]
        #     uvs = np.round(uvs).astype(int)
        #     x_good = np.logical_and(uvs[0, :] >= 0, uvs[0, :] < width)
        #     y_good = np.logical_and(uvs[1, :] >= 0, uvs[1, :] < height)
        #     good = np.logical_and(x_good, y_good)
        #     indices = np.where(good)[0]
        #     fill = np.zeros(uvs.shape[1])
        #     sub_uvs = uvs[:2, indices]
        #     res = im[sub_uvs[1, :], sub_uvs[0, :]]
        #     fill[indices] = res 
        #     filled.append(fill)
        count = count + 1
        
    filled = np.vstack(filled)
    # print(filled)
    # print(filled.shape)
    # input()

    # the occupancy is computed as the number of camera in which the point "seems" not empty
    occupancy = np.sort(filled, axis=0)[1, :]
    # print(occupancy)
    # print(occupancy.shape)

    # Select occupied voxels
    pts = pts.T
    good_points = pts[occupancy > 0, :]

    verts, faces, normals, values = measure.marching_cubes_lewiner(
                volume=occupancy.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                         grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=0.5,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1],
                         grid['xyz'][0][2] - grid['xyz'][0][1]))
    verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
    meshexport = trimesh.Trimesh(verts, faces, normals)
    # meshexport.export(root_dir + 'space_carving_mesh.ply')

    # t_offset = [ -12.48852921, -107.15911824,    1.06117058]
    # scale = 0.0037770700873581086
    # meshexport.vertices /= scale
    # meshexport.vertices -= t_offset

    meshexport.export(root_dir + 'DRT_shape_view72.ply')

    # vis_pcd = o3d.geometry.PointCloud()
    # vis_pcd.points = o3d.utility.Vector3dVector(good_points[:, :3])
    # o3d.visualization.draw_geometries(geometry_list = [vis_pcd], width = 640, height = 480, window_name = "pcd", point_show_normal=False)  
        
    # from skimage import measure
    # import trimesh
    #     grid = get_grid_uniform(resolution)
    #     points = grid['grid_points']

    #     z = []
    #     for i, pnts in enumerate(torch.split(points, 10000, dim=0)):
    #         z.append(sdf(pnts).detach().cpu().numpy())
    #     z = np.concatenate(z, axis=0)

    #     if (not (np.min(z) > 0 or np.max(z) < 0)):

    #         z = z.astype(np.float32)

    #         # torch.cuda.synchronize()
    #         # start = time.time()
    #         verts, faces, normals, values = measure.marching_cubes_lewiner(
    #             volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
    #                             grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
    #             level=0,
    #             spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
    #                     grid['xyz'][0][2] - grid['xyz'][0][1],
    #                     grid['xyz'][0][2] - grid['xyz'][0][1]))

    #         verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

# verts, faces, normals, values = measure.marching_cubes_lewiner(
#     volume=occupancy.reshape(s, s, s),
#     level=0.5,
#     spacing=(1.0, 1.0, 1.0))
# # verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])
# meshexport = trimesh.Trimesh(verts, faces, normals)
# meshexport.export('../space_carving_mesh.ply')

# #%% save point cloud with occupancy scalar 
# filename = "shape.txt"
# with open(filename, "w") as fout:
#     fout.write("x,y,z,occ\n")
#     for occ, p in zip(occupancy, pts[:, :3]):
#         fout.write(",".join(p.astype(str)) + "," + str(occ) + "\n")
    
# #%% save as rectilinear grid (this enables paraview to display its iso-volume as a mesh)
# import vtk

# xCoords = vtk.vtkFloatArray()
# x = pts[::s*s, 0]
# y = pts[:s*s:s, 1]
# z = pts[:s, 2]
# for i in x:
#     xCoords.InsertNextValue(i)
# yCoords = vtk.vtkFloatArray()
# for i in y:
#     yCoords.InsertNextValue(i)
# zCoords = vtk.vtkFloatArray()
# for i in z:
#     zCoords.InsertNextValue(i)
# values = vtk.vtkFloatArray()
# for i in occupancy:
#     values.InsertNextValue(i)
# rgrid = vtk.vtkRectilinearGrid()
# rgrid.SetDimensions(len(x), len(y), len(z))
# rgrid.SetXCoordinates(xCoords)
# rgrid.SetYCoordinates(yCoords)
# rgrid.SetZCoordinates(zCoords)
# rgrid.GetPointData().SetScalars(values)

# writer = vtk.vtkXMLRectilinearGridWriter()
# writer.SetFileName("shape.vtr")
# writer.SetInputData(rgrid)
# writer.Write()