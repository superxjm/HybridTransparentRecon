import potpourri3d as pp3d
import numpy as np
import torch
import trimesh
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib as mpl

data_dir = '../real_data_dog/vsa_150/'

if False:
    # mesh = trimesh.load(data_dir + 'mask_loss_mesh_vsa_no_color.obj', process=False)
    # vertices = np.array(mesh.vertices)
    # vertices = vertices.reshape(40, -1, 3)
    # print(vertices.shape)
    # pcd_list = []
    # for i in range(40):
    #     vis_pcd = o3d.geometry.PointCloud()
    #     vis_pcd.points = o3d.utility.Vector3dVector(vertices[i, :, :])
    #     rgb = (((i + 1) * np.array([1111, 811, 217])) % 255) / 255.0 
    #     vis_pcd.paint_uniform_color(rgb)
    #     pcd_list.append(vis_pcd)
    # o3d.visualization.draw_geometries(geometry_list = pcd_list, width = 640, height = 480, window_name = "vis_pcd", point_show_normal=False) 
     
    knn_indices = np.load(data_dir + 'geodesic_indices.npy')
    knn_weights = np.load(data_dir + 'geodesic_weights.npy')
    print(knn_indices.shape)
    mesh = trimesh.load(data_dir + 'mask_loss_mesh_vsa_no_color.obj', process=False)
    vertices = np.array(mesh.vertices)
    
    vis_pcd_all_points = o3d.geometry.PointCloud()
    vis_pcd_all_points.points = o3d.utility.Vector3dVector(vertices)
    vis_pcd_all_points.paint_uniform_color([0.5, 0.5, 0.5])  
    
    for src_idx in range(1000):
        knn_idx = knn_indices[src_idx, :]
        knn_vertices = vertices[knn_idx, :]
        # knn_vertices = vertices[0:1000, :]
        vis_pcd = o3d.geometry.PointCloud()
        vis_pcd.points = o3d.utility.Vector3dVector(knn_vertices)
        rgb = (((src_idx + 1) * np.array([1111, 811, 217])) % 255) / 255.0 
        vis_pcd.paint_uniform_color(rgb)  
        pcd_list = [vis_pcd_all_points]
        pcd_list.append(vis_pcd)
        o3d.visualization.draw_geometries(geometry_list = pcd_list, width = 640, height = 480, window_name = "vis_pcd", point_show_normal=False) 
     
if False:
    knn_indices = np.load(data_dir + 'geodesic_indices.npy')
    knn_weights = np.load(data_dir + 'geodesic_weights.npy')
    point_num = knn_weights.shape[0]
    print(knn_indices[0, :])
    print(knn_weights[0, :])
    # index_array = np.argmax(knn_weights, axis=1)
    # # print(index_array.shape)
    # # print(np.linspace(0, point_num-1, point_num, dtype=np.integer))
    # # print(knn_weights[0, :])
    # temp = knn_indices[np.linspace(0, point_num-1, point_num, dtype=np.integer), index_array]
    # # temp[(temp>0.049) & (temp<0.051)] = 1.0
    # print(temp[0:1000])

    # mesh = trimesh.load(data_dir + 'mask_loss_mesh_vsa_no_color.obj', process=False)
    # vertices = np.array(mesh.vertices)
    # print(vertices[0:1000, :])

    # print(tTrueTrue
    # print(temp.min())
    # print(temp.max())

def num2color(values, cmap):
    print(np.min(values))
    print(np.max(values))
    """将数值映射为颜色"""
    # norm = mpl.colors.Normalize(vmin=np.min(values), vmax=np.max(values)) 
    # cmap = mpl.cm.get_cmap(cmap)
    # return np.array([cmap(norm(val)) for val in values])
    values = values * 3
    # values = values - np.floor(values) 
    values = np.clip(values, 0, 1)
    cmap = mpl.cm.get_cmap(cmap)
    return np.array([cmap(val) for val in values]) 
    
VIS = False 
if VIS:
    V, F = pp3d.read_mesh(data_dir + 'mask_loss_mesh_vsa_no_color.obj') # Reads a mesh from file. Returns numpy matrices V, F
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
    print('start')
    knn_indices = []
    knn_weights = []
    neighbour_num = 40
    # V.shape[0]
    for i in range(V.shape[0]):
        dist = solver.compute_distance(i)
        dist[i] = 0
        color = (num2color(dist, "plasma")*256).astype(np.uint8)
        color = color[:, 0:3]
        print(color.shape)
        print(V.shape)
        print(color[:10])
        vis_geo_mesh=trimesh.Trimesh(vertices=V,vertex_colors=color,faces=F)
        vis_geo_mesh.export('../vis_geodesic.obj')
        exit()
        
if True:
    # = Stateful solves (much faster if computing distance many times)
    V, F = pp3d.read_mesh(data_dir + 'mask_loss_mesh_vsa_no_color.obj') # Reads a mesh from file. Returns numpy matrices V, F
    solver = pp3d.MeshHeatMethodDistanceSolver(V, F)
    print('start')
    knn_indices = []
    knn_weights = []
    neighbour_num = 40
    # V.shape[0]
    for i in range(V.shape[0]):
        dist = solver.compute_distance(i)
        dist[i] = 0
        # print(np.sort(dist))
        # print(np.exp(-np.sort(dist) / 0.003)[0:160])
        # input()

        indices = np.ones(neighbour_num, dtype=np.integer) * i
        dist_threshold = np.sort(dist)[neighbour_num]
        # Need to consider the padded vertices, so we use temp_indices
        temp_indices = np.where(dist < dist_threshold)[0]
        temp_indices = temp_indices[:neighbour_num]
        indices[0:temp_indices.shape[0]] = temp_indices
        knn_indices.append(indices)
        if neighbour_num == 20:
            weight = np.exp(-dist[indices] / 0.003)
        if neighbour_num == 40:
            weight = np.exp(-dist[indices] / 0.005)
        weight = weight / weight.sum()
        weight = weight[:neighbour_num]
        # print(weight)
        # input()
        if weight.shape[0] != neighbour_num:
            print(weight.shape)
            print(indices.shape)
            input()
        knn_weights.append(weight)
        if i % 100 == 0:
            print(i)
    knn_indices = np.stack(knn_indices)
    print(knn_indices.shape)
    knn_weights = np.stack(knn_weights)
    print(knn_weights.shape)
    np.save(data_dir + 'geodesic_indices_knn_40.npy', knn_indices)
    np.save(data_dir + 'geodesic_weights_knn_40.npy', knn_weights)
    exit()

# input()
# input()
# geodesic_dists = np.load(data_dir + 'geodesic_dists.npy')
# point_0_dists = geodesic_dists[0, :].copy()
# dist_threshold = np.sort(point_0_dists)[8]
# print('dist_threshold')
# print(dist_threshold)
# point_num = geodesic_dists.shape[0]
# oo = np.where(geodesic_dists < dist_threshold)
# vv = 0.2 - geodesic_dists[oo]
# geodesic_dists_sp = torch.sparse_coo_tensor(oo, vv, (point_num, point_num)).float()

# row_sum = geodesic_dists_sp.sum(-1, keepdim=True)  # sum by row(tgt)
#         # threshold on 1 to avoid div by 0
# torch.nn.functional.threshold(row_sum, 1, 1, inplace=True)
# geodesic_dists_sp.div_(row_sum)
# print(geodesic_dists_sp)
# print(geodesic_dists_sp.shape)
# torch.save(geodesic_dists_sp, data_dir + 'geodesic_dists_sp.pt')

# geodesic_dists_tensor = torch.load(data_dir + 'geodesic_dists_sp.pt')
# geodesic_dists_tensor.coalesce()
# print(geodesic_dists_tensor)
# input()
# input()
# input()
# input()
# input()

      # self.geodesic_dists = np.load(self.data_dir + 'geodesic_dists.npy')
        # # print('cluster_vertex_start_end')
        # # print(self.cluster_vertex_start_end)
        # # print('geodesic_dists.shape')
        # # print(self.geodesic_dists.shape)
        # point_0_dists = self.geodesic_dists[0, :].copy()
        # dist_threshold = np.sort(point_0_dists)[8]
        # # print('dist_threshold')
        # # print(dist_threshold)
        # # print(self.geodesic_dists[0])
        # point_num = self.geodesic_dists.shape[0]
        # # self.geodesic_dists = self.geodesic_dists.reshape(-1)
        # oo = np.where(self.geodesic_dists < dist_threshold)
        # vv = self.geodesic_dists[oo]
        # geodesic_dists_sp = torch.sparse_coo_tensor(oo, vv, (point_num, point_num))
        # # self.geodesic_dists[self.geodesic_dists > dist_threshold] = 0.0
        # # self.geodesic_dists = self.geodesic_dists.reshape(point_num, -1)
        # # # print(self.geodesic_dists[0])
        # # print(self.geodesic_dists.shape)

        # #  i = [[0, 1, 1],
        # #  [2, 0, 2]]
        # #     v =  [3, 4, 5]

        # # input()
        # # input()
        # # geodesic_dists_tensor = torch.from_numpy(self.geodesic_dists)
        # # self.geodesic_dists_sp = geodesic_dists_tensor.to_sparse()
        # print(self.geodesic_dists_sp)
        # print(self.geodesic_dists_sp.shape)
        # # print('cluster_neighbours')
        # # self.cluster_neighbours = self.cluster_neighbours.reshape((-1, 5))
        # # print(self.cluster_neighbours.shape)

        # self.geodesic_dists[self.geodesic_dists > dist_threshold] = 0.0
        # self.geodesic_dists = self.geodesic_dists.reshape(point_num, -1)
        # # print(self.geodesic_dists[0])
        # print(self.geodesic_dists.shape)

        #  i = [[0, 1, 1],
        #  [2, 0, 2]]
        #     v =  [3, 4, 5]

        # input()
        # input()
        # geodesic_dists_tensor = torch.from_numpy(self.geodesic_dists)
        # self.geodesic_dists_sp = geodesic_dists_tensor.to_sparse(

        # print('cluster_neighbours')
        # self.cluster_neighbours = self.cluster_neighbours.reshape((-1, 5))
        # print(self.cluster_neighbours.shape)