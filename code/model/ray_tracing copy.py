import torch
import torch.nn as nn
import os
from utils import rend_util
import utils.plots as plt
import numpy as np
import open3d as o3d
from torch.autograd import Variable
import torchvision
import kornia
import time
from model.optix_intersect import Scene
from model.optix_intersect import Ray

import trimesh
from torchviz import make_dot
import pytorch3d.io

import cv2
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2,40).__str__()

class VGG(torch.nn.Module):
    def __init__(self, resize=False):
        super(VGG, self).__init__()
        blocks = []
        blocks.append(torchvision.models.vgg16(pretrained=True).features[:4].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[4:9].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[9:16].eval())
        blocks.append(torchvision.models.vgg16(pretrained=True).features[16:23].eval())
        for bl in blocks:
            for p in bl:
                p.requires_grad = False
        self.blocks = torch.nn.ModuleList(blocks)
        self.transform = torch.nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, input, feature_layers=[1], style_layers=[]):
        print("input.shape")
        print(input.shape)
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
        x = input
        y = None
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in feature_layers:
                y = x
        return y

def load_mesh_and_texture(mesh_path, texture_path, ksize=3, sigma=1.5, down_scale=1):
    
    ksize = ksize // down_scale
    if ksize % 2 == 0:
        ksize = ksize + 1
    sigma = sigma / float(down_scale)

    mesh = trimesh.load(mesh_path, use_embree=True, process=False)
    verts, faces, aux = pytorch3d.io.load_obj(mesh_path)
    texture_uvs = aux.verts_uvs.cuda()  # (V, 2)
    texture_uvs[:, 1] = 1.0 - texture_uvs[:, 1] # flip v

    print(f'texture_path: {texture_path}')
    temp_texture = cv2.imread(texture_path)[:, :, ::-1]
    print(f'finish load texture')
    texture_width = temp_texture.shape[1]
    texture_height = temp_texture.shape[0]
    if down_scale != 1:
        temp_texture = cv2.resize(temp_texture, (texture_width // down_scale, texture_height // down_scale), interpolation=cv2.INTER_LINEAR)
    temp_texture = temp_texture.astype(np.float32) / 255.0
    temp_texture = np.ascontiguousarray(temp_texture)
    temp_texture = cv2.GaussianBlur(temp_texture, (ksize, ksize), sigma, cv2.BORDER_DEFAULT)
    texture = torch.from_numpy(temp_texture).float().cuda()

    return mesh, texture_uvs, texture

class RayTracing(nn.Module):
    def __init__(
            self,
            object_bounding_sphere=1.0,
            sdf_threshold=5.0e-5,
            line_search_step=0.5,
            line_step_iters=1,
            sphere_tracing_iters=10,
            n_steps=100,
            board_num=4,
            n_secant_steps=8,
            data_dir="",
            mean_rgb=[],
            std_rgb=[]
    ):
        super().__init__()

        print('RayTracing init')

        self.object_bounding_sphere = object_bounding_sphere
        self.sdf_threshold = sdf_threshold
        self.sphere_tracing_iters = sphere_tracing_iters
        self.line_step_iters = line_step_iters
        self.line_search_step = line_search_step
        self.n_steps = n_steps
        self.n_secant_steps = n_secant_steps

        # get_vgg_image = VGG().cuda()

        self.eta1 = 1.0003
        self.eta2 = 1.52

        self.with_env_map = False 
        self.with_reflect = False
        self.with_global_scene = False
        self.with_board_scene = False 
        if board_num > 0:
            self.with_global_scene = True
            self.with_board_scene = True 

        if self.with_env_map:
            env_file_path = "../real_data_ball/insta360.jpg" 
            temp_env_image = cv2.imread(env_file_path)[:, :, ::-1]
            temp_env_image = temp_env_image.astype(np.float32)
            temp_env_image = temp_env_image / 255.0
            self.env_width, self.env_height = temp_env_image.shape[1], temp_env_image.shape[0]
            temp_env_image = np.ascontiguousarray(temp_env_image)
            temp_env_image = cv2.GaussianBlur(temp_env_image, (3,3), cv2.BORDER_DEFAULT)
            self.env_image = torch.from_numpy(temp_env_image).float().cuda()

        self.env_map_mesh = trimesh.load('../env_map_mesh.obj', use_embree=True, process=False)

        print('start load mesh and texture')
        if self.with_global_scene:
            self.scene_mesh, self.scene_texture_uvs, self.scene_texture = load_mesh_and_texture(data_dir + '/scaled_scene_mesh.obj', 
                                                                                                data_dir + '/global_mesh_u1_v1.png',
                                                                                                31, 15.0)

        self.board_mesh_list = []
        self.board_texture_uvs_list = []
        self.board_texture_list = []
        for board_idx in range((board_num * 15)):
            print('load board_{0}'.format(board_idx))
            board_mesh, board_texture_uvs, board_texture = load_mesh_and_texture(data_dir + '/env_matting/scaled_board_' + str(board_idx) + '.obj', 
                                                                                 data_dir + '/env_matting/board_' + str(board_idx) + '.png',
                                                                                 81, 40.0)
            self.board_mesh_list.append(board_mesh)
            self.board_texture_uvs_list.append(board_texture_uvs)
            self.board_texture_list.append(board_texture)

        self.board_scene_mesh_list = []
        self.board_scene_texture_uvs_list = []
        self.board_scene_texture_list = []
        for board_scene_idx in range(board_num):
            print('load board_{0}'.format(board_scene_idx))
            board_scene_mesh, board_scene_texture_uvs, board_scene_texture = load_mesh_and_texture(data_dir + '/scaled_board_' + str(board_scene_idx) + '_mesh_with_texture_clip.obj', 
                                                                                                   data_dir + '/board_' + str(board_scene_idx) + '_mesh_with_texture_u1_v1.png',
                                                                                                   31, 15.0, 3)
            self.board_scene_mesh_list.append(board_scene_mesh)
            self.board_scene_texture_uvs_list.append(board_scene_texture_uvs)
            self.board_scene_texture_list.append(board_scene_texture)
        print('finish load mesh and texture')

    def reflection(self, l, normal ):
        # l n x 3 x imHeight x imWidth
        # normal n x 3 x imHeight x imWidth
        # eta1 float
        # eta2 float

        cos_theta = torch.sum(l * (-normal), dim=1 ).unsqueeze(1)

        r_p = l + normal * cos_theta

        r_p_norm = torch.clamp(torch.sum(r_p * r_p, dim=1), 0, 0.999999 )

        r_i = torch.sqrt(1 - r_p_norm ).unsqueeze(1).expand_as(normal ) * normal
        r = -r_p + r_i
        r = r / torch.sqrt(torch.clamp(torch.sum(r*r, dim=1), min=1e-10 ).unsqueeze(1) )

        return r
    
    def refraction(self, l, normal, eta1, eta2 ):
        # l n x 3 x imHeight x imWidth
        # normal n x 3 x imHeight x imWidth
        # eta1 float
        # eta2 float

        cos_theta = torch.sum(l * (-normal), dim=1 ).unsqueeze(1)

        i_p = l + normal * cos_theta

        t_p = eta1 / eta2 * i_p

        t_p_norm = torch.sum(t_p * t_p, dim=1)
        total_reflect_mask = (t_p_norm.detach() > 0.999999)#.unsqueeze(1)

        t_i = torch.sqrt(1 - torch.clamp(t_p_norm, 0, 0.999999 ) ).unsqueeze(1).expand_as(normal ) * (-normal )
        t = t_i + t_p
        t = t / torch.sqrt( torch.clamp(torch.sum(t.detach() * t.detach(), dim=1 ), min=1e-10 ) ).unsqueeze(1)

        cos_theta_t = torch.sum(t * (-normal), dim=1 ).unsqueeze(1)

        e_i = (cos_theta_t * eta2 - cos_theta * eta1) / \
                torch.clamp(cos_theta_t * eta2 + cos_theta * eta1, min=1e-10 )
        e_p = (cos_theta_t * eta1 - cos_theta * eta2) / \
                torch.clamp(cos_theta_t * eta1 + cos_theta * eta2, min=1e-10 )

        attenuate = torch.clamp(0.5 * (e_i * e_i + e_p * e_p), 0, 1)

        return t, attenuate.squeeze().detach(), total_reflect_mask.detach()

    def sample_env_light(self, l, image, width, height):

        l = torch.clamp(l, -0.999999, 0.999999)
        # the env map need to be caliberated with captured images
        R_data = [[-0.32189582,  0.9466833,   0.01318344],
                  [ 0.02309316, -0.00606971,  0.99971489],
                  [ 0.94649342,  0.32210849, -0.0199081 ]]
        R = torch.tensor(R_data, dtype=torch.float).cuda().detach()
        l = l @ R.T
        # Compute theta and phi
        x, y, z = torch.split(l, [1, 1, 1], dim=1 )
        epsilon=1e-7 
        theta = torch.acos(torch.clamp(y, -1 + epsilon, 1 - epsilon))
        phi = torch.atan2( x, z )
        v = theta / np.pi
        u = (-phi / np.pi / 2.0 + 0.5)
        u, v = u.squeeze(-1), v.squeeze(-1)
      
        uvs = torch.stack((u, v), dim=1)
        uvs = (uvs - 0.5) * 2.0
        grid = uvs[None, :, None, :]
        image = image.permute(2, 0, 1)
        image = image[None, :, :, :]
        # input: (N, C, H, W)
        # grid : (N, H, W, 2)
        # outinput: (N, C, H, W)
        rendered_image = torch.nn.functional.grid_sample(image, grid, padding_mode='border', mode='bicubic', align_corners=True)
        rendered_image = rendered_image.permute(0, 2, 1, 3).squeeze(0).squeeze(-1)

        return rendered_image, uvs 

    def sample_from_texture(self, uvs, image):

        uvs = torch.clamp(uvs, 0.0, 1.0)
        uvs = (uvs - 0.5) * 2.0
        grid = uvs[None, :, None, :]
        image = image.permute(2, 0, 1)
        image = image[None, :, :, :]
        # input: (N, C, H, W)
        # grid : (N, H, W, 2)
        # outinput: (N, C, H, W)
        rendered_image = torch.nn.functional.grid_sample(image, grid, padding_mode='border', mode='bicubic', align_corners=True)
        rendered_image = rendered_image.permute(0, 2, 1, 3).squeeze(0).squeeze(-1)

        return rendered_image

    def calculate_sdf_normal(self, sdf, hit_points, retain_graph=False):

        if retain_graph:
            x = hit_points
        else:
            x = hit_points.detach()
        x.requires_grad_(True)
        y = sdf(x)
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        normals = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=retain_graph,
            retain_graph=retain_graph,
            only_inputs=True)[0]
        normals = normals / torch.sqrt(torch.clamp(torch.sum(normals*normals, dim=1), min=1e-10 ).unsqueeze(1) )
        if retain_graph:
            return normals
        else:
            return normals.detach()

    def merge_mesh(self, mesh_list):

        vertice_list = [mesh.vertices for mesh in mesh_list]
        faces_list = [mesh.faces for mesh in mesh_list]
        faces_offset = np.cumsum([v.shape[0] for v in vertice_list])
        faces_offset = np.insert(faces_offset, 0, 0)[:-1]

        vertices = np.vstack(vertice_list)
        faces = np.vstack([face + offset for face, offset in zip(faces_list, faces_offset)])

        # must set process=False!!!
        merged_meshes = trimesh.Trimesh(vertices, faces, process=False)
        return merged_meshes

    def diff_ray_triangle_intersec_trimesh(self, mesh, ray_origins, ray_directions, mesh_vertex_normals = None, mesh_vertex_uvs = None):
    
        points, index_ray, face_idx = mesh.ray.intersects_location(
            ray_origins.detach().cpu().numpy(), ray_directions.detach().cpu().numpy(), multiple_hits=False)

        vertices_tensor = torch.from_numpy(mesh.vertices).float().cuda()
        faces_tensor = torch.from_numpy(mesh.faces).long().cuda()
        face_idx_tensor = torch.from_numpy(face_idx).long().cuda()

        vertices = vertices_tensor[torch.flatten(faces_tensor[face_idx]), :]
        vertices = vertices.reshape(-1, 3, 3)
        v0 = vertices[:, 0, :]
        v1 = vertices[:, 1, :]
        v2 = vertices[:, 2, :]
        plane_normal = torch.cross(v2 - v0, v2 - v1)
        plane_normal = plane_normal / torch.sqrt(torch.clamp(torch.sum(plane_normal*plane_normal, dim=1), min=1e-10 ).unsqueeze(1))
        plane_point = (v0 + v1 + v2) * 0.33333333333333

        index_ray_tensor = torch.from_numpy(index_ray).long().cuda()
        ray_directions = ray_directions[index_ray_tensor]
        ray_origins = ray_origins[index_ray_tensor]
        points_tensor = self.line_plane_collision(plane_normal, plane_point, ray_directions, ray_origins)

        barycentric_weight = self.calculate_barycentric_weight(vertices_tensor, faces_tensor, points_tensor, face_idx_tensor)
        
        points_normals = None
        if mesh_vertex_normals != None:
            vertex_normals = mesh_vertex_normals[torch.flatten(faces_tensor[face_idx_tensor]), :]
            vertex_normals = vertex_normals.reshape(-1, 3, 3)
            v0_normals = vertex_normals[:, 0, :]
            v1_normals = vertex_normals[:, 1, :]
            v2_normals = vertex_normals[:, 2, :]
            points_normals = barycentric_weight[:, 0][:, None] * v0_normals + \
                barycentric_weight[:, 1][:, None] * v1_normals + \
                barycentric_weight[:, 2][:, None] * v2_normals
            points_normals = points_normals / torch.sqrt(torch.clamp(torch.sum(points_normals*points_normals, dim=1), min=1e-10 ).unsqueeze(1))
            points_normals = points_normals.float().cuda()
            
        points_uvs = None
        if mesh_vertex_uvs != None:
            uvs = mesh_vertex_uvs[torch.flatten(faces_tensor[face_idx_tensor]), :]
            uvs = uvs.reshape(-1, 3, 2)
            uv0 = uvs[:, 0, :]
            uv1 = uvs[:, 1, :]
            uv2 = uvs[:, 2, :]
            points_uvs = barycentric_weight[:, 0][:, None] * uv0 + \
                barycentric_weight[:, 1][:, None] * uv1 + \
                barycentric_weight[:, 2][:, None] * uv2

        return points_tensor, \
            torch.from_numpy(index_ray).long().cuda(), \
            torch.from_numpy(face_idx).long().cuda(), \
            points_normals, \
            points_uvs

    def diff_ray_plane_intersec_trimesh(self, mesh, ray_origins, ray_directions, mesh_vertex_uvs):
    
        vertices_tensor = torch.from_numpy(mesh.vertices).float().cuda()
        faces_tensor = torch.from_numpy(mesh.faces).long().cuda()

        ouv_3d = torch.zeros(3, 3).cuda()
        for i in range(mesh_vertex_uvs.shape[0]):
            if mesh_vertex_uvs[i, 0] < 0.01 and mesh_vertex_uvs[i, 1] < 0.01:
                ouv_3d[0, :] = vertices_tensor[i, :] 
            if mesh_vertex_uvs[i, 0] > 0.01 and mesh_vertex_uvs[i, 1] < 0.01:
                ouv_3d[1, :] = vertices_tensor[i, :] 
            if mesh_vertex_uvs[i, 0] < 0.01 and mesh_vertex_uvs[i, 1] > 0.01:
                ouv_3d[2, :] = vertices_tensor[i, :] 

        vertices = vertices_tensor[torch.flatten(faces_tensor), :]
        vertices = vertices.reshape(-1, 3, 3)
        v0 = vertices[0, 0, :]
        v1 = vertices[0, 1, :]
        v2 = vertices[0, 2, :]
        plane_normal = torch.cross(v2 - v0, v2 - v1)
        plane_normal = plane_normal / torch.sqrt(torch.clamp(torch.sum(plane_normal*plane_normal), min=1e-10 ))
        plane_point = (v0 + v1 + v2) * 0.33333333333333

        _, valid_mask, denom = trimesh.intersections.planes_lines(plane_point.detach().cpu().numpy(), plane_normal.detach().cpu().numpy(), 
                                                                  ray_origins.detach().cpu().numpy(), ray_directions.detach().cpu().numpy(),
                                                                  return_distance = False, return_denom = True)
        valid_mask[np.abs(denom) < 0.2] = False
        valid_mask_tensor = torch.from_numpy(valid_mask).bool().cuda()

        ray_directions = ray_directions[valid_mask_tensor, :]
        ray_origins = ray_origins[valid_mask_tensor, :]
        plane_normal = plane_normal[None, :].expand(ray_directions.shape)
        plane_point = plane_point[None, :].expand(ray_directions.shape)
        hit_points = self.line_plane_collision(plane_normal, plane_point, ray_directions, ray_origins)

        points_tensor = hit_points
        hit_points_vec = hit_points - ouv_3d[0, :][None, :]
        axis_u = ouv_3d[1, :] - ouv_3d[0, :]
        axis_u_len = torch.sqrt(torch.clamp(torch.sum(axis_u*axis_u, dim=0), min=1e-10 ))
        axis_u = axis_u[None, :].expand(hit_points_vec.shape)
        axis_v = ouv_3d[2, :] - ouv_3d[0, :]
        axis_v_len = torch.sqrt(torch.clamp(torch.sum(axis_v*axis_v, dim=0), min=1e-10 ))
        axis_v = axis_v[None, :].expand(hit_points_vec.shape)

        points_u = torch.bmm(hit_points_vec[:, None, :], axis_u[:, :, None]).squeeze(2) / axis_u_len / axis_u_len 
        points_v = torch.bmm(hit_points_vec[:, None, :], axis_v[:, :, None]).squeeze(2) / axis_v_len / axis_v_len
        points_uvs = torch.cat((points_u, points_v), dim=1)

        points_u = points_u.squeeze()
        points_v = points_v.squeeze()
        inside_mask = (points_u >= 0.0) & (points_u <= 1.0) & (points_v >= 0.0) & (points_v <= 1.0)

        index_ray = np.nonzero(valid_mask)
        index_ray = torch.from_numpy(np.array(index_ray).squeeze())
        
        return points_tensor, \
            index_ray.long().cuda(), \
            points_uvs, \
            inside_mask 

    def ray_triangle_intersec_trimesh(self, mesh, ray_origins, ray_directions, mesh_vertex_normals = None, mesh_vertex_uvs = None):

        # start = time.time()
        points, index_ray, face_idx = mesh.ray.intersects_location(
            ray_origins.detach().cpu().numpy(), ray_directions.detach().cpu().numpy(), multiple_hits=False)
        # torch.cuda.synchronize()
        # end = time.time()
        # print("trimesh_intersect time: {0} s".format(end - start))

        vertices_tensor = torch.from_numpy(mesh.vertices).float().cuda()
        faces_tensor = torch.from_numpy(mesh.faces).long().cuda()
        points_tensor = torch.from_numpy(points).float().cuda()
        face_idx_tensor = torch.from_numpy(face_idx).long().cuda()
        barycentric_weight = self.calculate_barycentric_weight(vertices_tensor, faces_tensor, points_tensor, face_idx_tensor)
        
        points_normals = None
        if mesh_vertex_normals != None:
            vertex_normals = mesh_vertex_normals[torch.flatten(faces_tensor[face_idx_tensor]), :]
            vertex_normals = vertex_normals.reshape(-1, 3, 3)
            v0_normals = vertex_normals[:, 0, :]
            v1_normals = vertex_normals[:, 1, :]
            v2_normals = vertex_normals[:, 2, :]
            points_normals = barycentric_weight[:, 0][:, None] * v0_normals + \
                barycentric_weight[:, 1][:, None] * v1_normals + \
                barycentric_weight[:, 2][:, None] * v2_normals
            points_normals = points_normals / torch.sqrt(torch.clamp(torch.sum(points_normals*points_normals, dim=1), min=1e-10 ).unsqueeze(1))
            points_normals = points_normals.float().cuda()
            
        points_uvs = None
        if mesh_vertex_uvs != None:
            uvs = mesh_vertex_uvs[torch.flatten(faces_tensor[face_idx_tensor]), :]
            uvs = uvs.reshape(-1, 3, 2)
            uv0 = uvs[:, 0, :]
            uv1 = uvs[:, 1, :]
            uv2 = uvs[:, 2, :]
            points_uvs = barycentric_weight[:, 0][:, None] * uv0 + \
                barycentric_weight[:, 1][:, None] * uv1 + \
                barycentric_weight[:, 2][:, None] * uv2

        return torch.from_numpy(points).float().cuda(), \
            torch.from_numpy(index_ray).long().cuda(), \
            torch.from_numpy(face_idx).long().cuda(), \
            points_normals, \
            points_uvs

    def ray_triangle_intersec_trimesh_optix(self, mesh, scene, ray_origins, ray_directions, mesh_vertex_normals = None, mesh_vertex_uvs = None):

        # print('ray_triangle_intersec_trimesh_optix')
        # start = time.time()
        if ray_origins.shape[0] > 0 and ray_directions.shape[0] > 0:
            points_tensor, face_idx_tensor, index_ray_tensor = scene.optix_intersect(Ray(ray_origins, ray_directions))
        else:
            points_tensor = torch.zeros_like(ray_origins)
            face_idx_tensor = torch.zeros(ray_origins.shape[0]).long().cuda()
            index_ray_tensor = torch.zeros(ray_origins.shape[0]).long().cuda()
        # torch.cuda.synchronize()
        # end = time.time()
        # print("optix_intersect time: {0} s".format(end - start))

        vertices_tensor = torch.from_numpy(mesh.vertices).float().cuda() 
        faces_tensor = torch.from_numpy(mesh.faces).long().cuda()
        barycentric_weight = self.calculate_barycentric_weight(vertices_tensor, faces_tensor, points_tensor, face_idx_tensor)
        
        points_normals = None
        if mesh_vertex_normals != None:
            vertex_normals = mesh_vertex_normals[torch.flatten(faces_tensor[face_idx_tensor]), :]
            vertex_normals = vertex_normals.reshape(-1, 3, 3)
            v0_normals = vertex_normals[:, 0, :]
            v1_normals = vertex_normals[:, 1, :]
            v2_normals = vertex_normals[:, 2, :]
            points_normals = barycentric_weight[:, 0][:, None] * v0_normals + \
                barycentric_weight[:, 1][:, None] * v1_normals + \
                barycentric_weight[:, 2][:, None] * v2_normals
            points_normals = points_normals / torch.sqrt(torch.clamp(torch.sum(points_normals*points_normals, dim=1), min=1e-10 ).unsqueeze(1))
            points_normals = points_normals.float().cuda()
            
        points_uvs = None
        if mesh_vertex_uvs != None:
            uvs = mesh_vertex_uvs[torch.flatten(faces_tensor[face_idx_tensor]), :]
            uvs = uvs.reshape(-1, 3, 2)
            uv0 = uvs[:, 0, :]
            uv1 = uvs[:, 1, :]
            uv2 = uvs[:, 2, :]
            points_uvs = barycentric_weight[:, 0][:, None] * uv0 + \
                barycentric_weight[:, 1][:, None] * uv1 + \
                barycentric_weight[:, 2][:, None] * uv2

        return points_tensor, \
            index_ray_tensor, \
            face_idx_tensor, \
            points_normals, \
            points_uvs

    def line_plane_collision(self, plane_normal, plane_point, ray_directions, ray_origins, epsilon=1e-6):
 
        ndotu = torch.bmm(plane_normal[:, None, :], ray_directions[:, :, None]).squeeze(-1)
        # if abs(ndotu) < epsilon:
        #     raise RuntimeError("no intersection or line is within plane")
    
        w = ray_origins - plane_point
        # si = -plane_normal.dot(w) / ndotu
        si = -torch.bmm(plane_normal[:, None, :], w[:, :, None]).squeeze(-1) / ndotu
        psi = w + si * ray_directions + plane_point

        return psi

    def calculate_barycentric_weight(self, vertices_tensor, faces_tensor, points, face_idx):

        vertices = vertices_tensor[torch.flatten(faces_tensor[face_idx]), :]
        vertices = vertices.reshape(-1, 3, 3)
        v0 = vertices[:, 0, :]
        v1 = vertices[:, 1, :]
        v2 = vertices[:, 2, :]
        P = points
 
        v0v1 = v1 - v0
        v0v2 = v2 - v0
        # no need to normalize
        N = torch.cross(v0v1, v0v2)
        area = torch.norm(N, p=2, dim=1) / 2 
    
        # edge 1
        edge1 = v2 - v1
        vp1 = P - v1
        C = torch.cross(edge1, vp1)
        u = torch.norm(C, p=2, dim=1) / 2 / (area + 1.0e-20) 
 
        # edge 2
        edge2 = v0 - v2
        vp2 = P - v2
        C = torch.cross(edge2, vp2)
        v = torch.norm(C, p=2, dim=1) / 2 / (area + 1.0e-20)

        u = torch.clamp(u, min=1e-10, max=0.999999)
        v = torch.clamp(v, min=1e-10, max=0.999999)

        barycentric_weight = torch.stack((u, v, 1.0 - u - v), dim=1)
        # print(barycentric_weight.shape)

        return barycentric_weight

    def get_background_image(self,
                  ray_origins,
                  ray_directions):

        image_tensor = torch.zeros(ray_origins.shape[0], 3).cuda()

        # image_tensor, _ = self.sample_env_light(ray_directions, self.env_image.detach(), self.env_width, self.env_height)

        _, index_ray, _, _, points_uvs = self.ray_triangle_intersec_trimesh(self.scene_mesh, ray_origins, ray_directions, None, self.scene_texture_uvs)
        image_tensor[index_ray, :] = self.sample_from_texture(points_uvs, self.scene_texture)
        return image_tensor
 
    # ray_origins_tensor, 
    # ray_directions_tensor,
    # valid_ray_mask,
    def diff_mesh_ray_tracing(self,
                              mesh_vertices_tensor,
                              mesh_faces_tensor,
                              mesh_vertex_normals_tensor,
                              face_idx_tensor, 
                              valid_ray_mask,
                              start_ray_origins,
                              start_ray_directions,
                              ray_board_indices,
                              material_ior,
                              use_our_corr,
                              patch_size):

        # torch.autograd.set_detect_anomaly(True) 
        total_vis_pcd = []
        vis_debug = False

        # reflect_ray_directions, reflect_ray_origins are first-bounce reflected ray 
        # ray_directions, ray_origins are second-bounce refracted ray  
        ################################################################################
        ray_origins_list = [start_ray_origins]
        ray_directions_list = [start_ray_directions]
        vertex_idx_list = []
        accu_refract_attenuate = torch.ones(ray_origins_list[0].shape[0]).cuda()
        reflect_attenuate = torch.ones(ray_origins_list[0].shape[0]).cuda()
        for ray_iter in range(face_idx_tensor.shape[0]):
            # print("ray_iter: {0}".format(ray_iter))
            face_idx = face_idx_tensor[ray_iter, :].detach()    
            ray_origins = ray_origins_list[ray_iter]
            ray_directions = ray_directions_list[ray_iter]
            if ray_iter == 2:
                break

            vertex_idx_list.append(torch.flatten(mesh_faces_tensor[face_idx]))
            vertices = mesh_vertices_tensor[torch.flatten(mesh_faces_tensor[face_idx]), :]
            vertices = vertices.reshape(-1, 3, 3)
            v0 = vertices[:, 0, :]
            v1 = vertices[:, 1, :]
            v2 = vertices[:, 2, :]
            plane_normal = torch.cross(v2 - v0, v2 - v1)
            plane_normal = plane_normal / torch.sqrt(torch.clamp(torch.sum(plane_normal*plane_normal, dim=1), min=1e-10 ).unsqueeze(1))
            plane_point = (v0 + v1 + v2) * 0.33333333333333

            hit_points = self.line_plane_collision(plane_normal, plane_point, ray_directions, ray_origins)
            if vis_debug:
                vis_pcd = o3d.geometry.PointCloud()
                vis_pcd.points = o3d.utility.Vector3dVector(hit_points.cpu().detach().numpy())
                colors = np.array([[0, 1 - ray_iter * 0.3, ray_iter * 0.3] for i in range(len(vis_pcd.points))])
                vis_pcd.colors = o3d.utility.Vector3dVector(colors[:, :])
                total_vis_pcd.append(vis_pcd)

            barycentric_weight = self.calculate_barycentric_weight(mesh_vertices_tensor, mesh_faces_tensor, hit_points, face_idx)
            barycentric_weight = barycentric_weight
            vertex_normals = mesh_vertex_normals_tensor[torch.flatten(mesh_faces_tensor[face_idx]), :]
            vertex_normals = vertex_normals.reshape(-1, 3, 3)
            v0_normals = vertex_normals[:, 0, :]
            v1_normals = vertex_normals[:, 1, :]
            v2_normals = vertex_normals[:, 2, :]
            hit_normals = barycentric_weight[:, 0][:, None] * v0_normals + \
                barycentric_weight[:, 1][:, None] * v1_normals + \
                barycentric_weight[:, 2][:, None] * v2_normals
            hit_normals = hit_normals / torch.sqrt(torch.clamp(torch.sum(hit_normals*hit_normals, dim=1), min=1e-10 ).unsqueeze(1))          
        
            if ray_iter % 2 == 0:
                refract_ray_directions, refract_attenuate, _ = self.refraction(ray_directions, hit_normals, eta1 = 1.0003, eta2 = material_ior)
            else:
                refract_ray_directions, refract_attenuate, _ = self.refraction(ray_directions, -hit_normals, eta1 = material_ior, eta2 = 1.0003)
            
            accu_refract_attenuate *= (1 - refract_attenuate)
            if ray_iter == 0:
                reflect_ray_directions = self.reflection(-ray_directions, hit_normals)
                reflect_ray_origins = hit_points
                reflect_attenuate = refract_attenuate
            ray_origins_list.append(hit_points.clone())
            ray_directions_list.append(refract_ray_directions.clone())

        ray_origins = ray_origins_list[2]
        ray_directions = ray_directions_list[2]
        relavent_vertex_idx_tensor = torch.cat(vertex_idx_list, dim=0)
        relavent_vertex_idx_tensor = torch.unique(relavent_vertex_idx_tensor, sorted=True)
        
        if vis_debug:
            vis_mesh = o3d.geometry.TriangleMesh()
            vis_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces_tensor.cpu().detach().numpy())
            vis_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
            total_vis_pcd.append(vis_mesh)
            o3d.visualization.draw_geometries(geometry_list = total_vis_pcd, width = 640, height = 480, window_name = "mesh_with_hit_points_diff", point_show_normal=False) 
        ################################################################################ 
      
        num_pixels = valid_ray_mask.shape[0]
        second_bounce_ray_origins, second_bounce_ray_directions = torch.zeros(num_pixels, 3).float().cuda(), torch.zeros(num_pixels, 3).float().cuda()
        second_bounce_ray_origins[valid_ray_mask, :] = ray_origins
        second_bounce_ray_directions[valid_ray_mask, :] = ray_directions

        refract_img, reflect_img = torch.zeros(num_pixels, 3).float().cuda(), torch.zeros(num_pixels, 3).float().cuda()
        all_board_hit_points_uvs = torch.FloatTensor(num_pixels, 2).zero_().cuda() / 0.0 # set all to NaN

        if torch.sum(valid_ray_mask).cpu().item() < 1 or refract_attenuate.numel() == 0:
            return refract_img, \
               valid_ray_mask.detach(), \
               relavent_vertex_idx_tensor.detach(), \
               all_board_hit_points_uvs.detach(), \
               second_bounce_ray_origins, \
               second_bounce_ray_directions

        # calculate refract image and reflect image
        ################################################################################ 
        if self.with_env_map:
            refract_img[valid_ray_mask, :], _ = self.sample_env_light(ray_directions, self.env_image.detach(), self.env_width, self.env_height) 
            reflect_img[valid_ray_mask, :], _ = self.sample_env_light(reflect_ray_directions, self.env_image.detach(), self.env_width, self.env_height)
  
        # 1. fetch color from scene mesh
        if self.with_global_scene is True:
            hit_points, index_ray, _, _, points_uvs = self.diff_ray_triangle_intersec_trimesh(self.scene_mesh, ray_origins, ray_directions, None, self.scene_texture_uvs)
            temp_tensor = refract_img[valid_ray_mask, :]
            temp_tensor[index_ray, :] = self.sample_from_texture(points_uvs, self.scene_texture)
            refract_img[valid_ray_mask, :] = temp_tensor

        # 2. fetch color and ray-cell correspondence from background pattern
        if use_our_corr:
            board_indices = torch.unique(ray_board_indices)
            for i in range(board_indices.shape[0]):
                board_idx = board_indices[i]
                # print('board_idx: {0}'.format(board_idx))
                board_mesh, board_texture_uvs, board_texture = self.board_mesh_list[board_idx], self.board_texture_uvs_list[board_idx], self.board_texture_list[board_idx]
    
                board_valid_ray_mask = valid_ray_mask.clone() 
                board_valid_ray_mask[ray_board_indices != board_idx] = False
                board_ray_origins = ray_origins[ray_board_indices[valid_ray_mask] == board_idx, :]
                board_ray_directions = ray_directions[ray_board_indices[valid_ray_mask] == board_idx, :]
                board_reflect_ray_origins = reflect_ray_origins[ray_board_indices[valid_ray_mask] == board_idx, :]
                board_reflect_ray_directions = reflect_ray_directions[ray_board_indices[valid_ray_mask] == board_idx, :]

                if self.with_board_scene:
                    board_scene_idx = board_idx // 15 
                    board_scene_hit_points, board_scene_index_ray, _, _, board_scene_points_uvs = self.diff_ray_triangle_intersec_trimesh(self.board_scene_mesh_list[board_scene_idx], board_ray_origins, board_ray_directions, None, self.board_scene_texture_uvs_list[board_scene_idx])
                    temp_tensor = refract_img[board_valid_ray_mask, :]
                    temp_tensor[board_scene_index_ray, :] = self.sample_from_texture(board_scene_points_uvs, self.board_scene_texture_list[board_scene_idx])
                    refract_img[board_valid_ray_mask, :] = temp_tensor 

                    if self.with_reflect:
                        board_reflect_scene_hit_points, board_reflect_scene_index_ray, _, _, board_reflect_scene_points_uvs = self.diff_ray_triangle_intersec_trimesh(self.board_scene_mesh_list[board_scene_idx], board_reflect_ray_origins, board_reflect_ray_directions, None, self.board_scene_texture_uvs_list[board_scene_idx])
                        reflect_temp_tensor = refract_img[board_valid_ray_mask, :]
                        reflect_temp_tensor[board_reflect_scene_index_ray, :] = self.sample_from_texture(board_reflect_scene_points_uvs, self.board_scene_texture_list[board_scene_idx])
                        reflect_img[board_valid_ray_mask, :] = temp_tensor 

                # board_hit_points, index_ray, _, _, board_points_uvs = self.diff_ray_triangle_intersec_trimesh(board_mesh, board_ray_origins, board_ray_directions, None, board_texture_uvs)
                board_hit_points, index_ray, board_points_uvs, inside_mask = self.diff_ray_plane_intersec_trimesh(board_mesh, board_ray_origins, board_ray_directions, board_texture_uvs)
                
                temp_board_points_uvs = torch.FloatTensor(board_ray_origins.shape[0], 2).zero_().cuda() / 0.0 # set all to NaN
                temp_board_points_uvs[index_ray, :] = board_points_uvs
                all_board_hit_points_uvs[board_valid_ray_mask, :] = temp_board_points_uvs
                temp_tensor = refract_img[board_valid_ray_mask, :]
                
                if board_points_uvs.nelement() > 0 and \
                    inside_mask.nelement() > 0 and \
                    torch.sum(inside_mask) > 10:
                    
                    in_board_mask = torch.zeros(temp_tensor.shape[0]).bool().cuda() 
                    in_board_mask[index_ray] = True 
                    # 注意：in_board_mask[index_ray][~inside_mask] = False, torch导数传递不能两重括号，用下面语法代替
                    temp = in_board_mask[index_ray]
                    temp[~inside_mask] = False
                    in_board_mask[index_ray] = temp
                    temp_tensor[in_board_mask, :] = self.sample_from_texture(board_points_uvs[inside_mask, :], board_texture)
                    refract_img[board_valid_ray_mask, :] = temp_tensor

        refract_img[valid_ray_mask, :] = refract_img[valid_ray_mask, :] * accu_refract_attenuate[:, None].detach()  
        if self.with_reflect:
            reflect_img[valid_ray_mask, :] = reflect_img[valid_ray_mask, :] * reflect_attenuate[:, None].detach()  
        refract_img = torch.clamp(refract_img, 0, 0.999999)
        reflect_img = torch.clamp(reflect_img, 0, 0.999999)
        ################################################################################ 

        # add gaussian for refract and reflect parts
        if self.training:
            kernel_size_1, sigma_1 = (3, 3), (1.5, 1.5)
            kernel_size_2, sigma_2 = (3, 3), (1.5, 1.5)
            num_pixels = patch_size ** 2
            refract_img = refract_img.reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)
            refract_img = kornia.filters.gaussian_blur2d(refract_img, kernel_size_1, sigma_1, "replicate")
            refract_img = refract_img.permute(0, 2, 3, 1)
            reflect_img = reflect_img.reshape(-1,patch_size, patch_size, 3).permute(0, 3, 1, 2)
            reflect_img = kornia.filters.gaussian_blur2d(reflect_img, kernel_size_2, sigma_2, "replicate")
            reflect_img = reflect_img.permute(0, 2, 3, 1)

        if self.with_reflect:
            composite_img = refract_img.reshape(-1, 3) + reflect_img.reshape(-1, 3)
        else:
            composite_img = refract_img.reshape(-1, 3)
        composite_img = torch.clamp(composite_img, 0, 0.999999)
        valid_ray_mask[valid_ray_mask][accu_refract_attenuate < 0.01] = False 
        composite_img[valid_ray_mask, :][accu_refract_attenuate < 0.01, :] = torch.zeros(3).cuda()

        if not self.training:
            return composite_img.detach(), \
                   valid_ray_mask.detach(), \
                   relavent_vertex_idx_tensor.detach(), \
                   all_board_hit_points_uvs.detach(), \
                   second_bounce_ray_origins, \
                   second_bounce_ray_directions
        return composite_img, \
               valid_ray_mask.detach(), \
               relavent_vertex_idx_tensor.detach(), \
               all_board_hit_points_uvs, \
               second_bounce_ray_origins, \
               second_bounce_ray_directions
               
    def diff_mesh_ray_tracing_idr(self,
                                  sdf,
                                  calculate_sdf_normal,
                                  point_sampler,
                                  mesh_vertices_tensor,
                                  mesh_faces_tensor,
                                  mesh_vertex_normals_tensor,
                                  face_idx_tensor, 
                                  valid_ray_mask,
                                  start_ray_origins,
                                  start_ray_directions,
                                  ray_board_indices,
                                  material_ior,
                                  use_our_corr,
                                  patch_size):

        # torch.autograd.set_detect_anomaly(True) 
        total_vis_pcd = []
        vis_debug = False

        # reflect_ray_directions, reflect_ray_origins are first-bounce reflected ray 
        # ray_directions, ray_origins are second-bounce refracted ray  
        ################################################################################
        ray_origins_list = [start_ray_origins]
        ray_directions_list = [start_ray_directions]
        vertex_idx_list = []
        accu_refract_attenuate = torch.ones(ray_origins_list[0].shape[0]).cuda()
        reflect_attenuate = torch.ones(ray_origins_list[0].shape[0]).cuda()
        for ray_iter in range(face_idx_tensor.shape[0]):
            # print("ray_iter: {0}".format(ray_iter))
            face_idx = face_idx_tensor[ray_iter, :].detach()    
            ray_origins = ray_origins_list[ray_iter]
            ray_directions = ray_directions_list[ray_iter]
            if ray_iter == 2:
                break

            # calc hit points
            vertex_idx_list.append(torch.flatten(mesh_faces_tensor[face_idx]))
            vertices = mesh_vertices_tensor[torch.flatten(mesh_faces_tensor[face_idx]), :]
            vertices = vertices.reshape(-1, 3, 3)
            v0 = vertices[:, 0, :]
            v1 = vertices[:, 1, :]
            v2 = vertices[:, 2, :]
            plane_normal = torch.cross(v2 - v0, v2 - v1)
            plane_normal = plane_normal / torch.sqrt(torch.clamp(torch.sum(plane_normal*plane_normal, dim=1), min=1e-10 ).unsqueeze(1))
            plane_point = (v0 + v1 + v2) * 0.33333333333333
            hit_points = self.line_plane_collision(plane_normal, plane_point, ray_directions, ray_origins)

            sdf_normals = self.calculate_sdf_normal(sdf=sdf, 
                                                    hit_points=hit_points, 
                                                    retain_graph=False)
            sampler_min_max = torch.zeros(hit_points.shape[0], 2).cuda()
            sampler_min_max[:, 0] = torch.ones(hit_points.shape[0]).cuda() * (-0.002)
            sampler_min_max[:, 1] = torch.ones(hit_points.shape[0]).cuda() * (0.002)
            hit_points = point_sampler(sdf=sdf,
                                       ray_directions=-sdf_normals.detach(),
                                       sampler_min_max=sampler_min_max.detach(),
                                       origin_points=hit_points.detach(),
                                       n_steps=6)
            hit_points = hit_points.detach()
            
            hit_normals = calculate_sdf_normal(sdf=sdf, 
                                               hit_points=hit_points, 
                                               retain_graph=True)
            sdf_tensor = sdf(hit_points)[:, None]
            sdf_tensor_detach = sdf_tensor.clone().detach()
            if hit_points.shape[0] > 0:
                hit_points = hit_points - hit_normals * (sdf_tensor - sdf_tensor_detach)

            if vis_debug:
                vis_pcd = o3d.geometry.PointCloud()
                vis_pcd.points = o3d.utility.Vector3dVector(hit_points.cpu().detach().numpy())
                colors = np.array([[0, 1 - ray_iter * 0.3, ray_iter * 0.3] for i in range(len(vis_pcd.points))])
                vis_pcd.colors = o3d.utility.Vector3dVector(colors[:, :])
                total_vis_pcd.append(vis_pcd)
            
            if ray_iter % 2 == 0:
                refract_ray_directions, refract_attenuate, _ = self.refraction(ray_directions, hit_normals, eta1 = 1.0003, eta2 = material_ior)
            else:
                refract_ray_directions, refract_attenuate, _ = self.refraction(ray_directions, -hit_normals, eta1 = material_ior, eta2 = 1.0003)
            
            accu_refract_attenuate *= (1 - refract_attenuate)
            if ray_iter == 0:
                reflect_ray_directions = self.reflection(-ray_directions, hit_normals)
                reflect_ray_origins = hit_points
                reflect_attenuate = refract_attenuate
            ray_origins_list.append(hit_points.clone())
            ray_directions_list.append(refract_ray_directions.clone())

        ray_origins = ray_origins_list[2]
        ray_directions = ray_directions_list[2]
        relavent_vertex_idx_tensor = torch.cat(vertex_idx_list, dim=0)
        relavent_vertex_idx_tensor = torch.unique(relavent_vertex_idx_tensor, sorted=True)
        
        if vis_debug:
            vis_mesh = o3d.geometry.TriangleMesh()
            vis_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces_tensor.cpu().detach().numpy())
            vis_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
            total_vis_pcd.append(vis_mesh)
            o3d.visualization.draw_geometries(geometry_list = total_vis_pcd, width = 640, height = 480, window_name = "mesh_with_hit_points_diff", point_show_normal=False) 
        ################################################################################ 
      
        num_pixels = valid_ray_mask.shape[0]
        second_bounce_ray_origins, second_bounce_ray_directions = torch.zeros(num_pixels, 3).float().cuda(), torch.zeros(num_pixels, 3).float().cuda()
        second_bounce_ray_origins[valid_ray_mask, :] = ray_origins
        second_bounce_ray_directions[valid_ray_mask, :] = ray_directions

        refract_img, reflect_img = torch.zeros(num_pixels, 3).float().cuda(), torch.zeros(num_pixels, 3).float().cuda()
        all_board_hit_points_uvs = torch.FloatTensor(num_pixels, 2).zero_().cuda() / 0.0 # set all to NaN

        if torch.sum(valid_ray_mask).cpu().item() < 1 or refract_attenuate.numel() == 0:
            return refract_img, \
               valid_ray_mask.detach(), \
               relavent_vertex_idx_tensor.detach(), \
               all_board_hit_points_uvs.detach(), \
               second_bounce_ray_origins, \
               second_bounce_ray_directions

        # calculate refract image and reflect image
        ################################################################################ 
        if self.with_env_map:
            refract_img[valid_ray_mask, :], _ = self.sample_env_light(ray_directions, self.env_image.detach(), self.env_width, self.env_height) 
            reflect_img[valid_ray_mask, :], _ = self.sample_env_light(reflect_ray_directions, self.env_image.detach(), self.env_width, self.env_height)
  
        # 1. fetch color from scene mesh
        if self.with_global_scene is True:
            hit_points, index_ray, _, _, points_uvs = self.diff_ray_triangle_intersec_trimesh(self.scene_mesh, ray_origins, ray_directions, None, self.scene_texture_uvs)
            temp_tensor = refract_img[valid_ray_mask, :]
            temp_tensor[index_ray, :] = self.sample_from_texture(points_uvs, self.scene_texture)
            refract_img[valid_ray_mask, :] = temp_tensor

        # 2. fetch color and ray-cell correspondence from background pattern
        if use_our_corr:
            board_indices = torch.unique(ray_board_indices)
            for i in range(board_indices.shape[0]):
                board_idx = board_indices[i]
                # print('board_idx: {0}'.format(board_idx))
                board_mesh, board_texture_uvs, board_texture = self.board_mesh_list[board_idx], self.board_texture_uvs_list[board_idx], self.board_texture_list[board_idx]
    
                board_valid_ray_mask = valid_ray_mask.clone() 
                board_valid_ray_mask[ray_board_indices != board_idx] = False
                board_ray_origins = ray_origins[ray_board_indices[valid_ray_mask] == board_idx, :]
                board_ray_directions = ray_directions[ray_board_indices[valid_ray_mask] == board_idx, :]
                board_reflect_ray_origins = reflect_ray_origins[ray_board_indices[valid_ray_mask] == board_idx, :]
                board_reflect_ray_directions = reflect_ray_directions[ray_board_indices[valid_ray_mask] == board_idx, :]

                if self.with_board_scene:
                    board_scene_idx = board_idx // 15 
                    board_scene_hit_points, board_scene_index_ray, _, _, board_scene_points_uvs = self.diff_ray_triangle_intersec_trimesh(self.board_scene_mesh_list[board_scene_idx], board_ray_origins, board_ray_directions, None, self.board_scene_texture_uvs_list[board_scene_idx])
                    temp_tensor = refract_img[board_valid_ray_mask, :]
                    temp_tensor[board_scene_index_ray, :] = self.sample_from_texture(board_scene_points_uvs, self.board_scene_texture_list[board_scene_idx])
                    refract_img[board_valid_ray_mask, :] = temp_tensor 

                    if self.with_reflect:
                        board_reflect_scene_hit_points, board_reflect_scene_index_ray, _, _, board_reflect_scene_points_uvs = self.diff_ray_triangle_intersec_trimesh(self.board_scene_mesh_list[board_scene_idx], board_reflect_ray_origins, board_reflect_ray_directions, None, self.board_scene_texture_uvs_list[board_scene_idx])
                        reflect_temp_tensor = refract_img[board_valid_ray_mask, :]
                        reflect_temp_tensor[board_reflect_scene_index_ray, :] = self.sample_from_texture(board_reflect_scene_points_uvs, self.board_scene_texture_list[board_scene_idx])
                        reflect_img[board_valid_ray_mask, :] = temp_tensor 

                # board_hit_points, index_ray, _, _, board_points_uvs = self.diff_ray_triangle_intersec_trimesh(board_mesh, board_ray_origins, board_ray_directions, None, board_texture_uvs)
                board_hit_points, index_ray, board_points_uvs, inside_mask = self.diff_ray_plane_intersec_trimesh(board_mesh, board_ray_origins, board_ray_directions, board_texture_uvs)
                
                temp_board_points_uvs = torch.FloatTensor(board_ray_origins.shape[0], 2).zero_().cuda() / 0.0 # set all to NaN
                temp_board_points_uvs[index_ray, :] = board_points_uvs
                all_board_hit_points_uvs[board_valid_ray_mask, :] = temp_board_points_uvs
                temp_tensor = refract_img[board_valid_ray_mask, :]
                
                if board_points_uvs.nelement() > 0 and \
                    inside_mask.nelement() > 0 and \
                    torch.sum(inside_mask) > 10:
                    
                    in_board_mask = torch.zeros(temp_tensor.shape[0]).bool().cuda() 
                    in_board_mask[index_ray] = True 
                    # 注意：in_board_mask[index_ray][~inside_mask] = False, torch导数传递不能两重括号，用下面语法代替
                    temp = in_board_mask[index_ray]
                    temp[~inside_mask] = False
                    in_board_mask[index_ray] = temp
                    temp_tensor[in_board_mask, :] = self.sample_from_texture(board_points_uvs[inside_mask, :], board_texture)
                    refract_img[board_valid_ray_mask, :] = temp_tensor

        refract_img[valid_ray_mask, :] = refract_img[valid_ray_mask, :] * accu_refract_attenuate[:, None].detach()  
        if self.with_reflect:
            reflect_img[valid_ray_mask, :] = reflect_img[valid_ray_mask, :] * reflect_attenuate[:, None].detach()  
        refract_img = torch.clamp(refract_img, 0, 0.999999)
        reflect_img = torch.clamp(reflect_img, 0, 0.999999)
        ################################################################################ 

        # add gaussian for refract and reflect parts
        if self.training:
            kernel_size_1, sigma_1 = (3, 3), (1.5, 1.5)
            kernel_size_2, sigma_2 = (3, 3), (1.5, 1.5)
            num_pixels = patch_size ** 2
            refract_img = refract_img.reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)
            refract_img = kornia.filters.gaussian_blur2d(refract_img, kernel_size_1, sigma_1, "replicate")
            refract_img = refract_img.permute(0, 2, 3, 1)
            reflect_img = reflect_img.reshape(-1,patch_size, patch_size, 3).permute(0, 3, 1, 2)
            reflect_img = kornia.filters.gaussian_blur2d(reflect_img, kernel_size_2, sigma_2, "replicate")
            reflect_img = reflect_img.permute(0, 2, 3, 1)

        if self.with_reflect:
            composite_img = refract_img.reshape(-1, 3) + reflect_img.reshape(-1, 3)
        else:
            composite_img = refract_img.reshape(-1, 3)
        composite_img = torch.clamp(composite_img, 0, 0.999999)
        valid_ray_mask[valid_ray_mask][accu_refract_attenuate < 0.01] = False 
        composite_img[valid_ray_mask, :][accu_refract_attenuate < 0.01, :] = torch.zeros(3).cuda()

        if not self.training:
            return composite_img.detach(), \
                   valid_ray_mask.detach(), \
                   relavent_vertex_idx_tensor.detach(), \
                   all_board_hit_points_uvs.detach(), \
                   second_bounce_ray_origins, \
                   second_bounce_ray_directions
        return composite_img, \
               valid_ray_mask.detach(), \
               relavent_vertex_idx_tensor.detach(), \
               all_board_hit_points_uvs, \
               second_bounce_ray_origins, \
               second_bounce_ray_directions

    def mesh_ray_tracing_optix(self,
                               mesh,
                               calculate_vertex_normal,
                               ray_origins,
                               ray_directions,
                               only_mask_loss,
                               material_ior):
        
        # start = time.time() 
        mesh_face_num = mesh.faces.shape[0]
        merged_mesh = self.merge_mesh([mesh, self.env_map_mesh])
        # merged_mesh.export('../test_merged_mesh.ply')

        # vis_mesh = o3d.geometry.TriangleMesh()
        # vis_mesh.vertices = o3d.utility.Vector3dVector(merged_mesh.vertices)
        # vis_mesh.triangles = o3d.utility.Vector3iVector(merged_mesh.faces)
        # o3d.visualization.draw_geometries(geometry_list = [vis_mesh], width = 640, height = 480, window_name = "mesh_with_hit_points", point_show_normal=False) 
            
        # print('merged_mesh')
        # print(mesh.vertices.shape)
        # print(self.env_map_mesh.vertices.shape)
        # print(merged_mesh.vertices.shape)
        # input()
        # input()
        scene = Scene(merged_mesh)
        mesh_vertices_tensor = torch.from_numpy(merged_mesh.vertices).float().cuda()
        mesh_faces_tensor = torch.from_numpy(merged_mesh.faces).long().cuda()
        mesh_vertex_faces_tensor = torch.from_numpy(np.array(merged_mesh.vertex_faces)).long().cuda()
        mesh_face_normals, mesh_vertex_normals = calculate_vertex_normal(mesh_vertices_tensor, mesh_faces_tensor, mesh_vertex_faces_tensor)
        # torch.cuda.synchronize()
        # end = time.time()
        # print("mesh_ray_tracing time: {0} s".format(end - start))
        vis_debug = False 
        if vis_debug:
            vis_mesh = o3d.geometry.TriangleMesh()
            vis_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces_tensor.cpu().detach().numpy())
            vis_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
            vis_pcd = o3d.geometry.PointCloud()
            vis_pcd.points = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
            vis_pcd.normals = o3d.utility.Vector3dVector(mesh_vertex_normals.cpu().detach().numpy())
            o3d.visualization.draw_geometries(geometry_list = [vis_mesh, vis_pcd], width = 640, height = 480, window_name = "mesh_with_hit_points_2", point_show_normal=False) 

        # torch.autograd.set_detect_anomaly(True)
        num_pixels, _ = ray_directions.shape
        # print("num_pixels: {0}".format(num_pixels))

        total_vis_pcd = []
        face_idx_list = []
        total_index_ray = torch.arange(ray_origins.shape[0]).cuda()
        refract_ray_origins = ray_origins.clone()
        refract_ray_directions = ray_directions.clone()
        for ray_iter in range(3):

            # start = time.time() 
            points, index_ray, face_idx, points_normals, _ = self.ray_triangle_intersec_trimesh_optix(merged_mesh, scene, refract_ray_origins, refract_ray_directions, mesh_vertex_normals, None)
            # torch.cuda.synchronize()
            # end = time.time()
            # print("ray_triangle_intersec_trimesh time: {0} s".format(end - start))

            # start = time.time()
            # self.scene = Scene(merged_mesh)
            # faces_ind, hitted = self.scene.optix_intersect(Ray(refract_ray_origins, refract_ray_directions))
            # # faces = self.faces[faces_ind[hitted]]
            # # triangles = self.vertices[faces]
            # # normals = self.normals[faces]
            # # ray_hitted = ray.select(hitted)
            # torch.cuda.synchronize()
            # end = time.time()
            # print("optix_intersect time: {0} s".format(end - start))
            # print(index_ray)
            # print(index_ray.shape)
            # print(points.shape)
            # print(face_idx.shape)
            # input()

            # print(ray_origins.shape)
            # print(refract_ray_origins.shape)
            # print(refract_ray_directions.shape)
            # input()
            if ray_iter == 2:
                # points = points[face_idx >= mesh_face_num]
                # points_normals = points_normals[face_idx >= mesh_face_num]
                index_ray = index_ray[face_idx >= mesh_face_num]
                face_idx = face_idx[face_idx >= mesh_face_num]
            else:
                refract_ray_origins = refract_ray_origins[index_ray]
                refract_ray_directions = refract_ray_directions[index_ray]
                ray_surf_angle = torch.abs(torch.bmm(points_normals[:, None, :], refract_ray_directions[:, :, None]).squeeze())
                ray_length = torch.norm(points - refract_ray_origins, p=2, dim=1)
                # ray_mask = (ray_length > 0.05) & (ray_surf_angle > 0.2) # for cow
                # ray_mask = (ray_length > 0.0) & (ray_surf_angle > 0.5) # for hand
                ray_mask = (ray_length > 0.0) & (ray_surf_angle > 0.2)
                # ray_mask = (ray_length > 0.0) # for DRT
                ray_mask = ray_mask & (face_idx < mesh_face_num)

                points = points[ray_mask]
                points_normals = points_normals[ray_mask]
                index_ray = index_ray[ray_mask]
                face_idx = face_idx[ray_mask]
                refract_ray_directions = refract_ray_directions[ray_mask]

                # print(index_ray.shape)
                # print(points.shape)
                # print(face_idx.shape)
                # input()

            if vis_debug:
                vis_pcd = o3d.geometry.PointCloud()
                vis_pcd.points = o3d.utility.Vector3dVector(points.cpu().detach().numpy())
                vis_pcd.normals = o3d.utility.Vector3dVector(points_normals.cpu().detach().numpy())
                colors = np.array([[0, 1 - ray_iter * 0.4, ray_iter * 0.4] for i in range(len(vis_pcd.points))])
                vis_pcd.colors = o3d.utility.Vector3dVector(colors[:, :])
                total_vis_pcd.append(vis_pcd)

            #valid_ray_mask = torch.ones(num_pixels).bool().cuda()
            total_index_ray = total_index_ray[index_ray]

            temp_face_idx = torch.zeros(num_pixels).long().cuda()
            temp_face_idx[total_index_ray] = face_idx
            face_idx_list.append(temp_face_idx)
            if ray_iter == 2:
                break
            
            if ray_iter % 2 == 0:
                refract_ray_directions, _, _ = self.refraction(refract_ray_directions, points_normals, eta1 = 1.0003, eta2 = material_ior)
            else:
                refract_ray_directions, _, _ = self.refraction(refract_ray_directions, -points_normals, eta1 = material_ior, eta2 = 1.0003)

            refract_ray_directions = refract_ray_directions / torch.sqrt(torch.clamp(torch.sum(refract_ray_directions.detach()*refract_ray_directions.detach(), dim=1), min=1e-10 ).unsqueeze(1))
            refract_ray_origins = points + refract_ray_directions * 0.01

        for i in range(len(face_idx_list)):
            face_idx_list[i] = face_idx_list[i][total_index_ray].unsqueeze(0)
        face_idx_tensor = torch.cat(face_idx_list, dim=0)

        if vis_debug:
            # vis_mesh = o3d.geometry.TriangleMesh()
            # vis_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces_tensor.cpu().detach().numpy())
            # vis_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
            # total_vis_pcd.append(vis_mesh)
            o3d.visualization.draw_geometries(geometry_list = total_vis_pcd, width = 640, height = 480, window_name = "mesh_with_hit_points", point_show_normal=False) 

        return total_index_ray.detach(), \
            face_idx_tensor.detach()

    def mesh_ray_tracing_trimesh(self,
                                 sdf_mesh,
                                 calculate_vertex_normal,
                                 ray_origins,
                                 ray_directions,
                                 only_mask_loss,
                                 material_ior):

        sdf_mesh_face_num = sdf_mesh.faces.shape[0]
        merged_mesh = self.merge_mesh([sdf_mesh, self.env_map_mesh])
        mesh_vertices_tensor = torch.from_numpy(merged_mesh.vertices).float().cuda()
        mesh_faces_tensor = torch.from_numpy(merged_mesh.faces).long().cuda()
        mesh_vertex_faces_tensor = torch.from_numpy(np.array(merged_mesh.vertex_faces)).long().cuda()
        mesh_face_normals, mesh_vertex_normals = calculate_vertex_normal(mesh_vertices_tensor, mesh_faces_tensor, mesh_vertex_faces_tensor)
        
        vis_debug = False
        if vis_debug:
            vis_mesh = o3d.geometry.TriangleMesh()
            vis_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces_tensor.cpu().detach().numpy())
            vis_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
            vis_pcd = o3d.geometry.PointCloud()
            vis_pcd.points = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
            vis_pcd.normals = o3d.utility.Vector3dVector(mesh_vertex_normals.cpu().detach().numpy())
            o3d.visualization.draw_geometries(geometry_list = [vis_mesh, vis_pcd], width = 640, height = 480, window_name = "mesh_with_hit_points_2", point_show_normal=False) 

        # torch.autograd.set_detect_anomaly(True)
        num_pixels, _ = ray_directions.shape
        # print("num_pixels: {0}".format(num_pixels))

        reflect_ray_directions = torch.zeros_like(ray_directions).cuda()
        refract_ray_directions = torch.zeros_like(ray_directions).cuda()
        reflect_attenuate = torch.zeros(num_pixels).cuda()
        refract_attenuate = torch.zeros(num_pixels).cuda()
        accu_refract_attenuate = torch.ones(num_pixels).cuda()
        valid_ray_mask = torch.ones(num_pixels).bool().cuda()
        hit_points = ray_origins.clone()
        hit_normals = ray_directions.clone()

        total_vis_pcd = []
        face_idx_list = []
        ray_origins_list = []
        ray_directions_list = []

        for ray_iter in range(3):
            hit_mask = torch.zeros(num_pixels).bool().cuda()      

            points, index_ray, face_idx, points_normals, _ = self.ray_triangle_intersec_trimesh(merged_mesh, ray_origins, ray_directions, mesh_vertex_normals, None)
            # print(points_normals.shape)
            # print(ray_directions[index_ray].shape)
            if ray_iter == 2:
                points = points[face_idx >= sdf_mesh_face_num]
                points_normals = points_normals[face_idx >= sdf_mesh_face_num]
                index_ray = index_ray[face_idx >= sdf_mesh_face_num]
                face_idx = face_idx[face_idx >= sdf_mesh_face_num]
            else:
                first_hit_index_ray = index_ray[face_idx < sdf_mesh_face_num]
                ray_surf_angle = torch.abs(torch.bmm(points_normals[:, None, :], ray_directions[index_ray][:, :, None]).squeeze())
                ray_length = torch.norm(points - ray_origins[index_ray], p=2, dim=1)
                ray_mask = (ray_length > 0.0) & (ray_surf_angle > 0.2)
                # ray_mask = (ray_length > 0.05) & (ray_surf_angle > 0.2) # for cow
                # ray_mask = (ray_length > 0.0) & (ray_surf_angle > 0.5) # for hand
                # ray_mask = (ray_length > 0.0) # for DRT
                points = points[ray_mask]
                points_normals = points_normals[ray_mask]
                index_ray = index_ray[ray_mask]
                face_idx = face_idx[ray_mask]

                points = points[face_idx < sdf_mesh_face_num]
                points_normals = points_normals[face_idx < sdf_mesh_face_num]
                index_ray = index_ray[face_idx < sdf_mesh_face_num]
                face_idx = face_idx[face_idx < sdf_mesh_face_num]
            hit_mask[index_ray] = True
            if ray_iter == 0:
                first_hit_mask = torch.zeros_like(hit_mask).bool().cuda()
                first_hit_mask[first_hit_index_ray] = True
                if only_mask_loss:
                    if vis_debug:
                        valid_ray_mask = valid_ray_mask & hit_mask
                        hit_points[valid_ray_mask, :] = points
                        hit_normals[valid_ray_mask, :] = points_normals
                        vis_pcd = o3d.geometry.PointCloud()
                        vis_pcd.points = o3d.utility.Vector3dVector(hit_points[valid_ray_mask, :].cpu().detach().numpy())
                        vis_pcd.normals = o3d.utility.Vector3dVector(hit_normals[valid_ray_mask, :].cpu().detach().numpy())
                        colors = np.array([[0, 1 - ray_iter * 0.3, ray_iter * 0.3] for i in range(len(vis_pcd.points))])
                        vis_pcd.colors = o3d.utility.Vector3dVector(colors[:, :])
                        total_vis_pcd.append(vis_pcd)
                        o3d.visualization.draw_geometries(geometry_list = total_vis_pcd, width = 640, height = 480, window_name = "mesh_with_hit_points", point_show_normal=False) 
                    return None, first_hit_mask.detach(), None, None, None, None, None, None
            valid_ray_mask = valid_ray_mask & hit_mask

            # print("valid_ray num: {0}".format(torch.sum(valid_ray_mask).cpu().item()))
            # print(points.shape)
            # print(face_idx.shape)
            hit_points[valid_ray_mask, :] = points
            hit_normals[valid_ray_mask, :] = points_normals

            temp_face_idx = torch.zeros(num_pixels).long().cuda()
            temp_ray_originals = torch.zeros(num_pixels, 3).float().cuda()
            temp_ray_directions = torch.zeros(num_pixels, 3).float().cuda()
            temp_face_idx[valid_ray_mask] = face_idx
            face_idx_list.append(temp_face_idx)
            ray_origins_list.append(ray_origins)
            ray_directions_list.append(ray_directions)
            if ray_iter == 2:
                break
            
            if vis_debug:
                vis_pcd = o3d.geometry.PointCloud()
                vis_pcd.points = o3d.utility.Vector3dVector(hit_points[valid_ray_mask, :].cpu().detach().numpy())
                vis_pcd.normals = o3d.utility.Vector3dVector(hit_normals[valid_ray_mask, :].cpu().detach().numpy())
                colors = np.array([[0, 1 - ray_iter * 0.4, ray_iter * 0.4] for i in range(len(vis_pcd.points))])
                vis_pcd.colors = o3d.utility.Vector3dVector(colors[:, :])
                total_vis_pcd.append(vis_pcd)

            # hit_normals = self.calculate_sdf_normal(sdf, hit_points, retain_graph=False) 
                
            if ray_iter % 2 == 0:
                refract_ray_directions[valid_ray_mask], _, total_reflect_mask = self.refraction(ray_directions[valid_ray_mask], hit_normals[valid_ray_mask], eta1 = 1.0003, eta2 = material_ior)
            else:
                refract_ray_directions[valid_ray_mask], _, total_reflect_mask = self.refraction(ray_directions[valid_ray_mask], -hit_normals[valid_ray_mask], eta1 = material_ior, eta2 = 1.0003)
            accu_refract_attenuate[valid_ray_mask] *= (1 - refract_attenuate[valid_ray_mask])
            
            # valid_ray_mask[valid_ray_mask][total_reflect_mask] = False
            
            if ray_iter == 0:
                reflect_ray_directions[valid_ray_mask] = self.reflection(-ray_directions[valid_ray_mask], hit_normals[valid_ray_mask])
                reflect_attenuate = refract_attenuate

            ray_directions = refract_ray_directions
            ray_directions = ray_directions / torch.sqrt(torch.clamp(torch.sum(ray_directions.detach()*ray_directions.detach(), dim=1), min=1e-10 ).unsqueeze(1))
            ray_origins = hit_points + ray_directions * torch.ones(num_pixels, 1).cuda() * 0.01
            ray_directions[~valid_ray_mask] = torch.zeros(3).cuda()

        for i in range(len(face_idx_list)):
            face_idx_list[i] = face_idx_list[i][valid_ray_mask].unsqueeze(0)
            ray_origins_list[i] = ray_origins_list[i][valid_ray_mask].unsqueeze(0)
            ray_directions_list[i] = ray_directions_list[i][valid_ray_mask].unsqueeze(0)
        face_idx_tensor = torch.cat(face_idx_list, dim=0)
        ray_origins_tensor = torch.cat(ray_origins_list, dim=0)
        ray_directions_tensor = torch.cat(ray_directions_list, dim=0)
        # print(face_idx_tensor.shape)
        # print(ray_origins_tensor.shape)
        # print(ray_directions_tensor.shape)

        if vis_debug:
            vis_mesh = o3d.geometry.TriangleMesh()
            vis_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces_tensor.cpu().detach().numpy())
            vis_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
            total_vis_pcd.append(vis_mesh)
            o3d.visualization.draw_geometries(geometry_list = total_vis_pcd, width = 640, height = 480, window_name = "mesh_with_hit_points", point_show_normal=False) 

        return valid_ray_mask.detach(), \
            face_idx_tensor.detach()
        # print("accu_refract_attenuate[valid_ray_mask]:")
        # print(accu_refract_attenuate[valid_ray_mask])
        # return valid_ray_mask.detach(), \
        #     first_hit_mask.detach(), \
        #     face_idx_tensor.detach(), \
        #     ray_origins_tensor.detach(), \
        #     ray_directions_tensor.detach(), \
        #     reflect_ray_directions[valid_ray_mask].detach(), \
        #     accu_refract_attenuate[valid_ray_mask].squeeze().detach(), \
        #     reflect_attenuate[valid_ray_mask].squeeze().detach()

    def find_minimum_sdf_points(self,
                                sdf,
                                ray_origins,
                                ray_directions,
                                in_mask):

        num_pixels, _ = ray_directions.shape
        ray_origins = ray_origins.unsqueeze(0)
        ray_directions = ray_directions.unsqueeze(0)

        sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(ray_origins, ray_directions, r=self.object_bounding_sphere)
        
        out_mask = torch.ones_like(in_mask).bool().cuda()
        curr_start_points = torch.zeros(num_pixels, 3).cuda()
        acc_start_dis = torch.zeros(num_pixels).cuda()

        min_mask_points, min_mask_dist, min_mask_vals = self.minimal_sdf_points(num_pixels, sdf, ray_origins, ray_directions.squeeze(0), out_mask, sphere_intersections[:, :, 0].squeeze(0), sphere_intersections[:, :, 1].squeeze(0))
        out_mask[out_mask][min_mask_vals < 0.0] = False
        curr_start_points[out_mask] = min_mask_points
        acc_start_dis[out_mask] = min_mask_dist

        return curr_start_points.squeeze(), \
               acc_start_dis.squeeze(), \
               ~out_mask

    def idr_sphere_tracing(self, batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections):
        ''' Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection '''

        sphere_intersections_points = cam_loc.reshape(batch_size, 1, 1, 3) + sphere_intersections.unsqueeze(-1) * ray_directions.unsqueeze(2)
        unfinished_mask_start = mask_intersect.reshape(-1).clone()
        unfinished_mask_end = mask_intersect.reshape(-1).clone()

        # Initialize start current points
        curr_start_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[:,:,0,:].reshape(-1,3)[unfinished_mask_start]
        acc_start_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_start_dis[unfinished_mask_start] = sphere_intersections.reshape(-1,2)[unfinished_mask_start,0]

        # Initialize end current points
        curr_end_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_end_points[unfinished_mask_end] = sphere_intersections_points[:,:,1,:].reshape(-1,3)[unfinished_mask_end]
        acc_end_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_end_dis[unfinished_mask_end] = sphere_intersections.reshape(-1,2)[unfinished_mask_end,1]

        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

        next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
        next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])

        while True:
            # Update sdf
            curr_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            curr_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold)
            unfinished_mask_end = unfinished_mask_end & (curr_sdf_end > self.sdf_threshold)

            if (unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0) or iters == self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            acc_start_dis = acc_start_dis + curr_sdf_start
            acc_end_dis = acc_end_dis - curr_sdf_end

            # Update points
            curr_start_points = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)
            curr_end_points = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)

            # Fix points which wrongly crossed the surface
            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

            next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])

            not_projected_start = next_sdf_start < 0
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            while (not_projected_start.sum() > 0 or not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (cam_loc.unsqueeze(1) + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_start]

                acc_end_dis[not_projected_end] += ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_end[not_projected_end]
                curr_end_points[not_projected_end] = (cam_loc.unsqueeze(1) + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_end]

                # Calc sdf
                next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start])
                next_sdf_end[not_projected_end] = sdf(curr_end_points[not_projected_end])

                # Update mask
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis)
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)

        return curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis

    def idr_minimal_sdf_points(self, num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis):

        n_mask_points = mask.sum()

        n = self.n_steps
        # steps = torch.linspace(0.0, 1.0,n).cuda()
        steps = torch.empty(n).uniform_(0.0, 1.0).cuda()
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis

        mask_points = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask]
        mask_rays = ray_directions[mask, :]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(
            1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        mask_sdf_all = []
        for pnts in torch.split(points, 100000, dim=0):
            mask_sdf_all.append(sdf(pnts))

        mask_sdf_all = torch.cat(mask_sdf_all).reshape(-1, n)
        min_vals, min_idx = mask_sdf_all.min(-1)
        min_mask_points = mask_points_all.reshape(-1, n, 3)[torch.arange(0, n_mask_points), min_idx]
        min_mask_dist = steps.reshape(-1, n)[torch.arange(0, n_mask_points), min_idx]

        return min_mask_points, min_mask_dist
    
    def sdf_ray_tracing(self,
                sdf,
                cam_loc,
                object_mask,
                ray_directions,
                mesh_vertices_tensor
                ):

        batch_size, num_pixels, _ = ray_directions.shape

        sphere_intersections, mask_intersect = rend_util.idr_get_sphere_intersection(cam_loc, ray_directions, r=1.0)
        
        # min_max_t = sphere_intersections
        # line_points_1 = cam_loc.unsqueeze(1) + min_max_t[:,:,0].unsqueeze(2) * ray_directions
        # line_points_2 = cam_loc.unsqueeze(1) + min_max_t[:,:,1].unsqueeze(2) * ray_directions
        # # line_points_1 = cam_loc.unsqueeze(1) + 0.001 * ray_directions
        # # line_points_2 = cam_loc.unsqueeze(1) + 1.0 * ray_directions
        # print(line_points_1.shape)
        # print(line_points_2.shape)
        # line_points = torch.stack((line_points_1,line_points_2), 2)
        # print(line_points)
        # line_points = line_points.reshape(-1, 3)
        # lines =  np.arange(0, 80000).reshape(-1, 2)
        # colors = np.array([[0, 1, 0] for i in range(len(lines))])
        # line_set = o3d.geometry.LineSet()
        # line_set.points = o3d.utility.Vector3dVector(line_points.cpu().detach().numpy())
        # line_set.lines = o3d.utility.Vector2iVector(lines[::1, :])
        # line_set.colors = o3d.utility.Vector3dVector(colors[::1, :])
        # vis_pcd_2 = o3d.geometry.PointCloud()
        # vis_pcd_2.points = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
        # o3d.visualization.draw_geometries(geometry_list = [vis_pcd_2,line_set], width = 640, height = 480, window_name = "rays")

        curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis = \
            self.idr_sphere_tracing(batch_size, num_pixels, sdf, cam_loc, ray_directions, mask_intersect, sphere_intersections)

        network_object_mask = (acc_start_dis < acc_end_dis)

        # The non convergent rays should be handled by the sampler
        sampler_mask = unfinished_mask_start
        sampler_net_obj_mask = torch.zeros_like(sampler_mask).bool().cuda()
        if sampler_mask.sum() > 0:
            sampler_min_max = torch.zeros((batch_size, num_pixels, 2)).cuda()
            sampler_min_max.reshape(-1, 2)[sampler_mask, 0] = acc_start_dis[sampler_mask]
            sampler_min_max.reshape(-1, 2)[sampler_mask, 1] = acc_end_dis[sampler_mask]

            sampler_pts, sampler_net_obj_mask, sampler_dists = self.idr_ray_sampler(sdf,
                                                                                cam_loc,
                                                                                object_mask,
                                                                                ray_directions,
                                                                                sampler_min_max,
                                                                                sampler_mask
                                                                                )

            curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
            acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
            network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]

        print('----------------------------------------------------------------')
        print('RayTracing: object = {0}/{1}, secant on {2}/{3}.'
              .format(network_object_mask.sum(), len(network_object_mask), sampler_net_obj_mask.sum(), sampler_mask.sum()))
        print('----------------------------------------------------------------')

        if not self.training:
            return curr_start_points, \
                   network_object_mask, \
                   acc_start_dis

        ray_directions = ray_directions.reshape(-1, 3)
        mask_intersect = mask_intersect.reshape(-1)

        in_mask = ~network_object_mask & object_mask & ~sampler_mask
        out_mask = ~object_mask & ~sampler_mask

        mask_left_out = (in_mask | out_mask) & ~mask_intersect
        if mask_left_out.sum() > 0:  # project the origin to the not intersect points on the sphere
            cam_left_out = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask_left_out]
            rays_left_out = ray_directions[mask_left_out]
            acc_start_dis[mask_left_out] = -torch.bmm(rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)).squeeze()
            curr_start_points[mask_left_out] = cam_left_out + acc_start_dis[mask_left_out].unsqueeze(1) * rays_left_out

        mask = (in_mask | out_mask) & mask_intersect

        if mask.sum() > 0:
            min_dis[network_object_mask & out_mask] = acc_start_dis[network_object_mask & out_mask]

            min_mask_points, min_mask_dist = self.idr_minimal_sdf_points(num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis)

            curr_start_points[mask] = min_mask_points
            acc_start_dis[mask] = min_mask_dist

        return curr_start_points, \
               network_object_mask, \
               acc_start_dis

    def hehe(self,
                sdf,
                sample_network,
                cam_loc,
                object_mask,
                ray_directions
                ):

        # torch.autograd.set_detect_anomaly(True)

        with torch.no_grad():
            # ray_directions [1, num_pixels, 3]
            # cam_loc [1, 3]
            batch_size, num_pixels, _ = ray_directions.shape
            original_ray_directions = ray_directions.clone()
            # origin_points [1, num_pixels, 3]
            # min_max_t [1, num_pixels, 2] 
            # cam_loc [1, num_pixels, 3]
            origin_points = cam_loc[:, None, :].expand(-1, num_pixels, -1).detach()
            # print(origin_points.shape)
            # print(object_mask.shape)
            # input()

            # sphere_intersections [1, num_pixels, 2]
            # mask_intersect [1, num_pixels]
            sphere_intersections, mask_intersect = rend_util.get_sphere_intersection(origin_points, ray_directions, r=self.object_bounding_sphere)
            min_max_t = sphere_intersections
            # print("mask_intersect: ", torch.sum(mask_intersect).cpu().item())
            # input()

            in_out_indicator = torch.ones(1).cuda()

            # mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=self.object_bounding_sphere)
            # mesh_sphere.compute_vertex_normals()
            # mesh_sphere.paint_uniform_color([0.9, 0.1, 0.1])
            # surface_traces = plt.get_surface_trace(path="./",
            #                                        epoch=0,
            #                                        sdf=sdf,
            #                                        resolution=100)
            # mesh = o3d.io.read_triangle_mesh("./surface_0.ply")
            # mesh.compute_vertex_normals()
            # o3d.visualization.draw_geometries(geometry_list = [mesh], width = 640, height = 480, window_name = "rays")

            # vis_sphere_intersections_front = o3d.geometry.PointCloud()
            # vis_sphere_intersections_back = o3d.geometry.PointCloud()
            # hit_points_front = origin_points.squeeze(0) + sphere_intersections.squeeze(0)[:, 0].unsqueeze(-1) * ray_directions.squeeze(0)
            # hit_points_back = origin_points.squeeze(0) + sphere_intersections.squeeze(0)[:, 1].unsqueeze(-1) * ray_directions.squeeze(0)
            # vis_sphere_intersections_front.points = o3d.utility.Vector3dVector(hit_points_front[mask_intersect.squeeze(0), :].cpu().detach().numpy())
            # vis_sphere_intersections_back.points = o3d.utility.Vector3dVector(hit_points_back[mask_intersect.squeeze(0), :].cpu().detach().numpy())
            # o3d.visualization.draw_geometries(geometry_list = [vis_sphere_intersections_back], width = 640, height = 480, window_name = "vis_sphere_intersections")

            # surface_traces = plt.get_surface_trace(path="./",
            #                                        epoch=0,
            #                                        sdf=sdf,
            #                                        resolution=100)
            # mesh = o3d.io.read_triangle_mesh("./surface_0.ply")
            # mesh.compute_vertex_normals()
            # mesh_box = o3d.geometry.TriangleMesh.create_box(width=1,
            #                                     height=1,
            #                                     depth=1)
            # mesh_box.compute_vertex_normals()
            # mesh_box.paint_uniform_color([0.9, 0.1, 0.1])
            
            # line_points = origin_points.unsqueeze(-2) + min_max_t.unsqueeze(-1) * ray_directions.unsqueeze(-2)
            # line_points[:, :, 0, :] = cam_loc.unsqueeze(0).repeat(1, num_pixels, 1) #[1, 2048, 3]
            # line_points = line_points.reshape(-1, 3)
            # lines =  np.arange(0, 4096).reshape(-1, 2)
            # # colors = np.array([[1, 0, 0] for i in range(len(lines))])
            # line_set = o3d.geometry.LineSet()
            # line_set.points = o3d.utility.Vector3dVector(line_points.cpu().detach().numpy())
            # line_set.lines = o3d.utility.Vector2iVector(lines[::40, :])
            # line_set.colors = o3d.utility.Vector3dVector(colors[::40, :])
            # o3d.visualization.draw_geometries(geometry_list = [mesh_box, line_set], width = 640, height = 480, window_name = "rays")

            curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis = \
                self.sphere_tracing(batch_size, num_pixels, sdf, \
                ray_directions, mask_intersect, \
                origin_points, min_max_t, in_out_indicator)
            original_min_max_t = min_max_t.clone()
            # print(acc_start_dis.shape)
            # print(acc_end_dis.shape)
            # input()
            temp_mask = acc_start_dis < acc_end_dis
            min_max_t[:, :, 0][:, temp_mask] = acc_start_dis[None, :][:, temp_mask] 
            min_max_t[:, :, 1][:, temp_mask] = acc_end_dis[None, :][:, temp_mask] 
            min_max_t[:, :, 0] = min_max_t[:, :, 0] - torch.ones(1, num_pixels).cuda() * 0.01 #[1, 2048, 3]
            min_max_t[:, :, 1] = min_max_t[:, :, 1] + torch.ones(1, num_pixels).cuda() * 0.01 #[1, 2048, 3]

            total_hit_points = torch.zeros(batch_size, num_pixels, 4, 3).cuda()
            total_hit_points_mask = torch.zeros(batch_size, num_pixels, 4).bool().cuda()
            self.reflect_attenuate = torch.zeros(num_pixels, 1).float().cuda()
            self.refract_attenuate = torch.zeros(num_pixels, 1).float().cuda()
            self.accu_refract_attenuate = torch.ones(num_pixels, 1).float().cuda()
            composite_img = torch.zeros(num_pixels, 3).cuda()
            refract_img = torch.zeros(num_pixels, 3).cuda()
            reflect_img = torch.zeros(num_pixels, 3).cuda()
            self.first_time_mask_intersect = torch.zeros(num_pixels).bool().cuda()

        reflect_ray_directions = torch.zeros(num_pixels, 3).cuda()
        #ray_directions， hit_points, hit_points_dist
        refract_ray_directions = ray_directions.squeeze()

        total_vis_pcd = []

        ##############
        # with torch.no_grad():
        #     origin_points_0 = origin_points
        #     ray_directions_0 = ray_directions
        #     min_max_t_0 = min_max_t
        #     hit_points_0, hit_points_dist_0, hit_mask_0 = self.ray_tracing_using_sampler(sdf, origin_points_0, ray_directions_0, min_max_t_0, object_mask, mask_intersect.squeeze(0), in_out_indicator)
        #     mask_intersect[:, ~hit_mask_0] = False

        # hit_points_0 = (origin_points_0 + hit_points_dist_0[None, :, None] * ray_directions_0).squeeze(0)
        # # x [2048, 3], y [2048]
        # # normals [2048, 3]
        # x = hit_points_0.detach()
        # x.requires_grad_(True)
        # y = sdf(x)
        # d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        # normals_0 = torch.autograd.grad(
        #     outputs=y,
        #     inputs=x,
        #     grad_outputs=d_output,
        #     create_graph=True,
        #     retain_graph=True,
        #     only_inputs=True)[0]
        # normals_0 = normals_0 / torch.sqrt(torch.clamp(torch.sum(normals_0.detach()*normals_0.detach(), dim=1), min=1e-10 ).unsqueeze(1) )
        # normals_0 = normals_0.detach()

        # surface_output_0 = sdf(hit_points_0)
        # surface_sdf_values_0 = sdf(hit_points_0).detach()
        # differentiable_surface_points = sample_network(surface_output_0[:, None],
        #                                                surface_sdf_values_0[:, None],
        #                                                normals_0.detach(),
        #                                                hit_points_dist_0[:, None].detach(),
        #                                                origin_points_0.squeeze(0).detach(),
        #                                                ray_directions_0.squeeze(0).detach())
        # hit_points_0 = differentiable_surface_points

        # reflect_ray_directions[hit_mask_0] = self.reflection(-ray_directions.squeeze(0)[hit_mask_0], normals_0[hit_mask_0])
        # refract_ray_directions[hit_mask_0], self.reflect_attenuate[hit_mask_0, :], total_reflect_mask = self.refraction(ray_directions.squeeze(0)[hit_mask_0], normals_0[hit_mask_0], eta1 = 1.0003, eta2 = 1.52)
        # with torch.no_grad():
        #     self.accu_refract_attenuate[hit_mask_0, :] *= (1 - self.reflect_attenuate[hit_mask_0, :])
        #     mask_intersect[:, hit_mask_0][:, total_reflect_mask] = False

        # with torch.no_grad():
        #     origin_points_1 = hit_points_0
        #     ray_directions_1 = reflect_ray_directions
        #     min_max_t_1 = min_max_t.clone()
        #     min_max_t_1[:, :, 0] = torch.ones(1, num_pixels) * 0.01 #[1, 2048, 3]
        #     min_max_t_1[:, :, 1] = torch.ones(1, num_pixels) * 1.0 #[1, 2048, 3]
        #     hit_points_1, hit_points_dist_1, hit_mask_1 = self.ray_tracing_using_sampler(sdf, origin_points_1, ray_directions_1, min_max_t_1, object_mask, mask_intersect.squeeze(0), in_out_indicator)
        #     mask_intersect[:, ~hit_mask_1] = False


        #     # hit_points = torch.where(hit_points < 2, hit_points, torch.zeros_like(hit_points))
        #     # hit_points = torch.where(hit_points > -2, hit_points, torch.zeros_like(hit_points))
        #     # hit_points = torch.where(~torch.isnan(hit_points), hit_points, torch.zeros_like(hit_points))
        #     # vis_pcd = o3d.geometry.PointCloud()
        #     # vis_pcd.points = o3d.utility.Vector3dVector(hit_points[hit_mask].cpu().detach().numpy())
        #     # vis_pcd.normals = o3d.utility.Vector3dVector(ray_directions.squeeze(0)[hit_mask].cpu().detach().numpy())
        #     # colors = np.array([[0, 1 - ray_iter * 0.8, ray_iter * 0.8] for i in range(len(vis_pcd.points))])
        #     # vis_pcd.colors = o3d.utility.Vector3dVector(colors[:, :])
        #     # total_vis_pcd.append(vis_pcd)
        #     # o3d.io.write_point_cloud("vis_pcd.ply", vis_pcd)
        #     # o3d.visualization.draw_geometries(geometry_list = [line_set, vis_pcd, mesh], width = 640, height = 480, window_name = "mesh_with_hit_points")
         
        #     # print(normals.requires_grad)
        #     # print(normals.grad_fn)
        #     # input()
            
        #     if ray_iter == 0:
        #         # eta2 = 1.4723
        #         reflect_ray_directions[hit_mask] = self.reflection(-ray_directions.squeeze(0)[hit_mask], normals[hit_mask])
        #         refract_ray_directions[hit_mask], self.reflect_attenuate[hit_mask, :], total_reflect_mask = self.refraction(ray_directions.squeeze(0)[hit_mask], normals[hit_mask], eta1 = 1.0003, eta2 = 1.52)
        #         with torch.no_grad():
        #             self.accu_refract_attenuate[hit_mask, :] *= (1 - self.reflect_attenuate[hit_mask, :])
        #     if ray_iter > 0:
        #         if ray_iter % 2 == 0:
        #             refract_ray_directions[hit_mask], self.refract_attenuate[hit_mask, :], total_reflect_mask = self.refraction(ray_directions.squeeze(0)[hit_mask], normals[hit_mask], eta1 = 1.0003, eta2 = 1.52)
        #         else:
        #             refract_ray_directions[hit_mask], self.refract_attenuate[hit_mask, :], total_reflect_mask = self.refraction(ray_directions.squeeze(0)[hit_mask], -normals[hit_mask], eta1 = 1.52, eta2 = 1.0003)
        #         with torch.no_grad():
        #             self.accu_refract_attenuate[hit_mask, :] *= (1 - self.refract_attenuate[hit_mask, :])
        #     with torch.no_grad():
        #         mask_intersect[:, hit_mask][:, total_reflect_mask] = False

        #     # print(reflect_ray_directions.requires_grad)
        #     # print(reflect_ray_directions.grad_fn)
        #     # print(refract_ray_directions.requires_grad)
        #     # print(refract_ray_directions.grad_fn)
        #     # input()

        #     # min_max_t [1, 2048, 2] 
        #     # vis_min_max_t = torch.zeros_like(min_max_t)
        #     # vis_min_max_t[:, :, 0] = torch.zeros(1, num_pixels) #[1, 2048, 3]
        #     # vis_min_max_t[:, :, 1] = torch.ones(1, num_pixels) * 0.3 #[1, 2048, 3]
        #     # line_points = hit_points.unsqueeze(-2) + vis_min_max_t.unsqueeze(-1) * reflect_ray_directions.unsqueeze(-2)
        #     # line_points = line_points.reshape(-1, 3)
        #     # lines =  np.arange(0, num_pixels * 2).reshape(-1, 2)
        #     # colors = np.array([[0, 1, 0] for i in range(len(lines))])
        #     # reflect_line_set = o3d.geometry.LineSet()
        #     # reflect_line_set.points = o3d.utility.Vector3dVector(line_points.cpu().detach().numpy())
        #     # reflect_line_set.lines = o3d.utility.Vector2iVector(lines[::40, :])
        #     # reflect_line_set.colors = o3d.utility.Vector3dVector(colors[::40, :])
        #     # line_points = hit_points.unsqueeze(-2) + vis_min_max_t.unsqueeze(-1) * refract_ray_directions.unsqueeze(-2)
        #     # line_points = line_points.reshape(-1, 3)
        #     # lines =  np.arange(0, num_pixels * 2).reshape(-1, 2)
        #     # colors = np.array([[0, 0, 1] for i in range(len(lines))])
        #     # refract_line_set = o3d.geometry.LineSet()
        #     # refract_line_set.points = o3d.utility.Vector3dVector(line_points.cpu().detach().numpy())
        #     # refract_line_set.lines = o3d.utility.Vector2iVector(lines[::40, :])
        #     # refract_line_set.colors = o3d.utility.Vector3dVector(colors[::40, :])
        #     # line_points = hit_points.unsqueeze(-2) + vis_min_max_t.unsqueeze(-1) * normals.unsqueeze(-2)
        #     # line_points = line_points.reshape(-1, 3)
        #     # lines =  np.arange(0, num_pixels * 2).reshape(-1, 2)
        #     # colors = np.array([[0, 0, 1] for i in range(len(lines))])
        #     # normal_line_set = o3d.geometry.LineSet()
        #     # normal_line_set.points = o3d.utility.Vector3dVector(line_points.cpu().detach().numpy())
        #     # normal_line_set.lines = o3d.utility.Vector2iVector(lines[::40, :])
        #     # normal_line_set.colors = o3d.utility.Vector3dVector(colors[::40, :])
        #     # total_vis_pcd.append(vis_sphere_intersections_back)
        #     # o3d.visualization.draw_geometries(geometry_list = total_vis_pcd, width = 640, height = 480, window_name = "mesh_with_hit_points")

        #     origin_points = hit_points.unsqueeze(0)
        #     ray_directions = refract_ray_directions.unsqueeze(0)
        #     ray_directions = ray_directions / torch.sqrt(torch.clamp(torch.sum(ray_directions.detach()*ray_directions.detach(), dim=2), min=1e-10 ).unsqueeze(2))
        #     with torch.no_grad():
        #         min_max_t[:, :, 0] = torch.ones(1, num_pixels) * 0.01 #[1, 2048, 3]
        #         min_max_t[:, :, 1] = torch.ones(1, num_pixels) * 1.0 #[1, 2048, 3]
          
        #     continue
        ##############

        for ray_iter in range(4):
            with torch.no_grad():

                self.refract_attenuate.zero_()
                
                temp_mask_intersect = mask_intersect.clone()
                hit_points, hit_points_dist, hit_mask = self.ray_tracing_using_sampler(sdf, origin_points, ray_directions, min_max_t, object_mask, mask_intersect.squeeze(0), in_out_indicator)

                mask_intersect[:, ~hit_mask] = False
                # cv2.imshow("mask_intersect", mask_intersect.reshape(100, 100, 1).float().detach().cpu().numpy())
                # cv2.waitKey(1)

                # if torch.sum(mask_intersect).cpu().item() > 0:
                #     vis_pcd = o3d.geometry.PointCloud()
                #     vis_pcd.points = o3d.utility.Vector3dVector(hit_points[hit_mask].cpu().detach().numpy())
                #     colors = np.array([[0, 1 - ray_iter * 0.4, ray_iter * 0.8] for i in range(len(vis_pcd.points))])
                #     vis_pcd.colors = o3d.utility.Vector3dVector(colors[:, :])
                #     total_vis_pcd.append(vis_pcd)
                
                if ray_iter == 0:
                    self.first_time_mask_intersect = mask_intersect.clone()

                    # mask_intersect = mask_intersect.reshape(-1, 200, 1)
                    # mask_intersect = mask_intersect.cpu().detach().numpy()
                    # mask_intersect = mask_intersect.astype(float)
                    # cv2.imshow("mask_intersect", mask_intersect)
                    # cv2.waitKey(0)
     
                    # composite_img = (refract_img + reflect_img) * 100
                    # composite_img = np.power(composite_img, 1 / 2.2)
                    # cv2.imshow("composite_img", composite_img)
                    # cv2.imshow("refract_img", np.power(refract_img * 100, 1 / 2.2))
                    # cv2.imshow("reflect_img", np.power(reflect_img * 100, 1 / 2.2))

                    # break
                # print("{0}: valid point num: {1}".format(ray_iter, torch.sum(mask_intersect).cpu().item()))
                if torch.sum(mask_intersect).cpu().item() < 1:
                    mask_intersect = temp_mask_intersect
                    # print("output point num: ", torch.sum(mask_intersect).cpu().item())
                    break
                total_hit_points[:, hit_mask, ray_iter, :] = origin_points[:, hit_mask, :] 
                total_hit_points_mask[:, hit_mask, ray_iter] = mask_intersect[:, hit_mask] 

            # x [2048, 3], y [2048]
            # normals [2048, 3]
            # hit_points = (origin_points + hit_points_dist[None, :, None] * ray_directions).squeeze(0)
            x = hit_points.detach()
            x.requires_grad_(True)
            y = sdf(x)
            d_output = torch.ones_like(y, requires_grad=False, device=y.device)
            normals = torch.autograd.grad(
                outputs=y,
                inputs=x,
                grad_outputs=d_output,
                create_graph=True,
                retain_graph=True,
                only_inputs=True)[0]
            normals = normals / torch.sqrt(torch.clamp(torch.sum(normals.detach()*normals.detach(), dim=1), min=1e-10 ).unsqueeze(1) )
            normals = normals.detach()

            # make_dot(normals).render("attached", format="png")
            # input()
            # normals = normals.detach()

            hit_points = (origin_points + hit_points_dist[None, :, None].detach() * ray_directions.detach()).squeeze(0)
            surface_output = sdf(hit_points)
            surface_sdf_values = sdf(hit_points).detach()
            # hit_points_dist, origin_points, ray_directions
            # print(surface_output.shape)
            # print(surface_sdf_values.shape)
            # print(normals.shape)
            # print(hit_points_dist.shape)
            # print(origin_points.shape)
            # print(ray_directions.shape)
            # input()
            differentiable_surface_points = sample_network(surface_output[:, None],
                                                           surface_sdf_values[:, None],
                                                           normals.detach(),
                                                           hit_points_dist[:, None].detach(),
                                                           origin_points.squeeze(0).detach(),
                                                           ray_directions.squeeze(0).detach())
            hit_points = differentiable_surface_points

            # hit_points = torch.where(hit_points < 2, hit_points, torch.zeros_like(hit_points))
            # hit_points = torch.where(hit_points > -2, hit_points, torch.zeros_like(hit_points))
            # hit_points = torch.where(~torch.isnan(hit_points), hit_points, torch.zeros_like(hit_points))
            # vis_pcd = o3d.geometry.PointCloud()
            # vis_pcd.points = o3d.utility.Vector3dVector(hit_points[hit_mask].cpu().detach().numpy())
            # vis_pcd.normals = o3d.utility.Vector3dVector(ray_directions.squeeze(0)[hit_mask].cpu().detach().numpy())
            # colors = np.array([[0, 1 - ray_iter * 0.8, ray_iter * 0.8] for i in range(len(vis_pcd.points))])
            # vis_pcd.colors = o3d.utility.Vector3dVector(colors[:, :])
            # total_vis_pcd.append(vis_pcd)
            # o3d.io.write_point_cloud("vis_pcd.ply", vis_pcd)
            # o3d.visualization.draw_geometries(geometry_list = [line_set, vis_pcd, mesh], width = 640, height = 480, window_name = "mesh_with_hit_points")
         
            # print(normals.requires_grad)
            # print(normals.grad_fn)
            # input()
            
            if ray_iter == 0:
                # eta2 = 1.4723
                reflect_ray_directions[hit_mask] = self.reflection(-ray_directions.squeeze(0)[hit_mask], normals[hit_mask])
                refract_ray_directions[hit_mask], self.reflect_attenuate[hit_mask, :], total_reflect_mask = self.refraction(ray_directions.squeeze(0)[hit_mask], normals[hit_mask], eta1 = 1.0003, eta2 = 1.52)
                with torch.no_grad():
                    self.accu_refract_attenuate[hit_mask, :] *= (1 - self.reflect_attenuate[hit_mask, :])
            if ray_iter > 0:
                if ray_iter % 2 == 0:
                    refract_ray_directions[hit_mask], self.refract_attenuate[hit_mask, :], total_reflect_mask = self.refraction(ray_directions.squeeze(0)[hit_mask], normals[hit_mask], eta1 = 1.0003, eta2 = 1.52)
                else:
                    refract_ray_directions[hit_mask], self.refract_attenuate[hit_mask, :], total_reflect_mask = self.refraction(ray_directions.squeeze(0)[hit_mask], -normals[hit_mask], eta1 = 1.52, eta2 = 1.0003)
                with torch.no_grad():
                    self.accu_refract_attenuate[hit_mask, :] *= (1 - self.refract_attenuate[hit_mask, :])
            with torch.no_grad():
                mask_intersect[:, hit_mask][:, total_reflect_mask] = False

            # print(reflect_ray_directions.requires_grad)
            # print(reflect_ray_directions.grad_fn)
            # print(refract_ray_directions.requires_grad)
            # print(refract_ray_directions.grad_fn)
            # input()

            # min_max_t [1, 2048, 2] 
            # vis_min_max_t = torch.zeros_like(min_max_t)
            # vis_min_max_t[:, :, 0] = torch.zeros(1, num_pixels) #[1, 2048, 3]
            # vis_min_max_t[:, :, 1] = torch.ones(1, num_pixels) * 0.3 #[1, 2048, 3]
            # line_points = hit_points.unsqueeze(-2) + vis_min_max_t.unsqueeze(-1) * reflect_ray_directions.unsqueeze(-2)
            # line_points = line_points.reshape(-1, 3)
            # lines =  np.arange(0, num_pixels * 2).reshape(-1, 2)
            # colors = np.array([[0, 1, 0] for i in range(len(lines))])
            # reflect_line_set = o3d.geometry.LineSet()
            # reflect_line_set.points = o3d.utility.Vector3dVector(line_points.cpu().detach().numpy())
            # reflect_line_set.lines = o3d.utility.Vector2iVector(lines[::40, :])
            # reflect_line_set.colors = o3d.utility.Vector3dVector(colors[::40, :])
            # line_points = hit_points.unsqueeze(-2) + vis_min_max_t.unsqueeze(-1) * refract_ray_directions.unsqueeze(-2)
            # line_points = line_points.reshape(-1, 3)
            # lines =  np.arange(0, num_pixels * 2).reshape(-1, 2)
            # colors = np.array([[0, 0, 1] for i in range(len(lines))])
            # refract_line_set = o3d.geometry.LineSet()
            # refract_line_set.points = o3d.utility.Vector3dVector(line_points.cpu().detach().numpy())
            # refract_line_set.lines = o3d.utility.Vector2iVector(lines[::40, :])
            # refract_line_set.colors = o3d.utility.Vector3dVector(colors[::40, :])
            # line_points = hit_points.unsqueeze(-2) + vis_min_max_t.unsqueeze(-1) * normals.unsqueeze(-2)
            # line_points = line_points.reshape(-1, 3)
            # lines =  np.arange(0, num_pixels * 2).reshape(-1, 2)
            # colors = np.array([[0, 0, 1] for i in range(len(lines))])
            # normal_line_set = o3d.geometry.LineSet()
            # normal_line_set.points = o3d.utility.Vector3dVector(line_points.cpu().detach().numpy())
            # normal_line_set.lines = o3d.utility.Vector2iVector(lines[::40, :])
            # normal_line_set.colors = o3d.utility.Vector3dVector(colors[::40, :])
            # total_vis_pcd.append(vis_sphere_intersections_back)
            # o3d.visualization.draw_geometries(geometry_list = total_vis_pcd, width = 640, height = 480, window_name = "mesh_with_hit_points")

            origin_points = hit_points.unsqueeze(0)
            ray_directions = refract_ray_directions.unsqueeze(0)
            ray_directions = ray_directions / torch.sqrt(torch.clamp(torch.sum(ray_directions.detach()*ray_directions.detach(), dim=2), min=1e-10 ).unsqueeze(2))
            with torch.no_grad():
                min_max_t[:, :, 0] = torch.ones(1, num_pixels) * 0.01 #[1, 2048, 3]
                min_max_t[:, :, 1] = torch.ones(1, num_pixels) * 1.0 #[1, 2048, 3]
          
            continue
        
        # with torch.no_grad():
        #     sphere_intersections, _ = rend_util.get_sphere_intersection(origin_points, ray_directions, r=self.object_bounding_sphere)
        #     sqhere_hit_points = origin_points.squeeze(0) + sphere_intersections.squeeze(0)[:, 1].unsqueeze(-1) * ray_directions.squeeze(0)
        # # surface_traces = plt.get_surface_trace(path="./",
        #                                        epoch=0,
        #                                        sdf=sdf,
        #                                        resolution=100)
        # mesh = o3d.io.read_triangle_mesh("./surface_0.ply")
        # mesh.compute_vertex_normals()
        # total_vis_pcd.append(mesh)
        # vis_sqhere_hit_points = o3d.geometry.PointCloud()
        # vis_sqhere_hit_points.points = o3d.utility.Vector3dVector(sqhere_hit_points[mask_intersect.squeeze(0)].cpu().detach().numpy())
        # total_vis_pcd.append(vis_sqhere_hit_points)
        # o3d.visualization.draw_geometries(geometry_list = total_vis_pcd, width = 640, height = 480, window_name = "mesh_with_hit_points")

        #    differentiable_surface_points = self.sample_network(surface_output,
        #                                                         surface_sdf_values,
        #                                                         surface_points_grad,
        #                                                         surface_dists,
        #                                                         surface_cam_loc,
        #                                                         surface_ray_dirs)
        # composite_img[~mask_intersect.squeeze(0), :]  = self.sample_env_light(original_ray_directions.squeeze(0)[~mask_intersect.squeeze(0), :] , self.env_map)
        
        # o3d.visualization.draw_geometries(geometry_list = total_vis_pcd, width = 640, height = 480, window_name = "vis_pcd")
        
        mask_intersect = mask_intersect.detach()
        if torch.sum(mask_intersect).cpu().item() == 0:
            with torch.no_grad():
                composite_img[:, :] = self.sample_env_light(original_ray_directions.squeeze(0)[:, :], self.env_map)

        if torch.sum(mask_intersect).cpu().item() > 0:
            with torch.no_grad():
                mask_intersect[:, self.accu_refract_attenuate.squeeze() < 1.0e-1] = False
                # cv2.imshow("tt", mask_intersect.float().reshape(100, 100, 1).detach().cpu().numpy())
                # cv2.waitKey(1)

            refract_img[mask_intersect.squeeze(0), :] = self.sample_env_light(ray_directions.squeeze(0)[mask_intersect.squeeze(0), :], self.env_map.detach())
            refract_img[mask_intersect.squeeze(0), :] = refract_img[mask_intersect.squeeze(0), :] * self.accu_refract_attenuate.detach()[mask_intersect.squeeze(0), :]  
            reflect_img[mask_intersect.squeeze(0), :] = self.sample_env_light(reflect_ray_directions.squeeze(0)[mask_intersect.squeeze(0), :], self.env_map.detach())
            reflect_img[mask_intersect.squeeze(0), :] = reflect_img[mask_intersect.squeeze(0), :] * self.reflect_attenuate.detach()[mask_intersect.squeeze(0), :]
            
            composite_img[~mask_intersect.squeeze(0), :] = self.sample_env_light(original_ray_directions.squeeze(0)[~mask_intersect.squeeze(0), :].detach(), self.env_map.detach())
            # composite_img[mask_intersect.squeeze(0), :] = (refract_img[mask_intersect.squeeze(0), :] + reflect_img[mask_intersect.squeeze(0), :])
            composite_img[mask_intersect.squeeze(0), :] = (refract_img[mask_intersect.squeeze(0), :])
        composite_img = composite_img * 1000
        # composite_img = torch.pow(composite_img, 1 / 2.2)
        composite_img = torch.clamp(composite_img, 0, 0.999999)

        # print("test point num: ", torch.sum(mask_intersect).cpu().item())
        # cv2.imshow("mask", mask_intersect.float().reshape(100, 100, 1).detach().cpu().numpy())
        # cv2.imshow("composite_img", composite_img.reshape(100, 100, 3).float().detach().cpu().numpy())
        # cv2.waitKey(0)

        # print("composite_img.requires_grad")
        # print(refract_img.requires_grad)
        # print(refract_img.grad_fn)
        # print(reflect_img.requires_grad)
        # print(reflect_img.grad_fn)
        # print(composite_img.requires_grad)
        # print(composite_img.grad_fn)

        # curr_start_points [2048, 3]
        # unfinished_mask_start [2048] bool
        # acc_start_dis [2048], acc_end_dis [2048]
        with torch.no_grad():
            network_object_mask = self.first_time_mask_intersect
            mask =  ~self.first_time_mask_intersect
            mask = mask.squeeze(0)
            curr_start_points = torch.zeros(num_pixels, 3).cuda()
            acc_start_dis = torch.zeros(num_pixels).cuda()

            if not self.training:
                return composite_img, \
                   mask_intersect.squeeze(), \
                   curr_start_points, \
                   network_object_mask.squeeze(0), \
                   acc_start_dis

            if mask.sum() > 0:
                min_mask_points, min_mask_dist = self.minimal_sdf_points(num_pixels, sdf, cam_loc, original_ray_directions.squeeze(0), mask, original_min_max_t[:, :, 0].squeeze(0), original_min_max_t[:, :, 1].squeeze(0))

                curr_start_points[mask] = min_mask_points
                acc_start_dis[mask] = min_mask_dist

            network_object_mask = network_object_mask.squeeze(0)

        return composite_img, \
               mask_intersect.squeeze().detach(), \
               curr_start_points, \
               network_object_mask, \
               acc_start_dis

        #     with torch.no_grad():
        #         # curr_start_points [2048, 3]
        #         # unfinished_mask_start [2048] bool
        #         # acc_start_dis [2048], acc_end_dis [2048]
        #         # min_dis [2048], max_dis [2048]
        #         curr_start_points, unfinished_mask_start, finished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis = \
        #             self.sphere_tracing(batch_size, num_pixels, sdf, \
        #                                 ray_directions, mask_intersect, \
        #                                 origin_points, min_max_t, in_out_indicator)

        #         # network_object_mask [2048] bool
        #         network_object_mask = (acc_start_dis < acc_end_dis)

        #         # The non convergent rays should be handled by the sampler
        #         # sampler_mask [2048] bool
        #         sampler_mask = unfinished_mask_start
        #         sampler_net_obj_mask = torch.zeros_like(sampler_mask).bool().cuda()
        #         if sampler_mask.sum() > 0:
        #             sampler_min_max = torch.zeros((batch_size, num_pixels, 2)).cuda()
        #             sampler_min_max.reshape(-1, 2)[sampler_mask, 0] = acc_start_dis[sampler_mask]
        #             sampler_min_max.reshape(-1, 2)[sampler_mask, 1] = acc_end_dis[sampler_mask]

        #             sampler_pts, sampler_net_obj_mask, sampler_dists = self.ray_sampler(sdf,
        #                                                                                 object_mask,
        #                                                                                 ray_directions,
        #                                                                                 sampler_min_max,
        #                                                                                 sampler_mask,
        #                                                                                 origin_points
        #                                                                                 )

        #             curr_start_points[sampler_mask] = sampler_pts[sampler_mask]
        #             acc_start_dis[sampler_mask] = sampler_dists[sampler_mask]
        #             network_object_mask[sampler_mask] = sampler_net_obj_mask[sampler_mask]
        #         print('----------------------------------------------------------------')
        #         print(': object = {0}/{1}, secant on {2}/{3}.'
        #             .format(network_object_mask.sum(), len(network_object_mask), sampler_net_obj_mask.sum(), sampler_mask.sum()))
        #         print('----------------------------------------------------------------')
        #         input()
 
        #     # x [2048, 3], y [2048]
        #     # normals [2048, 3]
        #     x = curr_start_points.detach()
        #     x.requires_grad_(True)
        #     y = sdf(x)
        #     d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        #     normals = torch.autograd.grad(
        #         outputs=y,
        #         inputs=x,
        #         grad_outputs=d_output,
        #         create_graph=True,
        #         retain_graph=True,
        #         only_inputs=True)[0]

        #     print(origin_points.shape)
        #     print(min_max_t.unsqueeze(-1).shape)
        #     print(ray_directions.unsqueeze(2).shape)
        #     print(cam_loc.unsqueeze(0).shape)
        #     print(cam_loc.unsqueeze(0).repeat(1, 2048, 1).shape)
        #     # origin_points [1, 1, 1, 3]
        #     # near_far_t [1, 2048, 2],  ray_directions [1, 2048, 3] 
        #     # sphere_intersections_points [1, 2048, 2, 3] 
        #     sphere_intersections_points = origin_points.unsqueeze(-2) + min_max_t.unsqueeze(-1) * ray_directions.unsqueeze(-2)
        #     sphere_intersections_points[:, :, 1, :] = x.unsqueeze(0) #[1, 2048, 3]
        #     sphere_intersections_points[:, :, 0, :] = cam_loc.unsqueeze(0).repeat(1, 2048, 1) #[1, 2048, 3]
        #     sphere_intersections_points = sphere_intersections_points.reshape(-1, 3)
     
        #     lines =  np.arange(0, 4096).reshape(-1, 2)
        #     # colors = [[1, 0, 0] for i in range(len(lines))]
        #     line_set = o3d.geometry.LineSet()
        #     line_set.points = o3d.utility.Vector3dVector(sphere_intersections_points.cpu().detach().numpy())
        #     line_set.lines = o3d.utility.Vector2iVector(lines[::40, :])
        #     # line_set.colors = o3d.utility.Vector3dVector(colors)

        #     surface_traces = plt.get_surface_trace(path="./",
        #                                            epoch=0,
        #                                            sdf=sdf,
        #                                            resolution=100)

        #     mesh = o3d.io.read_triangle_mesh("./surface_0.ply")
        #     mesh.compute_vertex_normals()

        #     vis_pcd = o3d.geometry.PointCloud()
        #     vis_pcd.points = o3d.utility.Vector3dVector(x[finished_mask_start].cpu().detach().numpy())
        #     vis_pcd.normals = o3d.utility.Vector3dVector(normals[finished_mask_start].cpu().detach().numpy())

        #     # o3d.visualization.draw_geometries(geometry_list = [line_set, vis_pcd], width = 640, height = 480, window_name = "lines_with_hit_points")
        #     # o3d.visualization.draw_geometries(geometry_list = [mesh, vis_pcd], width = 640, height = 480, window_name = "mesh_with_hit_points")

        #     # ray_directions
        #     # print("ray_directions.squeeze(0)[finished_mask_start].shape")
        #     # print(ray_directions.squeeze(0)[finished_mask_start].shape)
        #     # print("normals[finished_mask_start].shape")
        #     # print(normals[finished_mask_start].shape)
        #     reflect_ray_directions = self.reflection(ray_directions.squeeze(0)[finished_mask_start], normals[finished_mask_start])
        #     refract_ray_directions, attenuate, total_reflect_mask = self.refraction(ray_directions.squeeze(0)[finished_mask_start], normals[finished_mask_start], eta1 = 1.0003, eta2 = 1.4723)
        #     print("reflect_ray_directions.shape")
        #     print(reflect_ray_directions.shape)
        #     print("refract_ray_directions.shape")
        #     print(refract_ray_directions.shape)

        #     input()

        #     origin_points = curr_start_points[finished_mask_start]
        #     # min_max_t [1, 2048, 2] 
        #     min_max_t = min_max_t.squeeze(0)
        #     min_max_t[:, 0].fill_(0)
        #     min_max_t = min_max_t[finished_mask_start]

        #     print("origin_points.shape")
        #     print(origin_points.shape)
        #     print("min_max_t.shape")
        #     print(min_max_t.shape)

        #     input()

        #     # vis_pcd = o3d.geometry.PointCloud()
        #     # vis_pcd.points = o3d.utility.Vector3dVector(x[network_object_mask].cpu().detach().numpy())
        #     # vis_pcd.normals = o3d.utility.Vector3dVector(normal[network_object_mask].cpu().detach().numpy())
        #     # o3d.io.write_point_cloud("vis_pcd.ply", vis_pcd)
        #     # vis_pcd = o3d.geometry.PointCloud()
        #     # vis_pcd.points = o3d.utility.Vector3dVector(x[finished_mask_start].cpu().detach().numpy())
        #     # vis_pcd.normals = o3d.utility.Vector3dVector(normal[finished_mask_start].cpu().detach().numpy())
        #     # o3d.io.write_point_cloud("vis_pcd.ply", vis_pcd)
        #     # input()

        # if not self.training:
        #     return curr_start_points, \
        #            network_object_mask, \
        #            acc_start_dis

        # # ray_directions [1, 2048, 3] 
        # ray_directions = ray_directions.reshape(-1, 3)
        # # mask_intersect [1, 2048]
        # mask_intersect = mask_intersect.reshape(-1)
        # # mask_intersect [2048]

        # # in_mask [2048], out_mask [2048]
        # in_mask = ~network_object_mask & object_mask & ~sampler_mask
        # out_mask = ~object_mask & ~sampler_mask

        # # pcd_in_network_object_mask = o3d.geometry.PointCloud()
        # # pcd_in_mask = o3d.geometry.PointCloud()
        # # pcd_out_mask = o3d.geometry.PointCloud()
        # # pcd_out_inter = o3d.geometry.PointCloud()
        # # pcd_in_network_object_mask.points = o3d.utility.Vector3dVector(curr_start_points[network_object_mask].cpu().numpy())   
        # # pcd_in_mask.points = o3d.utility.Vector3dVector(curr_start_points[in_mask].cpu().numpy())   
        # # pcd_out_mask.points = o3d.utility.Vector3dVector(curr_start_points[out_mask].cpu().numpy())   
        # # pcd_out_inter.points = o3d.utility.Vector3dVector(curr_start_points[mask_intersect].cpu().numpy())   
        # # o3d.io.write_point_cloud("pcd_in_network_object_mask.ply", pcd_in_network_object_mask)
        # # o3d.io.write_point_cloud("pcd_in_mask.ply", pcd_in_mask)
        # # o3d.io.write_point_cloud("pcd_out_mask.ply", pcd_out_mask)
        # # o3d.io.write_point_cloud("pcd_out_inter.ply", pcd_out_inter)
        # # input()

        # mask_left_out = (in_mask | out_mask) & ~mask_intersect
        # print("sampler_mask.sum: {0}".format(sampler_mask.sum()))
        # if mask_left_out.sum() > 0:  # project the origin to the not intersect points on the sphere
        #     cam_left_out = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask_left_out]
        #     rays_left_out = ray_directions[mask_left_out]
        #     acc_start_dis[mask_left_out] = -torch.bmm(rays_left_out.view(-1, 1, 3), cam_left_out.view(-1, 3, 1)).squeeze()
        #     curr_start_points[mask_left_out] = cam_left_out + acc_start_dis[mask_left_out].unsqueeze(1) * rays_left_out

        #     # pcd_left_out = o3d.geometry.PointCloud()
        #     # pcd_left_out.points = o3d.utility.Vector3dVector(curr_start_points[mask_left_out].cpu().numpy())      
        #     # o3d.io.write_point_cloud("pcd_left_out.ply", pcd_left_out)
        #     # surface_traces = plt.get_surface_trace(path="./",
        #     #                                         epoch=0,
        #     #                                         sdf=sdf,
        #     #                                         resolution=100)
        #     # # pcd_in_mask = o3d.geometry.PointCloud()
        #     # pcd_out_mask = o3d.geometry.PointCloud()
        #     # pcd_out_inter = o3d.geometry.PointCloud()
        #     # pcd_sampler_mask = o3d.geometry.PointCloud()
        #     # pcd_in_mask.points = o3d.utility.Vector3dVector(curr_start_points[in_mask].cpu().numpy())   
        #     # pcd_out_mask.points = o3d.utility.Vector3dVector(curr_start_points[out_mask].cpu().numpy())   
        #     # pcd_out_inter.points = o3d.utility.Vector3dVector(curr_start_points[mask_intersect].cpu().numpy())   
        #     # pcd_sampler_mask.points = o3d.utility.Vector3dVector(curr_start_points[sampler_mask].cpu().numpy())   
        #     # o3d.io.write_point_cloud("pcd_in_mask.ply", pcd_in_mask)
        #     # o3d.io.write_point_cloud("pcd_out_mask.ply", pcd_out_mask)
        #     # o3d.io.write_point_cloud("pcd_out_inter.ply", pcd_out_inter)
        #     # o3d.io.write_point_cloud("pcd_sampler_mask.ply", pcd_sampler_mask)
        #     # input()

        # # mask [2048], out_mask [2048]
        # mask = (in_mask | out_mask) & mask_intersect

        # if mask.sum() > 0:
        #     min_dis[network_object_mask & out_mask] = acc_start_dis[network_object_mask & out_mask]

        #     min_mask_points, min_mask_dist = self.minimal_sdf_points(num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis)

        #     curr_start_points[mask] = min_mask_points
        #     acc_start_dis[mask] = min_mask_dist

        # return curr_start_points, \
        #        network_object_mask, \
        #        acc_start_dis

    def sphere_tracing_single_direction(self, batch_size, num_pixels, sdf, ray_directions, mask_intersect, \
                                        origin_points, min_max_t, in_out_indicator):
        ''' Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection '''

        # origin_points [1, 2048, 3]
        # near_far_t [1, 2048, 2],  ray_directions [1, 2048, 3] 
        sphere_intersections_points = origin_points.unsqueeze(-2) + min_max_t.unsqueeze(-1) * ray_directions.unsqueeze(-2)
       
        unfinished_mask_start = mask_intersect.reshape(-1).clone()

        curr_start_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[:,:,0,:].reshape(-1,3)[unfinished_mask_start]
        acc_start_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_start_dis[unfinished_mask_start] = min_max_t.reshape(-1,2)[unfinished_mask_start,0]

        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

        while True:
            # Update sdf
            curr_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold)

            if (unfinished_mask_start.sum() == 0) or iters == self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            acc_start_dis = acc_start_dis + curr_sdf_start

            # Update points
            curr_start_points = (origin_points + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)

            # Fix points which wrongly crossed the surface
            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

            not_projected_start = next_sdf_start < 0
            not_proj_iters = 0
            while (not_projected_start.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (origin_points + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_start]

                # Calc sdf
                next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start])
                next_sdf_end[not_projected_end] = sdf(curr_end_points[not_projected_end])

                # Update mask
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start
            finished_mask_start = not unfinished_mask_start

        return curr_start_points, unfinished_mask_start, finished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis

    def sphere_tracing(self, batch_size, num_pixels, sdf, ray_directions, mask_intersect, \
                       origin_points, min_max_t):
        ''' Run sphere tracing algorithm for max iterations from both sides of unit sphere intersection '''

        # origin_points [1, 2048, 3]
        # near_far_t [1, 2048, 2],  ray_directions [1, 2048, 3] 
        sphere_intersections_points = origin_points.unsqueeze(-2) + min_max_t.unsqueeze(-1) * ray_directions.unsqueeze(-2)
       
        unfinished_mask_start = mask_intersect.reshape(-1).clone()
        unfinished_mask_end = mask_intersect.reshape(-1).clone()

        # print(sphere_intersections_points.shape)
        # print(unfinished_mask_start.shape)
        # print(unfinished_mask_end.shape)
        # print("-------------------------------")

        # Initialize start current points
        curr_start_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_start_points[unfinished_mask_start] = sphere_intersections_points[:,:,0,:].reshape(-1,3)[unfinished_mask_start]
        acc_start_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_start_dis[unfinished_mask_start] = min_max_t.reshape(-1,2)[unfinished_mask_start,0]

        # Initialize end current points
        curr_end_points = torch.zeros(batch_size * num_pixels, 3).cuda().float()
        curr_end_points[unfinished_mask_end] = sphere_intersections_points[:,:,1,:].reshape(-1,3)[unfinished_mask_end]
        acc_end_dis = torch.zeros(batch_size * num_pixels).cuda().float()
        acc_end_dis[unfinished_mask_end] = min_max_t.reshape(-1,2)[unfinished_mask_end,1]

        # Initizliae min and max depth
        min_dis = acc_start_dis.clone()
        max_dis = acc_end_dis.clone()

        # Iterate on the rays (from both sides) till finding a surface
        iters = 0

        next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
        next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

        next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
        next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])

        while True:
            # Update sdf
            curr_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            curr_sdf_start[unfinished_mask_start] = next_sdf_start[unfinished_mask_start]
            curr_sdf_start[curr_sdf_start <= self.sdf_threshold] = 0

            curr_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            curr_sdf_end[unfinished_mask_end] = next_sdf_end[unfinished_mask_end]
            curr_sdf_end[curr_sdf_end <= self.sdf_threshold] = 0

            # Update masks
            unfinished_mask_start = unfinished_mask_start & (curr_sdf_start > self.sdf_threshold)
            unfinished_mask_end = unfinished_mask_end & (curr_sdf_end > self.sdf_threshold)

            if (unfinished_mask_start.sum() == 0 and unfinished_mask_end.sum() == 0) or iters == self.sphere_tracing_iters:
                break
            iters += 1

            # Make step
            # Update distance
            acc_start_dis = acc_start_dis + curr_sdf_start
            acc_end_dis = acc_end_dis - curr_sdf_end

            # Update points
            curr_start_points = (origin_points + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)
            curr_end_points = (origin_points + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)

            # Fix points which wrongly crossed the surface
            next_sdf_start = torch.zeros_like(acc_start_dis).cuda()
            next_sdf_start[unfinished_mask_start] = sdf(curr_start_points[unfinished_mask_start])

            next_sdf_end = torch.zeros_like(acc_end_dis).cuda()
            next_sdf_end[unfinished_mask_end] = sdf(curr_end_points[unfinished_mask_end])

            not_projected_start = next_sdf_start < 0
            not_projected_end = next_sdf_end < 0
            not_proj_iters = 0
            while (not_projected_start.sum() > 0 or not_projected_end.sum() > 0) and not_proj_iters < self.line_step_iters:
                # Step backwards
                acc_start_dis[not_projected_start] -= ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_start[not_projected_start]
                curr_start_points[not_projected_start] = (origin_points + acc_start_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_start]

                acc_end_dis[not_projected_end] += ((1 - self.line_search_step) / (2 ** not_proj_iters)) * curr_sdf_end[not_projected_end]
                curr_end_points[not_projected_end] = (origin_points + acc_end_dis.reshape(batch_size, num_pixels, 1) * ray_directions).reshape(-1, 3)[not_projected_end]

                # Calc sdf
                next_sdf_start[not_projected_start] = sdf(curr_start_points[not_projected_start])
                next_sdf_end[not_projected_end] = sdf(curr_end_points[not_projected_end])

                # Update mask
                not_projected_start = next_sdf_start < 0
                not_projected_end = next_sdf_end < 0
                not_proj_iters += 1

            unfinished_mask_start = unfinished_mask_start & (acc_start_dis < acc_end_dis)
            unfinished_mask_end = unfinished_mask_end & (acc_start_dis < acc_end_dis)

            # curr_start_pcd = o3d.geometry.PointCloud()
            # curr_start_pcd.points = o3d.utility.Vector3dVector(curr_start_points[~unfinished_mask_start].cpu().numpy())      
            # o3d.io.write_point_cloud("curr_start_pcd.ply", curr_start_pcd)
            # curr_end_pcd = o3d.geometry.PointCloud()
            # curr_end_pcd.points = o3d.utility.Vector3dVector(curr_end_points[~unfinished_mask_end].cpu().numpy())      
            # o3d.io.write_point_cloud("curr_end_pcd.ply", curr_end_pcd)

        return curr_start_points, unfinished_mask_start, acc_start_dis, acc_end_dis, min_dis, max_dis
        
    def ray_tracing_using_sampler(self, sdf, origin_points, ray_directions, sampler_min_max, object_mask, sampler_mask, in_out_indicator):
        # origin_points [1, 2048, 3]
        # near_far_t [1, 2048, 2]
        ''' Sample the ray in a given range and run secant on rays which have sign transition '''

        batch_size, num_pixels, _ = ray_directions.shape
        n_total_pxl = batch_size * num_pixels
        sampler_pts = torch.zeros(n_total_pxl, 3).cuda().float()
        sampler_dists = torch.zeros(n_total_pxl).cuda().float()

        # sampler_mask [2048], pts_intervals [1, 2048, 100], points [1, 2048, 100, 3]
        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).cuda().view(1, 1, -1)
        pts_intervals = sampler_min_max[:, :, 0].unsqueeze(-1) + intervals_dist * (sampler_min_max[:, :, 1] - sampler_min_max[:, :, 0]).unsqueeze(-1)
        points = origin_points.unsqueeze(-2) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(2)

        # Get the non convergent rays
        # sampler_mask [2048], mask_intersect_idx [1998], points [1998, 100, 3], pts_intervals [1998, 100]
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask, :]

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf(pnts))
        # sdf_val [1998, 100]
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)

        if sdf_val.shape[0] == 0:
            hit_mask = torch.zeros(n_total_pxl).cuda().bool()
            return sampler_pts, sampler_dists, hit_mask

        sign_sdf_val = torch.sign(sdf_val).float()
        signchange_sdf_val = torch.ones_like(sign_sdf_val)
        signchange_sdf_val[:, :-1] = sign_sdf_val[:, :-1] * sign_sdf_val[:, 1:]
        tmp = signchange_sdf_val * torch.arange(self.n_steps, 0, -1).cuda().float().reshape((1, self.n_steps))  # Force argmin to return the first min value
        sampler_pts_ind = torch.argmin(tmp, -1)
    
        net_surface_pts = (in_out_indicator * signchange_sdf_val[torch.arange(signchange_sdf_val.shape[0]), sampler_pts_ind] < 0)

        secant_pts = net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            z_low = pts_intervals[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            sdf_low = sdf_val[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]

            # cam_loc_secant = origin_points.squeeze(1).repeat(1, num_pixels, 1).reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            cam_loc_secant = origin_points.squeeze(1).reshape((-1, 3))[mask_intersect_idx[secant_pts]]        
            ray_directions_secant = ray_directions.reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            z_pred_secant = self.secant(sdf_low, sdf_high, z_low, z_high, cam_loc_secant, ray_directions_secant, sdf)

            # Get points
            sampler_pts[mask_intersect_idx[secant_pts]] = cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant

        # sampler_pts_pcd = o3d.geometry.PointCloud()
        # sampler_pts_pcd.points = o3d.utility.Vector3dVector(sampler_pts[mask_intersect_idx[secant_pts]].cpu().detach().numpy())
        # o3d.visualization.draw_geometries(geometry_list = [sampler_pts_pcd], width = 640, height = 480, window_name = "lines_with_hit_points")

        hit_mask = torch.zeros(n_total_pxl).cuda().bool()
        hit_mask[mask_intersect_idx[secant_pts]] = True

        return sampler_pts, sampler_dists, hit_mask

    def idr_ray_sampler(self, sdf, cam_loc, object_mask, ray_directions, sampler_min_max, sampler_mask):

        batch_size, num_pixels, _ = ray_directions.shape
        n_total_pxl = batch_size * num_pixels
        sampler_pts = torch.zeros(n_total_pxl, 3).cuda().float()
        sampler_dists = torch.zeros(n_total_pxl).cuda().float()

        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).cuda().view(1, 1, -1)

        pts_intervals = sampler_min_max[:, :, 0].unsqueeze(-1) + intervals_dist * (sampler_min_max[:, :, 1] - sampler_min_max[:, :, 0]).unsqueeze(-1)
        points = cam_loc.reshape(batch_size, 1, 1, 3) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(2)

        # Get the non convergent rays
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask]

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).cuda().float().reshape((1, self.n_steps))  # Force argmin to return the first min value
        sampler_pts_ind = torch.argmin(tmp, -1)
        sampler_pts[mask_intersect_idx] = points[torch.arange(points.shape[0]), sampler_pts_ind, :]
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind]

        true_surface_pts = object_mask[sampler_mask]
        net_surface_pts = (sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0)

        # take points with minimal SDF value for P_out pixels
        p_out_mask = ~(true_surface_pts & net_surface_pts)
        n_p_out = p_out_mask.sum()
        if n_p_out > 0:
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1)
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][torch.arange(n_p_out), out_pts_idx, :]
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[p_out_mask, :][torch.arange(n_p_out), out_pts_idx]

        # Get Network object mask
        sampler_net_obj_mask = sampler_mask.clone()
        sampler_net_obj_mask[mask_intersect_idx[~net_surface_pts]] = False

        # Run Secant method
        secant_pts = net_surface_pts & true_surface_pts if self.training else net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            z_low = pts_intervals[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            sdf_low = sdf_val[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            cam_loc_secant = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            ray_directions_secant = ray_directions.reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            z_pred_secant = self.secant(sdf_low, sdf_high, z_low, z_high, cam_loc_secant, ray_directions_secant, sdf)

            # Get points
            sampler_pts[mask_intersect_idx[secant_pts]] = cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant

        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def ray_sampler(self, sdf, object_mask, ray_directions, sampler_min_max, sampler_mask, \
                    origin_points):
        # origin_points [1, 2048, 3]
        # near_far_t [1, 2048, 2]
        ''' Sample the ray in a given range and run secant on rays which have sign transition '''

        batch_size, num_pixels, _ = ray_directions.shape
        n_total_pxl = batch_size * num_pixels
        sampler_pts = torch.zeros(n_total_pxl, 3).cuda().float()
        sampler_dists = torch.zeros(n_total_pxl).cuda().float()
        print("ray_directions.shape")
        print(ray_directions.shape)
        input()

        intervals_dist = torch.linspace(0, 1, steps=self.n_steps).cuda().view(1, 1, -1)
        print("intervals_dist.shape")
        print(intervals_dist.shape)
        input()

        pts_intervals = sampler_min_max[:, :, 0].unsqueeze(-1) + intervals_dist * (sampler_min_max[:, :, 1] - sampler_min_max[:, :, 0]).unsqueeze(-1)
        points = origin_points.unsqueeze(-2) + pts_intervals.unsqueeze(-1) * ray_directions.unsqueeze(2)
        print("sampler_mask.shape")
        print(sampler_mask.shape)
        print("pts_intervals.shape")
        print(pts_intervals.shape)
        print("points.shape")
        print(points.shape)
        input()

        # Get the non convergent rays
        mask_intersect_idx = torch.nonzero(sampler_mask).flatten()
        points = points.reshape((-1, self.n_steps, 3))[sampler_mask, :, :]
        pts_intervals = pts_intervals.reshape((-1, self.n_steps))[sampler_mask, :]
        print("points.shape")
        print(points.shape)
        print("pts_intervals.shape")
        print(pts_intervals.shape)
        input()

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 100000, dim=0):
            sdf_val_all.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val_all).reshape(-1, self.n_steps)
        print("sdf_val.shape")
        print(sdf_val.shape)
        print(sdf_val.device)
        input()

        tmp = torch.sign(sdf_val) * torch.arange(self.n_steps, 0, -1).cuda().float().reshape((1, self.n_steps))  # Force argmin to return the first min value
        print("tmp.shape")
        print(tmp.shape)
        input()
        sampler_pts_ind = torch.argmin(tmp, -1)
        print("sampler_pts_ind.shape")
        print(sampler_pts_ind.shape)
        input()
        sampler_pts[mask_intersect_idx] = points[torch.arange(points.shape[0]), sampler_pts_ind, :]
        sampler_dists[mask_intersect_idx] = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind]

        true_surface_pts = object_mask[sampler_mask]
        net_surface_pts = (sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0)

        # take points with minimal SDF value for P_out pixels
        p_out_mask = not (true_surface_pts & net_surface_pts)
        n_p_out = p_out_mask.sum()
        if n_p_out > 0:
            out_pts_idx = torch.argmin(sdf_val[p_out_mask, :], -1)
            sampler_pts[mask_intersect_idx[p_out_mask]] = points[p_out_mask, :, :][torch.arange(n_p_out), out_pts_idx, :]
            sampler_dists[mask_intersect_idx[p_out_mask]] = pts_intervals[p_out_mask, :][torch.arange(n_p_out), out_pts_idx]

        # Get Network object mask
        sampler_net_obj_mask = sampler_mask.clone()
        sampler_net_obj_mask[mask_intersect_idx[not net_surface_pts]] = False

        # Run Secant method
        secant_pts = net_surface_pts & true_surface_pts if self.training else net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            z_low = pts_intervals[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            sdf_low = sdf_val[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]

            cam_loc_secant = origin_points.squeeze(1).repeat(1, num_pixels, 1).reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            ray_directions_secant = ray_directions.reshape((-1, 3))[mask_intersect_idx[secant_pts]]
            z_pred_secant = self.secant(sdf_low, sdf_high, z_low, z_high, cam_loc_secant, ray_directions_secant, sdf)

            # Get points
            sampler_pts[mask_intersect_idx[secant_pts]] = cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            sampler_dists[mask_intersect_idx[secant_pts]] = z_pred_secant

        return sampler_pts, sampler_net_obj_mask, sampler_dists

    def secant(self, sdf_low, sdf_high, z_low, z_high, cam_loc, ray_directions, sdf):
        ''' Runs the secant method for interval [z_low, z_high] for n_secant_steps '''

        z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        for i in range(self.n_secant_steps):
            p_mid = cam_loc + z_pred.unsqueeze(-1) * ray_directions
            sdf_mid = sdf(p_mid)
            ind_low = sdf_mid > 0
            if ind_low.sum() > 0:
                z_low[ind_low] = z_pred[ind_low]
                sdf_low[ind_low] = sdf_mid[ind_low]
            ind_high = sdf_mid < 0
            if ind_high.sum() > 0:
                z_high[ind_high] = z_pred[ind_high]
                sdf_high[ind_high] = sdf_mid[ind_high]

            z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low + 1.0e-20) + z_low

        return z_pred

    def minimal_sdf_points(self, num_pixels, sdf, cam_loc, ray_directions, mask, min_dis, max_dis):
        ''' Find points with minimal SDF value on rays for P_out pixels '''

        n_mask_points = mask.sum()

        n = self.n_steps
        # print("n_steps: {0}".format(n))
        # input()
        # steps = torch.linspace(0.0, 1.0,n).cuda()
        steps = torch.empty(n).uniform_(0.0, 1.0).cuda()
        mask_max_dis = max_dis[mask].unsqueeze(-1)
        mask_min_dis = min_dis[mask].unsqueeze(-1)
        steps = steps.unsqueeze(0).repeat(n_mask_points, 1) * (mask_max_dis - mask_min_dis) + mask_min_dis

        # print("max_dis.shape")
        # print(max_dis.shape)
        # print("mask.shape")
        # print(mask.shape)
        # print("max_dis[mask].shape")
        # print(max_dis[mask].shape)
        # print(mask_max_dis.shape)
        # input()

        mask_points = cam_loc.reshape(-1, 3)[mask]
        # mask_points = cam_loc.unsqueeze(1).repeat(1, num_pixels, 1).reshape(-1, 3)[mask]
        mask_rays = ray_directions[mask, :]

        mask_points_all = mask_points.unsqueeze(1).repeat(1, n, 1) + steps.unsqueeze(-1) * mask_rays.unsqueeze(1).repeat(1, n, 1)
        points = mask_points_all.reshape(-1, 3)

        mask_sdf_all = []
        for pnts in torch.split(points, 100000, dim=0):
            mask_sdf_all.append(sdf(pnts))

        mask_sdf_all = torch.cat(mask_sdf_all).reshape(-1, n)
        min_vals, min_idx = mask_sdf_all.min(-1)
        min_mask_points = mask_points_all.reshape(-1, n, 3)[torch.arange(0, n_mask_points), min_idx]
        min_mask_dist = steps.reshape(-1, n)[torch.arange(0, n_mask_points), min_idx]
        min_sdf_vals = min_vals

        return min_mask_points, min_mask_dist, min_sdf_vals

    # def diff_mesh_ray_tracing(self,
    #                           sdf,
    #                           mesh_vertices_tensor,
    #                           mesh_faces_tensor,
    #                           mesh_vertex_normals_tensor,
    #                           face_idx_tensor, 
    #                           ray_origins_tensor, 
    #                           ray_directions_tensor,
    #                           reflect_ray_directions,
    #                           refract_attenuate,
    #                           reflect_attenuate,
    #                           valid_ray_mask,
    #                           ray_board_indices,
    #                           material_ior,
    #                           is_no_corr):

    #     # print('ray_board_indices')
    #     # print(ray_board_indices.shape)
    #     # print('ray_board_indices')
    #     # print(ray_board_indices)
    #     # print('corr')
    #     # print(corr.shape)
    #     # all_ray_board_indices = ray_board_indices.reshape(1, 40000, 1).cpu()
    #     # all_corr = corr.reshape(1, 200, 200, 3).cpu()
    #     # for image_idx in range(1):
    #     #     board_idx = all_ray_board_indices[image_idx, 0, 0].item()
    #     #     print('board_idx')
    #     #     print(board_idx)
    #     #     single_curr = all_corr[image_idx, :, :, 0:2]
    #     #     single_curr = single_curr.reshape(-1, 2)
    #     #     valid_mask = ~torch.isnan(single_curr[:, 0])
    #     #     print('valid pixel num:')
    #     #     print(torch.sum(valid_mask))

    #     #     board_mesh = self.board_mesh_list[board_idx]
    #     #     board_texture_uvs = self.board_texture_uvs_list[board_idx].cpu()
    #     #     board_texture = self.board_texture_list[board_idx].cpu()
    #     #     corr_image = torch.zeros(200 * 200, 3)
    #     #     corr_image[valid_mask, :] = self.sample_from_texture(single_curr[valid_mask], board_texture)
    #     #     # print(corr_image.shape)
    #     #     corr_image = corr_image.reshape(200, 200, 3).numpy()[:, :, ::-1]
    #     #     print('../corr_image_' + str(image_idx) + '.jpg')
    #     #     cv2.imwrite('../corr_image_' + str(image_idx) + '.jpg', (corr_image * 255.0).astype(np.uint8))

    #     # exit()

    #     # print("diff_mesh_ray_tracing")
    #     # torch.autograd.set_detect_anomaly(True) 

    #     total_vis_pcd = []
    #     vis_debug = False

    #     # print("ray_directions_tensor.shape")
    #     # print(ray_directions_tensor.shape)
    #     # input()
 
    #     ray_origins_list = [ray_origins_tensor[0, ...].clone().detach()]
    #     ray_directions_list = [ray_directions_tensor[0, ...].clone().detach()]
    #     vertex_idx_list = []
    #     accu_refract_attenuate = torch.ones(ray_origins_list[0].shape[0]).cuda()
    #     reflect_attenuate = torch.ones(ray_origins_list[0].shape[0]).cuda()
    #     for ray_iter in range(face_idx_tensor.shape[0]):
    #         # print("ray_iter: {0}".format(ray_iter))
    #         face_idx = face_idx_tensor[ray_iter, :].detach()    
    #         ray_origins = ray_origins_list[ray_iter]
    #         ray_directions = ray_directions_list[ray_iter]
    #         if ray_iter == 2:
    #             break

    #         vertex_idx_list.append(torch.flatten(mesh_faces_tensor[face_idx]))
    #         vertices = mesh_vertices_tensor[torch.flatten(mesh_faces_tensor[face_idx]), :]
    #         vertices = vertices.reshape(-1, 3, 3)
    #         v0 = vertices[:, 0, :]
    #         v1 = vertices[:, 1, :]
    #         v2 = vertices[:, 2, :]
    #         plane_normal = torch.cross(v2 - v0, v2 - v1)
    #         plane_normal = plane_normal / torch.sqrt(torch.clamp(torch.sum(plane_normal*plane_normal, dim=1), min=1e-10 ).unsqueeze(1))
    #         plane_point = (v0 + v1 + v2) * 0.33333333333333

    #         hit_points = self.line_plane_collision(plane_normal, plane_point, ray_directions, ray_origins)
    #         if vis_debug:
    #             vis_pcd = o3d.geometry.PointCloud()
    #             vis_pcd.points = o3d.utility.Vector3dVector(hit_points.cpu().detach().numpy())
    #             colors = np.array([[0, 1 - ray_iter * 0.3, ray_iter * 0.3] for i in range(len(vis_pcd.points))])
    #             vis_pcd.colors = o3d.utility.Vector3dVector(colors[:, :])
    #             total_vis_pcd.append(vis_pcd)

    #         barycentric_weight = self.calculate_barycentric_weight(mesh_vertices_tensor, mesh_faces_tensor, hit_points, face_idx)
    #         barycentric_weight = barycentric_weight
    #         vertex_normals = mesh_vertex_normals_tensor[torch.flatten(mesh_faces_tensor[face_idx]), :]
    #         vertex_normals = vertex_normals.reshape(-1, 3, 3)
    #         v0_normals = vertex_normals[:, 0, :]
    #         v1_normals = vertex_normals[:, 1, :]
    #         v2_normals = vertex_normals[:, 2, :]
    #         hit_normals = barycentric_weight[:, 0][:, None] * v0_normals + \
    #             barycentric_weight[:, 1][:, None] * v1_normals + \
    #             barycentric_weight[:, 2][:, None] * v2_normals
    #         hit_normals = hit_normals / torch.sqrt(torch.clamp(torch.sum(hit_normals*hit_normals, dim=1), min=1e-10 ).unsqueeze(1))          
        
    #         if ray_iter % 2 == 0:
    #             refract_ray_directions, refract_attenuate, _ = self.refraction(ray_directions, hit_normals, eta1 = 1.0003, eta2 = material_ior)
    #         else:
    #             refract_ray_directions, refract_attenuate, _ = self.refraction(ray_directions, -hit_normals, eta1 = material_ior, eta2 = 1.0003)
    #         # print('refract_attenuate')
    #         # print(refract_attenuate)
    #         # print(ray_iter)
    #         # input()
    #         # input()
    #         # input()
    #         accu_refract_attenuate *= (1 - refract_attenuate)
    #         if ray_iter == 0:
    #             reflect_ray_directions = self.reflection(-ray_directions, hit_normals)
    #             reflect_ray_origins = hit_points
    #             reflect_attenuate = refract_attenuate
    #         ray_origins_list.append(hit_points.clone())
    #         ray_directions_list.append(refract_ray_directions.clone())

    #     ray_origins = ray_origins_list[2]
    #     ray_directions = ray_directions_list[2]
        
    #     if vis_debug:
    #         vis_mesh = o3d.geometry.TriangleMesh()
    #         vis_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces_tensor.cpu().detach().numpy())
    #         vis_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
    #         total_vis_pcd.append(vis_mesh)
    #         o3d.visualization.draw_geometries(geometry_list = total_vis_pcd, width = 640, height = 480, window_name = "mesh_with_hit_points_diff", point_show_normal=False) 
        
    #     relavent_vertex_idx_tensor = torch.cat(vertex_idx_list, dim=0)
    #     relavent_vertex_idx_tensor = torch.unique(relavent_vertex_idx_tensor, sorted=True)

    #     num_pixels = valid_ray_mask.shape[0]
    #     refract_img = torch.zeros(num_pixels, 3).float().cuda()
    #     reflect_img = torch.zeros(num_pixels, 3).float().cuda()

    #     all_board_hit_points_uvs = torch.FloatTensor(num_pixels, 2).zero_().cuda() / 0.0
    #     if torch.sum(valid_ray_mask).cpu().item() < 1 or refract_attenuate.numel() == 0:
    #         return refract_img, \
    #            valid_ray_mask.detach(), \
    #            relavent_vertex_idx_tensor.detach(), \
    #            all_board_hit_points_uvs.detach()

    #     # refract image
    #     # color from env map
    #     # print("ray dir nan num: {0}".format(torch.sum(torch.isnan(ray_directions.reshape(-1)))))
    #     if self.with_env_map:
    #         temp, uvs = self.sample_env_light(ray_directions, self.env_image.detach(), self.env_width, self.env_height) 
    #         refract_img[valid_ray_mask, :] = temp 
    #     # print("temp nan num: {0}".format(torch.sum(torch.isnan(temp.reshape(-1)))))
    #     # if torch.sum(torch.isnan(temp.reshape(-1))) > 0:
    #     #     print('temp')
    #     #     print(temp.shape)
    #     #     print(temp)
    #     #     print('nan dir')
    #     #     temp_sum = torch.sum(temp, dim=1)
    #     #     print(temp[torch.isnan(temp_sum), :])
    #     #     print(ray_directions[torch.isnan(temp_sum), :])
    #     #     print(uvs[torch.isnan(temp_sum), :])
        
    #     # color from scene mesh
    #     hit_points, index_ray, _, _, points_uvs = self.diff_ray_triangle_intersec_trimesh(self.scene_mesh, ray_origins, ray_directions, None, self.scene_texture_uvs)
    #     # temp_hit_points = torch.FloatTensor(ray_origins.shape[0], 3).zero_().cuda() / 0.0 # set all to NaN
    #     # output_hit_points = torch.FloatTensor(num_pixels, 3).zero_().cuda() / 0.0 # set all to NaN
    #     # temp_hit_points[index_ray, :] = hit_points
    #     # output_hit_points[valid_ray_mask, :] = temp_hit_points
    #     temp_tensor = refract_img[valid_ray_mask, :]
    #     # print('sample_from_texture 0', flush=True)
    #     temp_tensor[index_ray, :] = self.sample_from_texture(points_uvs, self.scene_texture)
    #     # 1.
    #     refract_img[valid_ray_mask, :] = temp_tensor

    #     all_board_hit_points_uvs = torch.FloatTensor(num_pixels, 2).zero_().cuda() / 0.0 # set all to NaN
    #     # print("all_board_hit_points_uvs.shape")
    #     # print(all_board_hit_points_uvs.shape)
    #     # exit()
    #     with_reflect = False 
    #     if is_no_corr is False:
    #         board_indices = torch.unique(ray_board_indices)
    #         for i in range(board_indices.shape[0]):
    #             board_idx = board_indices[i]
    #             # print('board_idx: {0}'.format(board_idx))
    #             board_mesh = self.board_mesh_list[board_idx]
    #             board_texture_uvs = self.board_texture_uvs_list[board_idx]
    #             board_texture = self.board_texture_list[board_idx]
    
    #             board_valid_ray_mask = valid_ray_mask.clone() 
    #             board_valid_ray_mask[ray_board_indices != board_idx] = False
    #             board_ray_origins = ray_origins[ray_board_indices[valid_ray_mask] == board_idx, :]
    #             board_ray_directions = ray_directions[ray_board_indices[valid_ray_mask] == board_idx, :]
    #             board_reflect_ray_origins = reflect_ray_origins[ray_board_indices[valid_ray_mask] == board_idx, :]
    #             board_reflect_ray_directions = reflect_ray_directions[ray_board_indices[valid_ray_mask] == board_idx, :]

    #             with_board_scene = True 
    #             if with_board_scene:
    #                 board_scene_idx = board_idx // 15 
    #                 board_scene_hit_points, board_scene_index_ray, _, _, board_scene_points_uvs = self.diff_ray_triangle_intersec_trimesh(self.board_scene_mesh_list[board_scene_idx], board_ray_origins, board_ray_directions, None, self.board_scene_texture_uvs_list[board_scene_idx])
    #                 temp_tensor = refract_img[board_valid_ray_mask, :]
    #                 # print('sample_from_texture 1', flush=True)
    #                 temp_tensor[board_scene_index_ray, :] = self.sample_from_texture(board_scene_points_uvs, self.board_scene_texture_list[board_scene_idx])
    #                 # 2.
    #                 refract_img[board_valid_ray_mask, :] = temp_tensor 

    #                 if with_reflect:
    #                     board_reflect_scene_hit_points, board_reflect_scene_index_ray, _, _, board_reflect_scene_points_uvs = self.diff_ray_triangle_intersec_trimesh(self.board_scene_mesh_list[board_scene_idx], board_reflect_ray_origins, board_reflect_ray_directions, None, self.board_scene_texture_uvs_list[board_scene_idx])
    #                     reflect_temp_tensor = refract_img[board_valid_ray_mask, :]
    #                     # print('sample_from_texture 1', flush=True)
    #                     reflect_temp_tensor[board_reflect_scene_index_ray, :] = self.sample_from_texture(board_reflect_scene_points_uvs, self.board_scene_texture_list[board_scene_idx])
    #                     reflect_img[board_valid_ray_mask, :] = temp_tensor 

    #             # board_hit_points, index_ray, _, _, board_points_uvs = self.diff_ray_triangle_intersec_trimesh(board_mesh, board_ray_origins, board_ray_directions, None, board_texture_uvs)
    #             board_hit_points, index_ray, board_points_uvs, inside_mask = self.diff_ray_plane_intersec_trimesh(board_mesh, board_ray_origins, board_ray_directions, board_texture_uvs)
                
    #             temp_board_points_uvs = torch.FloatTensor(board_ray_origins.shape[0], 2).zero_().cuda() / 0.0 # set all to NaN
    #             temp_board_points_uvs[index_ray, :] = board_points_uvs
    #             all_board_hit_points_uvs[board_valid_ray_mask, :] = temp_board_points_uvs
    #             temp_tensor = refract_img[board_valid_ray_mask, :]
                
    #             # temp_tensor[index_ray, :] = self.sample_from_texture(board_points_uvs, board_texture)
    #             # refract_img[board_valid_ray_mask, :] = temp_tensor 
    #             if board_points_uvs.nelement() > 0 and \
    #                 inside_mask.nelement() > 0 and \
    #                 torch.sum(inside_mask) > 10:
                    
    #                 in_board_mask = torch.zeros(temp_tensor.shape[0]).bool().cuda() 
    #                 in_board_mask[index_ray] = True 
    #                 # in_board_mask[index_ray][~inside_mask] = False, torch不能两重括号，用下面语法代替
    #                 temp = in_board_mask[index_ray]
    #                 temp[~inside_mask] = False
    #                 in_board_mask[index_ray] = temp
    #                 # print('board_points_uvs[inside_mask, :].shape')
    #                 # print(board_points_uvs[inside_mask, :].shape)
    #                 # print('sample_from_texture 2', flush=True)
    #                 # print('inside_mask.shape')
    #                 # print(inside_mask.shape)
    #                 # print(inside_mask)
    #                 # print(board_points_uvs.shape)
    #                 # print(torch.sum(inside_mask))
    #                 temp_tensor[in_board_mask, :] = self.sample_from_texture(board_points_uvs[inside_mask, :], board_texture)
    #                 # 3.
    #                 refract_img[board_valid_ray_mask, :] = temp_tensor

    #     refract_img[valid_ray_mask, :] = refract_img[valid_ray_mask, :] * accu_refract_attenuate[:, None].detach()  

    #     # # reflect image
    #     # # color from env map
    #     if self.with_env_map:
    #         reflect_img[valid_ray_mask, :], _ = self.sample_env_light(reflect_ray_directions, self.env_image.detach(), self.env_width, self.env_height)
        
    #     # # color from scene mesh
    #     # _, index_ray, _, _, points_uvs = self.diff_ray_triangle_intersec_trimesh(self.scene_mesh, reflect_ray_origins, reflect_ray_directions, None, self.scene_texture_uvs)
    #     # temp_tensor = reflect_img[valid_ray_mask, :]
    #     # temp_tensor[index_ray, :] = self.sample_from_texture(points_uvs, self.scene_texture)
    #     # reflect_img[valid_ray_mask, :] = temp_tensor

    #     if with_reflect:
    #         reflect_img[valid_ray_mask, :] = reflect_img[valid_ray_mask, :] * reflect_attenuate[:, None].detach()  
            
    #     refract_img = torch.clamp(refract_img, 0, 0.999999)
    #     reflect_img = torch.clamp(reflect_img, 0, 0.999999)

    #     # add gaussian for refract and reflect parts
    #     if self.training:
    #         kernel_size_1, sigma_1 = (3, 3), (1.5, 1.5)
    #         kernel_size_2, sigma_2 = (3, 3), (1.5, 1.5)
    #         patch_size = 200
    #         num_pixels = patch_size ** 2
    #         refract_img = refract_img.reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)
    #         refract_img = kornia.filters.gaussian_blur2d(refract_img, kernel_size_1, sigma_1, "replicate")
    #         refract_img = refract_img.permute(0, 2, 3, 1)
    #         reflect_img = reflect_img.reshape(-1,patch_size, patch_size, 3).permute(0, 3, 1, 2)
    #         reflect_img = kornia.filters.gaussian_blur2d(reflect_img, kernel_size_2, sigma_2, "replicate")
    #         reflect_img = reflect_img.permute(0, 2, 3, 1)

    #         # print(reflect_img.shape) 
    #         # print(refract_img.shape) 
    #         # vis_refract_img = refract_img.squeeze(0).detach().cpu().numpy() * 255
    #         # vis_refract_img = vis_refract_img.astype(np.uint8)
    #         # cv2.imwrite('../vis_refract_img.png', vis_refract_img)
    #         # vis_reflect_img = reflect_img.squeeze(0).detach().cpu().numpy() * 255
    #         # vis_reflect_img = vis_reflect_img.astype(np.uint8)
    #         # cv2.imwrite('../vis_reflect_img.png', vis_reflect_img)
    #         # exit()
        
    #     # im1 = refract_img.detach().cpu().reshape(patch_size, patch_size, 3).numpy()
    #     # im2 = reflect_img.detach().cpu().reshape(patch_size, patch_size, 3).numpy()
    #     # cv2.imshow("refract_img", im1[:, :, ::-1])
    #     # cv2.imshow("reflect_img", im2[:, :, ::-1])
    #     # cv2.waitKey(0)

    #     # accu_refract_attenuate < 0.01 for total reflection
    #     # print("refract_img nan num: {0}".format(torch.sum(torch.isnan(refract_img.reshape(-1)))))
    #     # print("reflect_img nan num: {0}".format(torch.sum(torch.isnan(reflect_img.reshape(-1)))))
    #     if with_reflect:
    #         composite_img = refract_img.reshape(-1, 3) + reflect_img.reshape(-1, 3)
    #     else:
    #         composite_img = refract_img.reshape(-1, 3)# + reflect_img.reshape(-1, 3)
    #     composite_img = torch.clamp(composite_img, 0, 0.999999)
    #     valid_ray_mask[valid_ray_mask][accu_refract_attenuate < 0.01] = False 
    #     composite_img[valid_ray_mask, :][accu_refract_attenuate < 0.01, :] = torch.zeros(3).cuda()

    #     # cv2.imshow("composite_img", composite_img.reshape(20, 20, 3).float().detach().cpu().numpy())
    #     # cv2.waitKey(0)
    #     # input()
    #     if not self.training:
    #         return composite_img.detach(), \
    #                valid_ray_mask.detach(), \
    #                relavent_vertex_idx_tensor.detach(), \
    #                all_board_hit_points_uvs.detach()

    #     return composite_img, \
    #            valid_ray_mask.detach(), \
    #            relavent_vertex_idx_tensor.detach(), \
    #            all_board_hit_points_uvs

    # def diff_mesh_ray_tracing(self,
    #                           sdf,
    #                           mesh_vertices_tensor,
    #                           mesh_faces_tensor,
    #                           mesh_vertex_normals_tensor,
    #                           face_idx_tensor, 
    #                           start_ray_origins,
    #                           start_ray_directions,
    #                           reflect_ray_directions,
    #                           refract_attenuate,
    #                           reflect_attenuate,
    #                           valid_ray_mask,
    #                           ray_board_indices,
    #                           material_ior,
    #                           is_no_corr):

    #     # torch.autograd.set_detect_anomaly(True) 
    #     total_vis_pcd = []
    #     vis_debug = False

    #     # reflect_ray_directions, reflect_ray_origins are first-bounce reflected ray 
    #     # ray_directions, ray_origins are second-bounce refracted ray  
    #     ################################################################################
    #     ray_origins_list = [start_ray_origins]
    #     ray_directions_list = [start_ray_directions]
    #     vertex_idx_list = []
    #     accu_refract_attenuate = torch.ones(ray_origins_list[0].shape[0]).cuda()
    #     reflect_attenuate = torch.ones(ray_origins_list[0].shape[0]).cuda()
    #     for ray_iter in range(face_idx_tensor.shape[0]):
    #         # print("ray_iter: {0}".format(ray_iter))
    #         face_idx = face_idx_tensor[ray_iter, :].detach()    
    #         ray_origins = ray_origins_list[ray_iter]
    #         ray_directions = ray_directions_list[ray_iter]
    #         if ray_iter == 2:
    #             break

    #         vertex_idx_list.append(torch.flatten(mesh_faces_tensor[face_idx]))
    #         vertices = mesh_vertices_tensor[torch.flatten(mesh_faces_tensor[face_idx]), :]
    #         vertices = vertices.reshape(-1, 3, 3)
    #         v0 = vertices[:, 0, :]
    #         v1 = vertices[:, 1, :]
    #         v2 = vertices[:, 2, :]
    #         plane_normal = torch.cross(v2 - v0, v2 - v1)
    #         plane_normal = plane_normal / torch.sqrt(torch.clamp(torch.sum(plane_normal*plane_normal, dim=1), min=1e-10 ).unsqueeze(1))
    #         plane_point = (v0 + v1 + v2) * 0.33333333333333

    #         hit_points = self.line_plane_collision(plane_normal, plane_point, ray_directions, ray_origins)
    #         if vis_debug:
    #             vis_pcd = o3d.geometry.PointCloud()
    #             vis_pcd.points = o3d.utility.Vector3dVector(hit_points.cpu().detach().numpy())
    #             colors = np.array([[0, 1 - ray_iter * 0.3, ray_iter * 0.3] for i in range(len(vis_pcd.points))])
    #             vis_pcd.colors = o3d.utility.Vector3dVector(colors[:, :])
    #             total_vis_pcd.append(vis_pcd)

    #         barycentric_weight = self.calculate_barycentric_weight(mesh_vertices_tensor, mesh_faces_tensor, hit_points, face_idx)
    #         barycentric_weight = barycentric_weight
    #         vertex_normals = mesh_vertex_normals_tensor[torch.flatten(mesh_faces_tensor[face_idx]), :]
    #         vertex_normals = vertex_normals.reshape(-1, 3, 3)
    #         v0_normals = vertex_normals[:, 0, :]
    #         v1_normals = vertex_normals[:, 1, :]
    #         v2_normals = vertex_normals[:, 2, :]
    #         hit_normals = barycentric_weight[:, 0][:, None] * v0_normals + \
    #             barycentric_weight[:, 1][:, None] * v1_normals + \
    #             barycentric_weight[:, 2][:, None] * v2_normals
    #         hit_normals = hit_normals / torch.sqrt(torch.clamp(torch.sum(hit_normals*hit_normals, dim=1), min=1e-10 ).unsqueeze(1))          
        
    #         if ray_iter % 2 == 0:
    #             refract_ray_directions, refract_attenuate, _ = self.refraction(ray_directions, hit_normals, eta1 = 1.0003, eta2 = material_ior)
    #         else:
    #             refract_ray_directions, refract_attenuate, _ = self.refraction(ray_directions, -hit_normals, eta1 = material_ior, eta2 = 1.0003)
            
    #         accu_refract_attenuate *= (1 - refract_attenuate)
    #         if ray_iter == 0:
    #             reflect_ray_directions = self.reflection(-ray_directions, hit_normals)
    #             reflect_ray_origins = hit_points
    #             reflect_attenuate = refract_attenuate
    #         ray_origins_list.append(hit_points.clone())
    #         ray_directions_list.append(refract_ray_directions.clone())

    #     ray_origins = ray_origins_list[2]
    #     ray_directions = ray_directions_list[2]
    #     relavent_vertex_idx_tensor = torch.cat(vertex_idx_list, dim=0)
    #     relavent_vertex_idx_tensor = torch.unique(relavent_vertex_idx_tensor, sorted=True)
        
    #     if vis_debug:
    #         vis_mesh = o3d.geometry.TriangleMesh()
    #         vis_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces_tensor.cpu().detach().numpy())
    #         vis_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
    #         total_vis_pcd.append(vis_mesh)
    #         o3d.visualization.draw_geometries(geometry_list = total_vis_pcd, width = 640, height = 480, window_name = "mesh_with_hit_points_diff", point_show_normal=False) 
    #     ################################################################################ 
      
    #     num_pixels = valid_ray_mask.shape[0]
    #     second_bounce_ray_origins, second_bounce_ray_directions = torch.zeros(num_pixels, 3).float().cuda(), torch.zeros(num_pixels, 3).float().cuda()
    #     second_bounce_ray_origins[valid_ray_mask, :] = ray_origins
    #     second_bounce_ray_directions[valid_ray_mask, :] = ray_directions

    #     refract_img, reflect_img = torch.zeros(num_pixels, 3).float().cuda(), torch.zeros(num_pixels, 3).float().cuda()
    #     all_board_hit_points_uvs = torch.FloatTensor(num_pixels, 2).zero_().cuda() / 0.0 # set all to NaN

    #     if torch.sum(valid_ray_mask).cpu().item() < 1 or refract_attenuate.numel() == 0:
    #         return refract_img, \
    #            valid_ray_mask.detach(), \
    #            relavent_vertex_idx_tensor.detach(), \
    #            all_board_hit_points_uvs.detach(), \
    #            second_bounce_ray_origins, \
    #            second_bounce_ray_directions

    #     # calculate refract image and reflect image
    #     ################################################################################ 
    #     if self.with_env_map:
    #         refract_img[valid_ray_mask, :], _ = self.sample_env_light(ray_directions, self.env_image.detach(), self.env_width, self.env_height) 
    #         reflect_img[valid_ray_mask, :], _ = self.sample_env_light(reflect_ray_directions, self.env_image.detach(), self.env_width, self.env_height)
  
    #     # 1. fetch color from scene mesh
    #     if self.with_global_scene is True:
    #         hit_points, index_ray, _, _, points_uvs = self.diff_ray_triangle_intersec_trimesh(self.scene_mesh, ray_origins, ray_directions, None, self.scene_texture_uvs)
    #         temp_tensor = refract_img[valid_ray_mask, :]
    #         temp_tensor[index_ray, :] = self.sample_from_texture(points_uvs, self.scene_texture)
    #         refract_img[valid_ray_mask, :] = temp_tensor

    #     # 2. fetch color and ray-cell correspondence from background pattern
    #     if is_no_corr is False:
    #         board_indices = torch.unique(ray_board_indices)
    #         for i in range(board_indices.shape[0]):
    #             board_idx = board_indices[i]
    #             # print('board_idx: {0}'.format(board_idx))
    #             board_mesh, board_texture_uvs, board_texture = self.board_mesh_list[board_idx], self.board_texture_uvs_list[board_idx], self.board_texture_list[board_idx]
    
    #             board_valid_ray_mask = valid_ray_mask.clone() 
    #             board_valid_ray_mask[ray_board_indices != board_idx] = False
    #             board_ray_origins = ray_origins[ray_board_indices[valid_ray_mask] == board_idx, :]
    #             board_ray_directions = ray_directions[ray_board_indices[valid_ray_mask] == board_idx, :]
    #             board_reflect_ray_origins = reflect_ray_origins[ray_board_indices[valid_ray_mask] == board_idx, :]
    #             board_reflect_ray_directions = reflect_ray_directions[ray_board_indices[valid_ray_mask] == board_idx, :]

    #             if with_board_scene:
    #                 board_scene_idx = board_idx // 15 
    #                 board_scene_hit_points, board_scene_index_ray, _, _, board_scene_points_uvs = self.diff_ray_triangle_intersec_trimesh(self.board_scene_mesh_list[board_scene_idx], board_ray_origins, board_ray_directions, None, self.board_scene_texture_uvs_list[board_scene_idx])
    #                 temp_tensor = refract_img[board_valid_ray_mask, :]
    #                 temp_tensor[board_scene_index_ray, :] = self.sample_from_texture(board_scene_points_uvs, self.board_scene_texture_list[board_scene_idx])
    #                 refract_img[board_valid_ray_mask, :] = temp_tensor 

    #                 if self.with_reflect:
    #                     board_reflect_scene_hit_points, board_reflect_scene_index_ray, _, _, board_reflect_scene_points_uvs = self.diff_ray_triangle_intersec_trimesh(self.board_scene_mesh_list[board_scene_idx], board_reflect_ray_origins, board_reflect_ray_directions, None, self.board_scene_texture_uvs_list[board_scene_idx])
    #                     reflect_temp_tensor = refract_img[board_valid_ray_mask, :]
    #                     reflect_temp_tensor[board_reflect_scene_index_ray, :] = self.sample_from_texture(board_reflect_scene_points_uvs, self.board_scene_texture_list[board_scene_idx])
    #                     reflect_img[board_valid_ray_mask, :] = temp_tensor 

    #             # board_hit_points, index_ray, _, _, board_points_uvs = self.diff_ray_triangle_intersec_trimesh(board_mesh, board_ray_origins, board_ray_directions, None, board_texture_uvs)
    #             board_hit_points, index_ray, board_points_uvs, inside_mask = self.diff_ray_plane_intersec_trimesh(board_mesh, board_ray_origins, board_ray_directions, board_texture_uvs)
                
    #             temp_board_points_uvs = torch.FloatTensor(board_ray_origins.shape[0], 2).zero_().cuda() / 0.0 # set all to NaN
    #             temp_board_points_uvs[index_ray, :] = board_points_uvs
    #             all_board_hit_points_uvs[board_valid_ray_mask, :] = temp_board_points_uvs
    #             temp_tensor = refract_img[board_valid_ray_mask, :]
                
    #             if board_points_uvs.nelement() > 0 and \
    #                 inside_mask.nelement() > 0 and \
    #                 torch.sum(inside_mask) > 10:
                    
    #                 in_board_mask = torch.zeros(temp_tensor.shape[0]).bool().cuda() 
    #                 in_board_mask[index_ray] = True 
    #                 # 注意：in_board_mask[index_ray][~inside_mask] = False, torch导数传递不能两重括号，用下面语法代替
    #                 temp = in_board_mask[index_ray]
    #                 temp[~inside_mask] = False
    #                 in_board_mask[index_ray] = temp
    #                 temp_tensor[in_board_mask, :] = self.sample_from_texture(board_points_uvs[inside_mask, :], board_texture)
    #                 refract_img[board_valid_ray_mask, :] = temp_tensor

    #     refract_img[valid_ray_mask, :] = refract_img[valid_ray_mask, :] * accu_refract_attenuate[:, None].detach()  
    #     if self.with_reflect:
    #         reflect_img[valid_ray_mask, :] = reflect_img[valid_ray_mask, :] * reflect_attenuate[:, None].detach()  
    #     refract_img = torch.clamp(refract_img, 0, 0.999999)
    #     reflect_img = torch.clamp(reflect_img, 0, 0.999999)
    #     ################################################################################ 

    #     # add gaussian for refract and reflect parts
    #     if self.training:
    #         kernel_size_1, sigma_1 = (3, 3), (1.5, 1.5)
    #         kernel_size_2, sigma_2 = (3, 3), (1.5, 1.5)
    #         patch_size = 300
    #         num_pixels = patch_size ** 2
    #         refract_img = refract_img.reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)
    #         refract_img = kornia.filters.gaussian_blur2d(refract_img, kernel_size_1, sigma_1, "replicate")
    #         refract_img = refract_img.permute(0, 2, 3, 1)
    #         reflect_img = reflect_img.reshape(-1,patch_size, patch_size, 3).permute(0, 3, 1, 2)
    #         reflect_img = kornia.filters.gaussian_blur2d(reflect_img, kernel_size_2, sigma_2, "replicate")
    #         reflect_img = reflect_img.permute(0, 2, 3, 1)

    #     if self.with_reflect:
    #         composite_img = refract_img.reshape(-1, 3) + reflect_img.reshape(-1, 3)
    #     else:
    #         composite_img = refract_img.reshape(-1, 3)
    #     composite_img = torch.clamp(composite_img, 0, 0.999999)
    #     valid_ray_mask[valid_ray_mask][accu_refract_attenuate < 0.01] = False 
    #     composite_img[valid_ray_mask, :][accu_refract_attenuate < 0.01, :] = torch.zeros(3).cuda()

    #     if not self.training:
    #         return composite_img.detach(), \
    #                valid_ray_mask.detach(), \
    #                relavent_vertex_idx_tensor.detach(), \
    #                all_board_hit_points_uvs.detach(), \
    #                second_bounce_ray_origins, \
    #                second_bounce_ray_directions
    #     return composite_img, \
    #            valid_ray_mask.detach(), \
    #            relavent_vertex_idx_tensor.detach(), \
    #            all_board_hit_points_uvs, \
    #            second_bounce_ray_origins, \
    #            second_bounce_ray_directions



    # def mesh_ray_tracing(self,
    #                      sdf,
    #                      sdf_mesh,
    #                      calculate_vertex_normal,
    #                      ray_origins,
    #                      ray_directions,
    #                      only_mask_loss,
    #                      material_ior):

    #     sdf_mesh_face_num = sdf_mesh.faces.shape[0]
    #     merged_mesh = self.merge_mesh([sdf_mesh, self.env_map_mesh])
    #     mesh_vertices_tensor = torch.from_numpy(merged_mesh.vertices).float().cuda()
    #     mesh_faces_tensor = torch.from_numpy(merged_mesh.faces).long().cuda()
    #     mesh_vertex_faces_tensor = torch.from_numpy(np.array(merged_mesh.vertex_faces)).long().cuda()
    #     mesh_face_normals, mesh_vertex_normals = calculate_vertex_normal(mesh_vertices_tensor, mesh_faces_tensor, mesh_vertex_faces_tensor)
    #     vis_debug = False
    #     if vis_debug:
    #         vis_mesh = o3d.geometry.TriangleMesh()
    #         vis_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces_tensor.cpu().detach().numpy())
    #         vis_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
    #         vis_pcd = o3d.geometry.PointCloud()
    #         vis_pcd.points = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
    #         vis_pcd.normals = o3d.utility.Vector3dVector(mesh_vertex_normals.cpu().detach().numpy())
    #         o3d.visualization.draw_geometries(geometry_list = [vis_mesh, vis_pcd], width = 640, height = 480, window_name = "mesh_with_hit_points_2", point_show_normal=False) 

    #     # torch.autograd.set_detect_anomaly(True)
    #     num_pixels, _ = ray_directions.shape
    #     # print("num_pixels: {0}".format(num_pixels))

    #     reflect_ray_directions = torch.zeros_like(ray_directions).cuda()
    #     refract_ray_directions = torch.zeros_like(ray_directions).cuda()
    #     reflect_attenuate = torch.zeros(num_pixels).cuda()
    #     refract_attenuate = torch.zeros(num_pixels).cuda()
    #     accu_refract_attenuate = torch.ones(num_pixels).cuda()
    #     valid_ray_mask = torch.ones(num_pixels).bool().cuda()
    #     hit_points = ray_origins.clone()
    #     hit_normals = ray_directions.clone()

    #     total_vis_pcd = []
    #     face_idx_list = []
    #     ray_origins_list = []
    #     ray_directions_list = []

    #     for ray_iter in range(3):
    #         hit_mask = torch.zeros(num_pixels).bool().cuda()      

    #         points, index_ray, face_idx, points_normals, _ = self.ray_triangle_intersec_trimesh(merged_mesh, ray_origins, ray_directions, mesh_vertex_normals, None)
    #         # print(points_normals.shape)
    #         # print(ray_directions[index_ray].shape)
    #         if ray_iter == 2:
    #             points = points[face_idx >= sdf_mesh_face_num]
    #             points_normals = points_normals[face_idx >= sdf_mesh_face_num]
    #             index_ray = index_ray[face_idx >= sdf_mesh_face_num]
    #             face_idx = face_idx[face_idx >= sdf_mesh_face_num]
    #         else:
    #             first_hit_index_ray = index_ray[face_idx < sdf_mesh_face_num]
    #             ray_surf_angle = torch.abs(torch.bmm(points_normals[:, None, :], ray_directions[index_ray][:, :, None]).squeeze())
    #             ray_length = torch.norm(points - ray_origins[index_ray], p=2, dim=1)
    #             # ray_mask = (ray_length > 0.05) & (ray_surf_angle > 0.2) # for cow
    #             # ray_mask = (ray_length > 0.0) & (ray_surf_angle > 0.5) # for hand
    #             # ray_mask = (ray_length > 0.0) & (ray_surf_angle > 0.2)
    #             ray_mask = (ray_length > 0.0) # for DRT
    #             points = points[ray_mask]
    #             points_normals = points_normals[ray_mask]
    #             index_ray = index_ray[ray_mask]
    #             face_idx = face_idx[ray_mask]

    #             points = points[face_idx < sdf_mesh_face_num]
    #             points_normals = points_normals[face_idx < sdf_mesh_face_num]
    #             index_ray = index_ray[face_idx < sdf_mesh_face_num]
    #             face_idx = face_idx[face_idx < sdf_mesh_face_num]
    #         hit_mask[index_ray] = True
    #         if ray_iter == 0:
    #             first_hit_mask = torch.zeros_like(hit_mask).bool().cuda()
    #             first_hit_mask[first_hit_index_ray] = True
    #             if only_mask_loss:
    #                 if vis_debug:
    #                     valid_ray_mask = valid_ray_mask & hit_mask
    #                     hit_points[valid_ray_mask, :] = points
    #                     hit_normals[valid_ray_mask, :] = points_normals
    #                     vis_pcd = o3d.geometry.PointCloud()
    #                     vis_pcd.points = o3d.utility.Vector3dVector(hit_points[valid_ray_mask, :].cpu().detach().numpy())
    #                     vis_pcd.normals = o3d.utility.Vector3dVector(hit_normals[valid_ray_mask, :].cpu().detach().numpy())
    #                     colors = np.array([[0, 1 - ray_iter * 0.3, ray_iter * 0.3] for i in range(len(vis_pcd.points))])
    #                     vis_pcd.colors = o3d.utility.Vector3dVector(colors[:, :])
    #                     total_vis_pcd.append(vis_pcd)
    #                     o3d.visualization.draw_geometries(geometry_list = total_vis_pcd, width = 640, height = 480, window_name = "mesh_with_hit_points", point_show_normal=False) 
    #                 return None, first_hit_mask.detach(), None, None, None, None, None, None
    #         valid_ray_mask = valid_ray_mask & hit_mask

    #         # print("valid_ray num: {0}".format(torch.sum(valid_ray_mask).cpu().item()))
    #         # print(points.shape)
    #         # print(face_idx.shape)
    #         hit_points[valid_ray_mask, :] = points
    #         hit_normals[valid_ray_mask, :] = points_normals

    #         temp_face_idx = torch.zeros(num_pixels).long().cuda()
    #         temp_ray_originals = torch.zeros(num_pixels, 3).float().cuda()
    #         temp_ray_directions = torch.zeros(num_pixels, 3).float().cuda()
    #         temp_face_idx[valid_ray_mask] = face_idx
    #         face_idx_list.append(temp_face_idx)
    #         ray_origins_list.append(ray_origins)
    #         ray_directions_list.append(ray_directions)
    #         if ray_iter == 2:
    #             break
            
    #         if vis_debug:
    #             vis_pcd = o3d.geometry.PointCloud()
    #             vis_pcd.points = o3d.utility.Vector3dVector(hit_points[valid_ray_mask, :].cpu().detach().numpy())
    #             vis_pcd.normals = o3d.utility.Vector3dVector(hit_normals[valid_ray_mask, :].cpu().detach().numpy())
    #             colors = np.array([[0, 1 - ray_iter * 0.4, ray_iter * 0.4] for i in range(len(vis_pcd.points))])
    #             vis_pcd.colors = o3d.utility.Vector3dVector(colors[:, :])
    #             total_vis_pcd.append(vis_pcd)

    #         # hit_normals = self.calculate_sdf_normal(sdf, hit_points, retain_graph=False) 
                
    #         if ray_iter % 2 == 0:
    #             refract_ray_directions[valid_ray_mask], refract_attenuate[valid_ray_mask], total_reflect_mask = self.refraction(ray_directions[valid_ray_mask], hit_normals[valid_ray_mask], eta1 = 1.0003, eta2 = material_ior)
    #         else:
    #             refract_ray_directions[valid_ray_mask], refract_attenuate[valid_ray_mask], total_reflect_mask = self.refraction(ray_directions[valid_ray_mask], -hit_normals[valid_ray_mask], eta1 = material_ior, eta2 = 1.0003)
    #         accu_refract_attenuate[valid_ray_mask] *= (1 - refract_attenuate[valid_ray_mask])
    #         # valid_ray_mask[valid_ray_mask][total_reflect_mask] = False
    #         if ray_iter == 0:
    #             reflect_ray_directions[valid_ray_mask] = self.reflection(-ray_directions[valid_ray_mask], hit_normals[valid_ray_mask])
    #             reflect_attenuate = refract_attenuate

    #         ray_directions = refract_ray_directions
    #         ray_directions = ray_directions / torch.sqrt(torch.clamp(torch.sum(ray_directions.detach()*ray_directions.detach(), dim=1), min=1e-10 ).unsqueeze(1))
    #         ray_origins = hit_points + ray_directions * torch.ones(num_pixels, 1).cuda() * 0.01
    #         ray_directions[~valid_ray_mask] = torch.zeros(3).cuda()

    #     for i in range(len(face_idx_list)):
    #         face_idx_list[i] = face_idx_list[i][valid_ray_mask].unsqueeze(0)
    #         ray_origins_list[i] = ray_origins_list[i][valid_ray_mask].unsqueeze(0)
    #         ray_directions_list[i] = ray_directions_list[i][valid_ray_mask].unsqueeze(0)
    #     face_idx_tensor = torch.cat(face_idx_list, dim=0)
    #     ray_origins_tensor = torch.cat(ray_origins_list, dim=0)
    #     ray_directions_tensor = torch.cat(ray_directions_list, dim=0)
    #     # print(face_idx_tensor.shape)
    #     # print(ray_origins_tensor.shape)
    #     # print(ray_directions_tensor.shape)

    #     if vis_debug:
    #         vis_mesh = o3d.geometry.TriangleMesh()
    #         vis_mesh.triangles = o3d.utility.Vector3iVector(mesh_faces_tensor.cpu().detach().numpy())
    #         vis_mesh.vertices = o3d.utility.Vector3dVector(mesh_vertices_tensor.cpu().detach().numpy())
    #         total_vis_pcd.append(vis_mesh)
    #         o3d.visualization.draw_geometries(geometry_list = total_vis_pcd, width = 640, height = 480, window_name = "mesh_with_hit_points", point_show_normal=False) 

    #     # print("accu_refract_attenuate[valid_ray_mask]:")
    #     # print(accu_refract_attenuate[valid_ray_mask])
    #     return valid_ray_mask.detach(), \
    #         first_hit_mask.detach(), \
    #         face_idx_tensor.detach(), \
    #         ray_origins_tensor.detach(), \
    #         ray_directions_tensor.detach(), \
    #         reflect_ray_directions[valid_ray_mask].detach(), \
    #         accu_refract_attenuate[valid_ray_mask].squeeze().detach(), \
    #         reflect_attenuate[valid_ray_mask].squeeze().detach()