import torch
import torch.nn as nn
import numpy as np
import kornia
import math
import time

from utils import rend_util
import utils.plots as plt
from utils.plots import get_grid_uniform
from utils.general import fix_mesh_raw
from torch.nn.functional import grid_sample

from model.embedder import *
from model.ray_tracing import RayTracing
from model.sample_network import SampleNetwork

# from model.optix_intersect import Scene
# from model.optix_intersect import Ray
# import pymesh
import cv2
import open3d as o3d
from plot_image_grid import image_grid

from skimage import measure
import trimesh
from sklearn.neighbors import KDTree
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
from pytorch3d.renderer import (MeshRasterizer, MeshRenderer,
                                RasterizationSettings, PointLights,
                                look_at_view_transform, TexturesVertex, Materials)
from pytorch3d.renderer.blending import BlendParams
from pytorch3d.renderer.mesh.shader import (SoftSilhouetteShader)
from pytorch3d.renderer.cameras import (PerspectiveCameras,
                                        OpenGLPerspectiveCameras,
                                        FoVPerspectiveCameras)
from pytorch3d.renderer.blending import (
    BlendParams,
    hard_rgb_blend,
    sigmoid_alpha_blend,
    softmax_rgb_blend,
)
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from camera_visualization import plot_camera_scene
from pytorch3d.structures import Meshes
import matplotlib.pyplot

from largesteps.parameterize import from_differential
# from largesteps.geometry import compute_matrix
# from largesteps.geometry import compute_matrix, laplacian_uniform
import nvdiffrast.torch as dr

@torch.jit.script
def dot(v1:torch.Tensor, v2:torch.Tensor, keepdim:bool = False):
    ''' v1, v2: [n,3]'''
    result = v1[:,0]*v2[:,0] + v1[:,1]*v2[:,1] + v1[:,2]*v2[:,2]
    if keepdim:
        return result.view(-1,1)
    return result

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

class ImplicitNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            geometric_init=True,
            bias=0.8,
            skip_in=(),
            weight_norm=True,
            multires=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out + feature_vector_size]

        self.embed_fn = None
        if multires > 0:
            embed_fn, input_ch = get_embedder(multires)
            self.embed_fn = embed_fn
            dims[0] = input_ch

        self.num_layers = len(dims)
        self.skip_in = skip_in

        for l in range(0, self.num_layers - 1):
            if l + 1 in self.skip_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                elif multires > 0 and l == 0:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.constant_(lin.weight[:, 3:], 0.0)
                    torch.nn.init.normal_(lin.weight[:, :3], 0.0, np.sqrt(2) / np.sqrt(out_dim))
                elif multires > 0 and l in self.skip_in:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
                    torch.nn.init.constant_(lin.weight[:, -(dims[0] - 3):], 0.0)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=100)

    def forward(self, input, compute_grad=False):
        if self.embed_fn is not None:
            input = self.embed_fn(input)

        x = input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.skip_in:
                x = torch.cat([x, input], 1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x

    def gradient(self, x):
        x.requires_grad_(True)
        y = self.forward(x)[:,:1]
        d_output = torch.ones_like(y, requires_grad=False, device=y.device)
        gradients = torch.autograd.grad(
            outputs=y,
            inputs=x,
            grad_outputs=d_output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return gradients.unsqueeze(1)

class RenderingNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            mode,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0
    ):
        super().__init__()

        self.mode = mode
        dims = [d_in + feature_vector_size] + dims + [d_out]

        self.embedview_fn = None
        if multires_view > 0:
            embedview_fn, input_ch = get_embedder(multires_view)
            self.embedview_fn = embedview_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, points, normals, view_dirs, feature_vectors):
        if self.embedview_fn is not None:
            view_dirs = self.embedview_fn(view_dirs)

        if self.mode == 'idr':
            rendering_input = torch.cat([points, view_dirs, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_view_dir':
            rendering_input = torch.cat([points, normals, feature_vectors], dim=-1)
        elif self.mode == 'no_normal':
            rendering_input = torch.cat([points, view_dirs, feature_vectors], dim=-1)

        x = rendering_input

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x

class DisplacementNetwork(nn.Module):
    def __init__(
            self,
            feature_vector_size,
            d_in,
            d_out,
            dims,
            weight_norm=True,
            multires_view=0
    ):
        super().__init__()

        dims = [d_in] + dims + [d_out]

        self.embed_fn = None
        if multires_view > 0:
            embed_fn, input_ch = get_embedder(multires_view)
            self.embed_fn = embed_fn
            dims[0] += (input_ch - 3)

        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            out_dim = dims[l + 1]
            lin = nn.Linear(dims[l], out_dim)

            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.relu = nn.ReLU()#nn.LeakyReLU(0.1)#
        self.tanh = nn.Tanh()

    def forward(self, points):
        
        if self.embed_fn is not None:
            points = self.embed_fn(points)

        x = points
        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.relu(x)

        x = self.tanh(x)
        return x
    
# Only this function had to be changed to account for multi networks (weight tensors have aditionally a network dimension)
def _calculate_fan_in_and_fan_out(tensor):
    fan_in = tensor.size(-1)
    fan_out = tensor.size(-2)
    return fan_in, fan_out

# All of the above functions are copy pasted from PyTorch's codebase. This is nessecary because of the adapted fan in computation
def _calculate_correct_fan(tensor, mode):
    mode = mode.lower()
    valid_modes = ['fan_in', 'fan_out']
    if mode not in valid_modes:
        raise ValueError("Mode {} not supported, please use one of {}".format(mode, valid_modes))

    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    return fan_in if mode == 'fan_in' else fan_out

def calculate_gain(nonlinearity, param=None):
    linear_fns = ['linear', 'conv1d', 'conv2d', 'conv3d', 'conv_transpose1d', 'conv_transpose2d', 'conv_transpose3d']
    if nonlinearity in linear_fns or nonlinearity == 'sigmoid':
        return 1
    elif nonlinearity == 'tanh':
        return 5.0 / 3
    elif nonlinearity == 'relu':
        return math.sqrt(2.0)
    elif nonlinearity == 'leaky_relu':
        if param is None:
            negative_slope = 0.01
        elif not isinstance(param, bool) and isinstance(param, int) or isinstance(param, float):
            # True/False are instances of int, hence check above
            negative_slope = param
        else:
            raise ValueError("negative_slope {} not a valid number".format(param))
        return math.sqrt(2.0 / (1 + negative_slope ** 2))
    else:
        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

def kaiming_uniform_(tensor, a=0, mode='fan_in', nonlinearity='leaky_relu'):
    fan = _calculate_correct_fan(tensor, mode)
    gain = calculate_gain(nonlinearity, a)
    std = gain / math.sqrt(fan)
    bound = math.sqrt(3.0) * std  # Calculate uniform bounds from standard deviation
    with torch.no_grad():
        return tensor.uniform_(-bound, bound)

class MultiNetworkLinear(nn.Module):
    rng_state = None

    def __init__(self, num_networks, in_features, out_features, bias=True):
        
        super(MultiNetworkLinear, self).__init__()
        self.num_networks = num_networks
        self.in_features = in_features
        self.out_features = out_features
        # weight is created in reset_parameters()
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_networks, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()
        self.implementation = 'bmm'
        
    def reset_parameters(self):
        self.weight = nn.Parameter(torch.Tensor(self.num_networks, self.out_features, self.in_features))
        
        kaiming_uniform_(self.weight, a=math.sqrt(5), nonlinearity='relu')
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias, -bound, bound)
        
    def forward(self, x):
         
        # x = num_networks x batch_size x in_features
        batch_size = x.size(1)
        if self.num_networks > 1:
            if self.implementation == 'bmm':
                weight_transposed = self.weight.permute(0, 2, 1) # num_networks x in_features x out_features
                # num_networks x batch_size x in_features @ num_networks x in_features x out_features = num_networks x batch_size x out_features
                product = torch.bmm(x, weight_transposed)
                bias_view = self.bias.unsqueeze(1)
            result = product + bias_view # (num_networks * batch_size) x out_features
        return result.view(self.num_networks, batch_size, self.out_features)

class MultiDisplacementNetwork(nn.Module):
    def __init__(self,
                 num_networks,              
                 geodesic_indices, 
                 geodesic_weights,
                 d_in,
                 d_out,
                 dims,
                 weight_norm=True,
                 multires_view=0):
        super(MultiDisplacementNetwork, self).__init__()
        
        print(f'num_networks: {num_networks}\n')
        self.num_networks = num_networks
        self.geodesic_indices = geodesic_indices
        self.geodesic_weights = geodesic_weights
        dims = [d_in] + dims + [d_out]

        if multires_view > 0:
            embed_fn, input_ch = get_embedder(multires_view)
            self.embed_fn = embed_fn
            dims[0] += (input_ch - 3)
        
        def new_linear_layer(in_features, out_features):
            return MultiNetworkLinear(self.num_networks, in_features, out_features)

        self.num_layers = len(dims)
        layers = []
        for l in range(0, self.num_layers - 1):
            print('---------------')
            print(dims[l])
            print(dims[l+1])
            print('---------------')
            lin = new_linear_layer(dims[l], dims[l+1])
            lin = nn.utils.weight_norm(lin)
            setattr(self, "lin_" + str(l), lin)
            if l < self.num_layers - 2: 
                layers += [getattr(self, "lin_" + str(l)), nn.ReLU()]
            else:
                layers += [getattr(self, "lin_" + str(l)), nn.Tanh()]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.embed_fn(x)
        feature_size = x.shape[1]
        x = x.reshape(self.num_networks, -1, feature_size)
        x = self.layers(x)
        x = x.reshape(-1, 3)
        
        return x 

class IDRNetwork(nn.Module):
    def __init__(self, cluster_vertex_start_end, geodesic_indices, geodesic_weights, conf):
        super().__init__()

        print('IDRNetwork init')

        self.feature_vector_size = conf.get_int('feature_vector_size')
        self.implicit_network = ImplicitNetwork(self.feature_vector_size, **conf.get_config('implicit_network'))
        # self.rendering_network = RenderingNetwork(self.feature_vector_size, **conf.get_config('rendering_network'))
        if cluster_vertex_start_end is not None:
            self.displacement_network = DisplacementNetwork(self.feature_vector_size, **conf.get_config('displacement_network'))
            self.multi_displacement_network = MultiDisplacementNetwork(cluster_vertex_start_end.shape[0] - 1, geodesic_indices, geodesic_weights, **conf.get_config('multi_displacement_network'))
        self.ray_tracer = RayTracing(**conf.get_config('ray_tracer'))
        
        self.object_bounding_sphere = conf.get_float('ray_tracer.object_bounding_sphere')
        self.resolution = conf.get_int('resolution')
        self.img_res = conf.get_list('img_res') 

        self.cluster_vertex_start_end = cluster_vertex_start_end
        self.geodesic_indices = geodesic_indices
        self.geodesic_weights = geodesic_weights
        
        self.mean_rgb = conf.get_list('mean_rgb')
        self.std_rgb = conf.get_list('std_rgb')
        self.mean_rgb = torch.from_numpy(np.array(self.mean_rgb).reshape(1, 3)).float().cuda()
        self.std_rgb = torch.from_numpy(np.array(self.std_rgb).reshape(1, 3)).float().cuda()
        
        self.count = 0
        self.sdf_mesh = None
        self.displaced_sdf_mesh = None
        self.prev_mesh_vertices_tensor = None
        self.E2F = None
        self.mean_len = None

        self.gt_vertices_tensor_idr = None
        self.gt_vertices_tensor_sc = None
        data_dir = conf.get('data_dir')
        gt_mesh_idr = trimesh.load(data_dir + '/idr_surface.ply', use_embree=True, process=False)
        self.gt_vertices_tensor_idr = torch.from_numpy(np.array(gt_mesh_idr.vertices)).float().cuda()

        # gt_mesh_sc = trimesh.load(data_dir + '/space_carving_mesh_origin_new.ply', use_embree=True, process=False)
        # self.gt_vertices_tensor_sc = torch.from_numpy(np.array(gt_mesh_sc.vertices)).float().cuda()

        self.glctx = dr.RasterizeGLContext()

    def get_surface_mesh(self, sdf, resolution=100, return_mesh=False):

        grid = get_grid_uniform(resolution)
        points = grid['grid_points']

        z = []
        with torch.no_grad():
            for i, pnts in enumerate(torch.split(points, resolution * resolution, dim=0)):
                z.append(sdf(pnts).cpu().numpy())
        z = np.concatenate(z, axis=0)

        if (not (np.min(z) > 0 or np.max(z) < 0)):

            z = z.astype(np.float32)

            torch.cuda.synchronize()
            start = time.time()
            verts, faces, normals, values = measure.marching_cubes_lewiner(
                volume=z.reshape(grid['xyz'][1].shape[0], grid['xyz'][0].shape[0],
                                grid['xyz'][2].shape[0]).transpose([1, 0, 2]),
                level=0,
                spacing=(grid['xyz'][0][2] - grid['xyz'][0][1],
                        grid['xyz'][0][2] - grid['xyz'][0][1],
                        grid['xyz'][0][2] - grid['xyz'][0][1]))

            verts = verts + np.array([grid['xyz'][0][0], grid['xyz'][1][0], grid['xyz'][2][0]])

            meshexport = trimesh.Trimesh(verts, faces, normals)
            torch.cuda.synchronize()
            end = time.time()
            print(f"marching cube time: {end - start} s")

            return meshexport
        return None

    # numpy  get rays
    def get_rays_np(self, H, W, focal, c2w):
        """Get ray origins, directions from a pinhole camera."""
        i, j = np.meshgrid(np.arange(W, dtype=np.float32),
                        np.arange(H, dtype=np.float32), indexing='xy')
        # 也可以理解为摄像机坐标下z=1的平面上的点
        dirs = np.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -np.ones_like(i)], -1) 
        rays_d = np.sum(dirs[..., np.newaxis, :] * c2w[:3, :3], -1)
        rays_o = np.broadcast_to(c2w[:3, -1], np.shape(rays_d))
        return rays_o, rays_d

    def calculate_mean_edge_length(self, vertices_tensor, faces_tensor):
 
        vertices = vertices_tensor[torch.flatten(faces_tensor), :]
        vertices = vertices.reshape(-1, 3, 3)
        v0 = vertices[:, 0, :]#face_num*3
        v1 = vertices[:, 1, :]#face_num*3
        v2 = vertices[:, 2, :]#face_num*3
        v0v1 = v0 - v1
        d0 = torch.sqrt(torch.bmm(v0v1[:, None, :], v0v1[:, :, None]).squeeze())
        v0v2 = v0 - v2
        d1 = torch.sqrt(torch.bmm(v0v2[:, None, :], v0v2[:, :, None]).squeeze())
        v1v2 = v1 - v2
        d2 = torch.sqrt(torch.bmm(v1v2[:, None, :], v1v2[:, :, None]).squeeze())
        d = torch.stack([d0, d1, d2])
      
        return d.mean().detach()

    def calculate_vertex_normal(self, vertices_tensor, faces_tensor, vertex_faces_tensor):

        vertices = vertices_tensor[torch.flatten(faces_tensor), :]
        vertices = vertices.reshape(-1, 3, 3)
        v0 = vertices[:, 0, :]#face_num*3
        v1 = vertices[:, 1, :]#face_num*3
        v2 = vertices[:, 2, :]#face_num*3
        plane_normals_no_normalize = torch.cross(v2 - v0, v2 - v1)
        face_norm = torch.sqrt(torch.clamp(torch.sum(plane_normals_no_normalize*plane_normals_no_normalize, dim=1), min=1e-10 ).unsqueeze(1))
        face_normals = plane_normals_no_normalize / face_norm

        weight_tensor = torch.where(vertex_faces_tensor > 0, torch.ones_like(vertex_faces_tensor), torch.zeros_like(vertex_faces_tensor)).detach()
        weighted_plane_normals = weight_tensor[:, :, None] * plane_normals_no_normalize[vertex_faces_tensor]
        vertex_normals = torch.sum(weighted_plane_normals, dim=1)
        vertex_normals = vertex_normals / torch.sqrt(torch.clamp(torch.sum(vertex_normals*vertex_normals, dim=1), min=1e-10 ).unsqueeze(1))

        return face_normals, vertex_normals

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

    def render_silhouette_pytorch3d(self, vertices, faces, K, R, T, W, H, scale):

        image_size = (H, W)

        K_cpu = K.detach().cpu()
        cameras = PerspectiveCameras(in_ndc=False,
                                     focal_length=((K_cpu[0, 0] * scale, K_cpu[1, 1] * scale),),
                                     principal_point=((K_cpu[0, 2] * scale, K_cpu[1, 2] * scale),),
                                     R=R[None, :, :], 
                                     T=T[None, :],
                                     image_size=((H, W),),
                                     device='cuda')

        # To blend the 100 faces we set a few parameters which control the opacity and the sharpness of
        # edges. Refer to blending.py for more details.
        blend_params = BlendParams(sigma=1e-5, gamma=1e-5)
        silhouette_raster_settings = RasterizationSettings(
            image_size=image_size,
            blur_radius=np.log(1. / 1e-4 - 1.) * blend_params.sigma, 
            faces_per_pixel=100, 
            max_faces_per_bin=30000,
            perspective_correct=False,
            cull_backfaces=True
        )
        # Create a silhouette mesh renderer by composing a rasterizer and a shader.
        silhouette_renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=cameras,
                raster_settings=silhouette_raster_settings
            ),
            shader=SoftSilhouetteShader(blend_params=blend_params)
        )

        zero_meshes = Meshes(verts=[torch.zeros_like(vertices)], faces=[faces])
        src_meshes = zero_meshes.offset_verts(vertices)
        lights = PointLights(ambient_color=((1.0, 1.0, 1.0),), diffuse_color=((0.0, 0.0, 0.0),),
                             specular_color=((0.0, 0.0, 0.0),), device='cuda', location=[[0.0, 0.0, -3.0]])
        silhouette_images = silhouette_renderer(src_meshes, cameras=cameras, lights=lights)
        return silhouette_images[0, ..., 3]
      
    def projection(self, fx, fy, cx, cy, w, h, z_near = 0.01, z_far = 100.0):
        return np.array([[2.0 * fx / w,             0,           -(2.0 * cx / w - 1.0),                                 0],
                         [           0,  2.0 * fy / h,            (2.0 * cy / h - 1.0),                                 0],
                         [           0,             0,  -(z_far+z_near)/(z_far-z_near),  -(2*z_far*z_near)/(z_far-z_near)],
                         [           0,             0,                              -1,                                 0]]).astype(np.float32)
                         
    def RT(self, R, T):
        mat = np.identity(4)
        mat[0:3,0:3] = R
        mat[0:3,3] = T
        return mat
    
    # Transform vertex positions to clip space
    def transform_pos(self, mtx, pos):
        t_mtx = torch.from_numpy(mtx).cuda() if isinstance(mtx, np.ndarray) else mtx
        # (x,y,z) -> (x,y,z,1)
        posw = torch.cat([pos, torch.ones([pos.shape[0], 1]).cuda()], axis=1)
        return torch.matmul(posw, t_mtx.t())[None, ...]

    # def render(glctx, mtx, pos, pos_idx, vtx_col, col_idx, resolution: int):
    def render_silhouette_nvdiffrast(self, vertices, faces, K, R, T, W, H, scale):
        K = K.cpu().numpy()
        R = R.cpu().numpy()
        T = T.cpu().numpy()
        proj  = self.projection(K[0, 0] * scale, K[1, 1] * scale, K[0, 2] * scale, K[1, 2] * scale, W, H)
        r_mv  = self.RT(R, T)
        vtx_col = torch.ones(1, 1).float().cuda()
        col_idx = torch.zeros_like(faces).int().cuda()
        pos_idx = faces.int() 
        r_mvp = np.matmul(proj, r_mv).astype(np.float32)
        pos_clip    = self.transform_pos(r_mvp, vertices)
        rast_out, _ = dr.rasterize(self.glctx, pos_clip, pos_idx, resolution=(H, W))
        color, _    = dr.interpolate(vtx_col[None, ...], rast_out, col_idx)
        color       = dr.antialias(color, rast_out, pos_clip, pos_idx)
        color = torch.flip(color, [1])
        return color

    def secant(self, sdf_low, sdf_high, z_low, z_high, cam_loc, ray_directions, sdf):
        ''' Runs the secant method for interval [z_low, z_high] for n_secant_steps '''

        n_secant_steps = 6

        z_pred = - sdf_low * (z_high - z_low) / (sdf_high - sdf_low) + z_low
        for i in range(n_secant_steps):
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

    def point_sampler(self, sdf, ray_directions, \
                      sampler_min_max, origin_points, n_steps):
        
        # origin_points [1, 2048, 3]
        # near_far_t [1, 2048, 2]
        ''' Sample the ray in a given range and run secant on rays which have sign transition '''

        num_pixels, _ = ray_directions.shape
        n_total_pxl = num_pixels
        sampler_pts = torch.zeros(n_total_pxl, 3).cuda().float()
        sampler_dists = torch.zeros(n_total_pxl).cuda().float()

        intervals_dist = torch.linspace(0, 1, steps=n_steps).cuda().view(1, -1)
        pts_intervals = sampler_min_max[:, 0][:, None] + intervals_dist * (sampler_min_max[:, 1] - sampler_min_max[:, 0])[:, None]
        points = origin_points[:, None, :] + pts_intervals[:, :, None] * ray_directions[:, None, :]

        # Get the non convergent rays
        points = points.reshape((-1, n_steps, 3))
        pts_intervals = pts_intervals.reshape((-1, n_steps))

        sdf_val_all = []
        for pnts in torch.split(points.reshape(-1, 3), 10000, dim=0):
            sdf_val_all.append(sdf(pnts))
        sdf_val = torch.cat(sdf_val_all).reshape(-1, n_steps).detach()
        tmp = torch.sign(sdf_val) * torch.arange(n_steps, 0, -1).cuda().float().reshape((1, n_steps))  # Force argmin to return the first min value
        sampler_pts_ind = torch.argmin(tmp, -1)
        sampler_pts = points[torch.arange(points.shape[0]), sampler_pts_ind, :]
        sampler_dists = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind]

        net_surface_pts = (sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind] < 0)

        # Get Network object mask
        # Run Secant method
        secant_pts = net_surface_pts
        n_secant_pts = secant_pts.sum()
        if n_secant_pts > 0:
            # Get secant z predictions
            z_high = pts_intervals[torch.arange(pts_intervals.shape[0]), sampler_pts_ind][secant_pts]
            sdf_high = sdf_val[torch.arange(sdf_val.shape[0]), sampler_pts_ind][secant_pts]
            z_low = pts_intervals[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]
            sdf_low = sdf_val[secant_pts][torch.arange(n_secant_pts), sampler_pts_ind[secant_pts] - 1]

            cam_loc_secant = origin_points.reshape((-1, 3))[secant_pts]
            ray_directions_secant = ray_directions.reshape((-1, 3))[secant_pts]
            z_pred_secant = self.secant(sdf_low, sdf_high, z_low, z_high, cam_loc_secant, ray_directions_secant, sdf)

            # Get points
            sampler_pts[secant_pts] = cam_loc_secant + z_pred_secant.unsqueeze(-1) * ray_directions_secant
            # sampler_dists[secant_pts] = z_pred_secant

        return sampler_pts

    def extract_mesh_from_sdf(self, with_simp):

        print('extract_mesh_from_sdf')

        self.sdf_mesh = self.get_surface_mesh(sdf=lambda x: self.implicit_network(x)[:, 0],
                                              resolution=self.resolution)

        if with_simp:
            fixed_verts, fixed_faces = fix_mesh_raw(self.sdf_mesh.vertices, self.sdf_mesh.faces, "normal")
            self.sdf_mesh = trimesh.Trimesh(fixed_verts, fixed_faces)
        self.mesh_vertices_tensor = torch.from_numpy(np.array(self.sdf_mesh.vertices)).float().cuda()

        return

    def forward(self, idr_input):

        # print('start forward')
        gauss_7x7 = kornia.filters.GaussianBlur2d((7, 7), (3.0, 3.0))

        # Parse model input
        epoch, iteration = idr_input['epoch'], idr_input['iteration']
        intrinsics, pose, uv = idr_input["intrinsics"], idr_input["pose"], idr_input["uv"] 
        
        object_mask, erode_object_mask = idr_input["object_mask"].reshape(-1).cuda(), idr_input["erode_object_mask"].reshape(-1).cuda()
        mask_scale = idr_input["mask_scale"][0, 0].item()
        patch_size = idr_input["patch_size"][0, 0].item()
        board_indices = idr_input["board_indices"]
        manual_anno_indices = idr_input["manual_anno_indices"]
        material_ior = idr_input["material_ior"][0]

        explicit_vertices_tensor_u = idr_input['explicit_vertices_u_for_large_step']
        if explicit_vertices_tensor_u is not None:
            M = idr_input['M']
            self.mesh_vertices_tensor = from_differential(M, explicit_vertices_tensor_u, 'Cholesky')
            self.sdf_mesh.vertices = self.mesh_vertices_tensor.detach().cpu().numpy()
            
        explicit_vertices_tensor = idr_input['explicit_vertices']
        if explicit_vertices_tensor is not None:
            self.mesh_vertices_tensor = explicit_vertices_tensor
            self.sdf_mesh.vertices = self.mesh_vertices_tensor.detach().cpu().numpy()
        
        self.is_mesh_sdf_single_mlp = idr_input['is_mesh_sdf_single_mlp'] 
        self.is_idr_sdf_single_mlp = idr_input['is_idr_sdf_single_mlp'] 
        self.is_displacements_single_mlp = idr_input['is_displacements_single_mlp'] 
        self.is_displacements_multi_mlp = idr_input['is_displacements_multi_mlp'] 
        self.use_our_corr = idr_input['use_our_corr']
        # print(f'is_mesh_sdf_single_mlp: {self.is_mesh_sdf_single_mlp}')
        # print(f'is_idr_sdf_single_mlp: {self.is_idr_sdf_single_mlp}')
        # print(f'is_displacements_single_mlp: {self.is_displacements_single_mlp}')
        # print(f'is_displacements_multi_mlp: {self.is_displacements_multi_mlp}')
        if self.is_mesh_sdf_single_mlp:
            self.extract_mesh_from_sdf(True) 
        if self.is_idr_sdf_single_mlp:
            self.extract_mesh_from_sdf(True) 
        if self.is_displacements_single_mlp: 
            pass
        if self.is_displacements_multi_mlp: 
            pass

        ray_dirs, cam_loc = rend_util.get_camera_params(uv, pose, intrinsics)
        batch_size, batch_num_pixels, _ = ray_dirs.shape
        cam_loc = cam_loc[:, None, :].expand(-1, batch_num_pixels, -1)
        ray_board_indices = board_indices[:, None, :].expand(-1, batch_num_pixels, -1)
        ray_dirs = ray_dirs.reshape(-1, 3)
        cam_loc = cam_loc.reshape(-1, 3)
        ray_board_indices = ray_board_indices.reshape(-1)
        num_pixels, _ = ray_dirs.shape
        ray_origins = cam_loc.reshape(-1, 3).detach()
        ray_directions = ray_dirs.reshape(-1, 3).detach()

        if self.is_mesh_sdf_single_mlp or self.is_idr_sdf_single_mlp:
            self.sdf_mesh.vertices = self.mesh_vertices_tensor.detach().cpu().numpy()
            if False:
                sdf_normals = self.calculate_sdf_normal(sdf=lambda x: self.implicit_network(x)[:, 0], 
                                                        hit_points=self.mesh_vertices_tensor, 
                                                        retain_graph=False)
                sampler_min_max = torch.zeros(self.mesh_vertices_tensor.shape[0], 2).cuda()
                sampler_min_max[:, 0] = torch.ones(self.mesh_vertices_tensor.shape[0]).cuda() * (-0.002)
                sampler_min_max[:, 1] = torch.ones(self.mesh_vertices_tensor.shape[0]).cuda() * (0.002)
                sampler_points = self.point_sampler(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                    ray_directions=-sdf_normals.detach(),
                                                    sampler_min_max=sampler_min_max.detach(),
                                                    origin_points=self.mesh_vertices_tensor.detach(),
                                                    n_steps=6)
                self.mesh_vertices_tensor = sampler_points.detach()
                self.sdf_mesh.vertices = self.mesh_vertices_tensor.detach().cpu().numpy()
        else:
            pass

        # ---------------------------------------------------------------------
        # make vertices differentiable
        # ---------------------------------------------------------------------
        if self.is_mesh_sdf_single_mlp or self.is_idr_sdf_single_mlp:
            sdf_normals = self.calculate_sdf_normal(sdf=lambda x: self.implicit_network(x)[:, 0], 
                                                    hit_points=self.mesh_vertices_tensor, 
                                                    retain_graph=False)
            sdf_tensor = self.implicit_network(self.mesh_vertices_tensor)
            sdf_tensor_detach = sdf_tensor.clone().detach()
            self.mesh_vertices_tensor = self.mesh_vertices_tensor - sdf_normals.detach() * (sdf_tensor - sdf_tensor_detach)
            
        # vis_pcd = o3d.geometry.PointCloud()
        # vis_pcd.points = o3d.utility.Vector3dVector(self.mesh_vertices_tensor.cpu().detach().numpy())
        # vis_pcd.normals = o3d.utility.Vector3dVector(sdf_normals.cpu().detach().numpy())
        # o3d.visualization.draw_geometries(geometry_list = [vis_pcd], width = 640, height = 480, window_name = "vis_pcd", point_show_normal=True) 

        # all_camera_actors = []
        # for i in range(pose.shape[0]):
        #     camera_actor = create_camera_actor(scale = 0.2)
        #     all_camera_actors.append(camera_actor.transform(pose[i, :, :].detach().cpu().numpy()))
        # object_pcd = o3d.geometry.PointCloud()
        # object_pcd.points = o3d.utility.Vector3dVector(mesh_vertices_tensor.detach().cpu().numpy())
        # all_camera_actors.append(object_pcd)
        # o3d.visualization.draw_geometries(geometry_list=all_camera_actors, width = 640, height = 480, window_name = "mesh_with_hit_points")
        
        # ---------------------------------------------------------------------
        # calculate silhouette_loss, normal_consistency_loss, 
        # laplacian_smoothing_loss
        # --------------------------------------------------------------------- 
        mesh_faces_tensor = torch.from_numpy(self.sdf_mesh.faces).long().cuda().detach()
        target_silhouette = idr_input["silhouette_loss_object_mask"].float().cuda()
        hard_mining_mask = idr_input["hard_mining_mask"].float().cuda()
        target_intrinsics = idr_input["silhouette_loss_intrinsics"].float().cuda()
        target_poses = idr_input["silhouette_loss_pose"].float().cuda()
        p3d_pose = target_poses.clone()
        p3d_pose[:, :, 0] = p3d_pose[:, :, 0] * -1
        p3d_pose[:, :, 1] = p3d_pose[:, :, 1] * -1
        p3d_pose_opengl = target_poses.clone()
        p3d_pose_opengl[:, :, 1] = p3d_pose_opengl[:, :, 1] * -1
        p3d_pose_opengl[:, :, 2] = p3d_pose_opengl[:, :, 2] * -1

        # add displacement offset to vertices
        displacement_values = torch.zeros(self.mesh_vertices_tensor.shape[0]).cuda()
        if self.is_displacements_single_mlp :
            displacement_values = self.get_displacement_value(self.mesh_vertices_tensor.detach(), is_multi_mlp=False)
            displaced_mesh_vertices_tensor = self.mesh_vertices_tensor.detach() + displacement_values
            self.displaced_sdf_mesh = self.sdf_mesh
            self.displaced_sdf_mesh.vertices = displaced_mesh_vertices_tensor.detach().cpu().numpy()
        elif self.is_displacements_multi_mlp:
            displacement_values = self.get_displacement_value(self.mesh_vertices_tensor.detach(), is_multi_mlp=True)
            neighbour_num = self.geodesic_indices.shape[1]
            displacement_values = self.geodesic_weights.reshape(-1)[:, None] * displacement_values[self.geodesic_indices.reshape(-1), :]
            displacement_values = displacement_values.reshape(-1, neighbour_num, 3)
            displacement_values = torch.sum(displacement_values, dim=1)
            displaced_mesh_vertices_tensor = self.mesh_vertices_tensor.detach() + displacement_values
            self.displaced_sdf_mesh = self.sdf_mesh
            self.displaced_sdf_mesh.vertices = displaced_mesh_vertices_tensor.detach().cpu().numpy()
        else:
            displaced_mesh_vertices_tensor = self.mesh_vertices_tensor
            self.displaced_sdf_mesh = self.sdf_mesh
            self.displaced_sdf_mesh.vertices = displaced_mesh_vertices_tensor.detach().cpu().numpy()
            displacement_values = None

        displaced_silhouette_loss = None 
        displaced_laplacian_smoothing_loss = None
        displaced_edge_loss = None
        displaced_silhouette_loss = torch.zeros(1).cuda()
        # start displaced_silhouette_loss
        if True:
            for i in range(batch_size):
                manual_anno_idx = manual_anno_indices[i, 0]
                extrinsics_opengl = torch.linalg.inv(p3d_pose_opengl)
                R_cuda_opengl = extrinsics_opengl[i, 0:3, 0:3]
                t_cuda_opengl = extrinsics_opengl[i, 0:3, 3]

                extrinsics = torch.linalg.inv(p3d_pose)
                K_cuda, R_cuda, t_cuda= target_intrinsics[i, :, :], extrinsics[i, 0:3, 0:3], extrinsics[i, 0:3, 3]
                # For Pytorch3D, it is x*R+t not R*x+t, so we transpose the R
                R_cuda_t = torch.transpose(R_cuda, 0, 1)
                W, H = (int)(self.img_res[1] // mask_scale), (int)(self.img_res[0] // mask_scale)

                # displaced_predicted_silhouette = self.render_silhouette_pytorch3d(displaced_mesh_vertices_tensor, mesh_faces_tensor, K_cuda, R_cuda_t, t_cuda, W, H, 1.0 / mask_scale)
                displaced_predicted_silhouette = self.render_silhouette_nvdiffrast(displaced_mesh_vertices_tensor.clone(), mesh_faces_tensor, K_cuda, R_cuda_opengl, t_cuda_opengl, W, H, 1.0 / mask_scale)
                curr_target_silhouette = target_silhouette[i, ...]
                curr_hard_mining_mask = hard_mining_mask[i, ...].bool()

                displaced_predicted_silhouette = displaced_predicted_silhouette.reshape(1, H, W, 1).permute(0, 3, 1, 2)
                curr_target_silhouette = curr_target_silhouette.reshape(1, H, W, 1).permute(0, 3, 1, 2)
                gauss_predicted_silhouette = gauss_7x7(displaced_predicted_silhouette)
                gauss_curr_target_silhouette = gauss_7x7(curr_target_silhouette)

                # gauss_predicted_silhouette = gauss_predicted_silhouette.squeeze(0).permute(1, 2, 0).reshape(-1, 1)
                # gauss_predicted_silhouette[curr_hard_mining_mask.reshape(-1), :] = 0.5
                # predicted_silhouette_image = gauss_predicted_silhouette.reshape(H, W, 1).float().detach().cpu().numpy()
                # predicted_silhouette_image *= 250.0
                # predicted_silhouette_image = predicted_silhouette_image.astype(np.uint8)
                # cv2.imwrite(f"../1_predicted_silhouette_image_{i}.png", predicted_silhouette_image)
                # curr_target_silhouette_image = gauss_curr_target_silhouette.squeeze(0).permute(1, 2, 0).reshape(H, W, 1).float().detach().cpu().numpy()
                # curr_target_silhouette_image *= 250.0
                # curr_target_silhouette_image = curr_target_silhouette_image.astype(np.uint8)
                # cv2.imwrite(f"../1_curr_target_silhouette_image_{i}.png", curr_target_silhouette_image)
                # cv2.imwrite(f"../1_diff_silhouette_image_{i}.png", np.abs(predicted_silhouette_image - curr_target_silhouette_image))
                # exit(0)

                gauss_predicted_silhouette = gauss_predicted_silhouette.squeeze(0).permute(1, 2, 0).reshape(-1, 1)
                gauss_curr_target_silhouette = gauss_curr_target_silhouette.squeeze(0).permute(1, 2, 0).reshape(-1, 1)
                pixel_weight = torch.ones_like(gauss_curr_target_silhouette).cuda()
                pixel_weight[curr_hard_mining_mask.reshape(-1)] *= 4
                temp = ((pixel_weight * (gauss_predicted_silhouette - gauss_curr_target_silhouette)) ** 2).mean() / (float)(batch_size)
                
                displaced_silhouette_loss += temp
        # finish displaced_silhouette_loss

        displaced_zero_meshes = Meshes(verts=[torch.zeros_like(displaced_mesh_vertices_tensor)], faces=[mesh_faces_tensor])
        displaced_src_meshes = displaced_zero_meshes.offset_verts(displaced_mesh_vertices_tensor)
        # displaced_laplacian_smoothing_loss = mesh_laplacian_smoothing(displaced_src_meshes, method="cot")
        # displaced_edge_loss = mesh_edge_loss(displaced_src_meshes)
        displaced_laplacian_smoothing_loss = 25.0 * mesh_laplacian_smoothing(displaced_src_meshes, method="cot")
        displaced_edge_loss = 10.0 * mesh_edge_loss(displaced_src_meshes)

        e1 = self.sdf_mesh.vertices[self.sdf_mesh.edges[:,0]]
        e2 = self.sdf_mesh.vertices[self.sdf_mesh.edges[:,1]]
        
        self.mean_len = np.linalg.norm(e1-e2, axis=1).mean()
        # if self.mean_len is None:
        #     self.mean_len = np.linalg.norm(e1-e2, axis=1).mean()
        #     print(f'mean_len of mesh: {self.mean_len}')
        
        E2F_index = torch.from_numpy(np.array(self.sdf_mesh.face_adjacency)).long().cuda().detach()
        self.E2F = mesh_faces_tensor[E2F_index] #[Ex2x3]
        v0 = displaced_mesh_vertices_tensor[self.E2F[:,0,0]]
        v1 = displaced_mesh_vertices_tensor[self.E2F[:,0,1]]
        v2 = displaced_mesh_vertices_tensor[self.E2F[:,0,2]]
        EF1N = torch.cross(v1-v0, v2-v0) #[Ex3]
        EF1N = EF1N / torch.clamp(EF1N.norm(p=2, dim=1, keepdim=True), min=1e-10)
        v0 = displaced_mesh_vertices_tensor[self.E2F[:,1,0]]
        v1 = displaced_mesh_vertices_tensor[self.E2F[:,1,1]]
        v2 = displaced_mesh_vertices_tensor[self.E2F[:,1,2]]
        EF2N = torch.cross(v1-v0, v2-v0) #[Ex3]    
        EF2N = EF2N / torch.clamp(EF2N.norm(p=2, dim=1, keepdim=True), min=1e-10)
        angle = dot(EF1N, EF2N)
        
        # ---------------------------------------------------------------------
        # calculate composite_rgb_image
        # --------------------------------------------------------------------- 
        ray_origins = cam_loc.reshape(-1, 3).detach()
        ray_directions = ray_dirs.reshape(-1, 3).detach()
        self.implicit_network.eval()
        with torch.no_grad():
            # valid_ray_idx, face_idx_tensor = self.ray_tracer.mesh_ray_tracing_optix(mesh=self.displaced_sdf_mesh,
            #                                                                         calculate_vertex_normal=lambda *args: self.calculate_vertex_normal(*args),
            #                                                                         ray_origins=ray_origins.clone(),
            #                                                                         ray_directions=ray_directions.clone(),
            #                                                                         only_mask_loss=False,
            #                                                                         material_ior=material_ior)
            valid_ray_idx, face_idx_tensor = self.ray_tracer.mesh_ray_tracing_trimesh(sdf_mesh=self.displaced_sdf_mesh,
                                                                                      calculate_vertex_normal=lambda *args: self.calculate_vertex_normal(*args),
                                                                                      ray_origins=ray_origins.clone(),
                                                                                      ray_directions=ray_directions.clone(),
                                                                                      material_ior=material_ior)
            valid_ray_mask = torch.zeros(num_pixels).bool().cuda() 
            valid_ray_mask[valid_ray_idx] = True
                
        # print("find_minimum_sdf_points")
        # minimum_sdf_points, minimum_sdf_points_dis, updated_hit_mask = self.ray_tracer.find_minimum_sdf_points(sdf=lambda x: self.implicit_network(x)[:, 0],
        #                                                                                                        ray_origins=ray_origins,
        #                                                                                                        ray_directions=ray_directions,
        #                                                                                                        in_mask=first_hit_mask)
        # network_object_mask = updated_hit_mask
        # points = minimum_sdf_points
        # dists = minimum_sdf_points_dis
        
        self.implicit_network.train()
        vis_debug = False
        # if vis_debug:
        #     vis_pcd_finished = o3d.geometry.PointCloud()
        #     vis_pcd_finished.points = o3d.utility.Vector3dVector(points[not network_object_mask, :].cpu().detach().numpy())
        #     o3d.visualization.draw_geometries(geometry_list = [vis_pcd_finished], width = 640, height = 480, window_name = "minimum_sdf_points", point_show_normal=False) 
        composite_rgb = None
        composite_rgb_mask = None
        rela_displacement_values = None
        board_hit_points_uvs = None
        second_bounce_ray_origins = None
        second_bounce_ray_directions = None
        mesh_faces_tensor = torch.from_numpy(self.sdf_mesh.faces).long().cuda().detach()
        mesh_vertex_faces_tensor = torch.from_numpy(np.array(self.sdf_mesh.vertex_faces)).long().cuda().detach()
        mesh_face_normals_tensor, mesh_vertex_normals_tensor = self.calculate_vertex_normal(displaced_mesh_vertices_tensor, mesh_faces_tensor, mesh_vertex_faces_tensor)  
        relavent_face_idx_tensor = torch.arange(displaced_mesh_vertices_tensor.shape[0]).cuda()
        start = time.time()
        if not self.is_idr_sdf_single_mlp:
            composite_rgb, composite_rgb_mask, \
            relavent_vertex_idx_tensor, board_hit_points_uvs, \
            second_bounce_ray_origins, second_bounce_ray_directions = \
                self.ray_tracer.diff_mesh_ray_tracing(mesh_vertices_tensor=displaced_mesh_vertices_tensor,
                                                    mesh_faces_tensor=mesh_faces_tensor,
                                                    mesh_vertex_normals_tensor=mesh_vertex_normals_tensor,
                                                    face_idx_tensor=face_idx_tensor, 
                                                    valid_ray_mask=valid_ray_mask,
                                                    start_ray_origins=ray_origins[valid_ray_mask, :], 
                                                    start_ray_directions=ray_directions[valid_ray_mask, :],
                                                    ray_board_indices=ray_board_indices,
                                                    material_ior=material_ior,
                                                    use_our_corr=self.use_our_corr,
                                                    patch_size=patch_size)
        else:
            composite_rgb, composite_rgb_mask, \
            relavent_vertex_idx_tensor, board_hit_points_uvs, \
            second_bounce_ray_origins, second_bounce_ray_directions = \
                self.ray_tracer.diff_mesh_ray_tracing_idr(sdf=lambda x: self.implicit_network(x)[:, 0],
                                                          calculate_sdf_normal=self.calculate_sdf_normal,
                                                          point_sampler=self.point_sampler,
                                                          mesh_vertices_tensor=displaced_mesh_vertices_tensor,
                                                          mesh_faces_tensor=mesh_faces_tensor,
                                                          mesh_vertex_normals_tensor=mesh_vertex_normals_tensor,
                                                          face_idx_tensor=face_idx_tensor, 
                                                          valid_ray_mask=valid_ray_mask,
                                                          start_ray_origins=ray_origins[valid_ray_mask, :], 
                                                          start_ray_directions=ray_directions[valid_ray_mask, :],
                                                          ray_board_indices=ray_board_indices,
                                                          material_ior=material_ior,
                                                          use_our_corr=self.use_our_corr,
                                                          patch_size=patch_size)

        # composite_rgb = (composite_rgb - self.mean_rgb) / self.std_rgb
   
        # ---------------------------------------------------------------------
        # calculate eikonal_loss
        # ---------------------------------------------------------------------
        eikonal_points_g = None
        if ((self.is_mesh_sdf_single_mlp or self.is_idr_sdf_single_mlp) and self.training):
            # Sample points for the eikonal loss
            eik_bounding_box = self.object_bounding_sphere
            n_eik_points = num_pixels // 4
            eikonal_points = torch.empty(n_eik_points, 3).uniform_(-eik_bounding_box, eik_bounding_box).cuda()
        
            eikonal_points_g = self.implicit_network.gradient(eikonal_points)
            eikonal_points_g = eikonal_points_g[:, 0, :]

        output = {
            'iteration': iteration,

            'eikonal_points_g': eikonal_points_g,
            'displaced_silhouette_loss': displaced_silhouette_loss,
            'displaced_laplacian_smoothing_loss': displaced_laplacian_smoothing_loss,
            'displaced_edge_loss': displaced_edge_loss,
            'angle': angle,
            'mean_len': self.mean_len,
            'displacement_values': displacement_values,
            
            'rgb_values': composite_rgb,
            'rgb_values_mask': composite_rgb_mask,
            'patch_size': patch_size,
            'composite_rgb_mask': composite_rgb_mask,
            'object_mask': object_mask,
            'erode_object_mask': erode_object_mask,

            'board_hit_points_uvs': board_hit_points_uvs,
            'mesh_vertices_tensor': displaced_mesh_vertices_tensor,
            'gt_vertices_tensor_idr': self.gt_vertices_tensor_idr,
            'gt_vertices_tensor_sc': self.gt_vertices_tensor_sc,
            'material_ior': material_ior,
            
            'second_bounce_ray_origins': second_bounce_ray_origins, 
            'second_bounce_ray_directions': second_bounce_ray_directions
        }

        return output

    def get_rbg_value(self, points, view_dirs):
        output = self.implicit_network(points)
        g = self.implicit_network.gradient(points)
        normals = g[:, 0, :]

        feature_vectors = output[:, 1:]
        rgb_vals = self.rendering_network(points, normals, view_dirs, feature_vectors)

        return rgb_vals

    def get_displacement_value(self, points, is_multi_mlp=False):

        if is_multi_mlp:
            displacement_vals = self.multi_displacement_network(points)
        else:
            displacement_vals = self.displacement_network(points)

        return displacement_vals * 0.04
