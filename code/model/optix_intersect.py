import torch
from torch.utils.cpp_extension import load
# from config import optix_include, optix_ld
print('start optix')
optix_include = "/home/2TB/xjm/NVIDIA-OptiX-SDK-6.5.0-linux64/include/"
optix_ld = "/usr/lib/"
optix = load(name="optix", sources=["/home/2TB/xjm/transparentobjectrecon/code/model/optix_extend.cpp"],
    extra_include_paths=[optix_include], extra_ldflags=["-L"+optix_ld, "-loptix_prime"])
print('finish optix')

import trimesh
import numpy as np
import imageio
from PIL import Image

debug = False
#render resolution
resy=960
resx=1280
Float = torch.float64
device='cuda'
extIOR, intIOR = 1.00029, 1.5

class Ray:
    def __init__(self, origin, direction, ray_ind = None):
        self.origin = origin
        self.direction = direction
        if ray_ind is None:
            self.ray_ind = torch.nonzero(torch.ones(len(origin))).squeeze()
        else:
            self.ray_ind = ray_ind
        assert(len(self.direction)==len(self.ray_ind))

    def select(self, mask):
        return Ray(self.origin[mask],self.direction[mask],self.ray_ind[mask])

    def __len__(self):
        return len(self.ray_ind)

class Intersection:
    def __init__(self, u, v, t, n, ray, faces_ind):
        self.u = u
        self.v = v
        self.t = t
        self.n = n
        self.ray = ray
        self.faces_ind = faces_ind
        assert(len(n)==len(ray))
        
    def __len__(self):
        return len(self.ray)

class Scene:
    def __init__(self, mesh, cuda_device = 0):
        self.optix_mesh = optix.optix_mesh(cuda_device)
        self.update_mesh(mesh)

    def update_mesh(self, mesh):
        # assert mesh.is_watertight
        # print('update_mesh start')
        self.mesh = mesh
        self.vertices = torch.tensor(mesh.vertices, dtype=Float, device=device)
        self.faces = torch.tensor(mesh.faces, dtype=torch.long, device=device)
        self.triangles = self.vertices[self.faces] #[Fx3x3]

        opt_v = self.vertices.detach().to(torch.float32).to(device)
        opt_F = self.faces.detach().to(torch.int32).to(device)
        # print(opt_F.shape)
        # print(opt_v.shape)
        self.optix_mesh.update_mesh(opt_F, opt_v)
        # print('update_mesh finish')

    def update_verticex(self, vertices:torch.Tensor):
        opt_v = vertices.detach().to(torch.float32).to(device)
        self.optix_mesh.update_vert(opt_v)
        self.mesh.vertices = vertices.detach().cpu().numpy()
        self.vertices = vertices
        self.triangles = vertices[self.faces] #[Fx3x3]
        self.init_VN()

    def optix_intersect(self, ray:Ray):
        optix_o = ray.origin.to(torch.float32).to(device)
        optix_d = ray.direction.to(torch.float32).to(device)
        optix_ray = torch.cat([optix_o, optix_d], dim=1)
        T, faces_ind = self.optix_mesh.intersect(optix_ray)
        # hitted = T>0 
        hitted = torch.nonzero(T>0).flatten()
        points = (optix_o + optix_d * T[:, None])
        points = points[hitted, :]
        faces_ind = faces_ind[hitted] 
        return points, faces_ind.to(torch.long), hitted

def save_torch(name, img:torch.Tensor):
    image = (255 * (img-img.min()) / (img.max()-img.min())).to(torch.uint8)
    imageio.imsave(name, image.view(resy,resx,-1).cpu())

def torch2pil(img:torch.Tensor):
    image = (255 * (img-img.min()) / (img.max()-img.min())).to(torch.uint8)
    image = image.view(resy,resx,-1).cpu().numpy()
    if image.shape[2] == 1: image = image[:,:,0]
    return Image.fromarray(image)

