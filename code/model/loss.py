from cgi import print_form
import torch
from torch import nn
from torch.nn import functional as F
import cv2
from kornia.losses import ssim_loss as get_ssim_loss
import kornia
import numpy as np
import math

import torch
import torchvision
from pytorch3d.loss import (
    chamfer_distance, 
    mesh_edge_loss, 
    mesh_laplacian_smoothing, 
    mesh_normal_consistency,
)
# from largesteps.parameterize import from_differential, to_differential
# from largesteps.geometry import compute_matrix

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self, resize=None):
        super(VGGPerceptualLoss, self).__init__()
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

    def forward(self, input, target, feature_layers=[3], style_layers=[]):
        if input.shape[1] != 3:
            input = input.repeat(1, 3, 1, 1)
            target = target.repeat(1, 3, 1, 1)
        # input = (input-self.mean) / self.std
        # target = (target-self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x = input
        y = target
        for i, block in enumerate(self.blocks):
            x = block(x)
            y = block(y)
            if i in feature_layers:
                loss += torch.nn.functional.l1_loss(x, y)
            if i in style_layers:
                act_x = x.reshape(x.shape[0], x.shape[1], -1)
                act_y = y.reshape(y.shape[0], y.shape[1], -1)
                gram_x = act_x @ act_x.permute(0, 2, 1)
                gram_y = act_y @ act_y.permute(0, 2, 1)
                loss += torch.nn.functional.l1_loss(gram_x, gram_y)
        return loss

class IDRLoss(nn.Module):

    def __init__(self, \
        large_step, \
        sdf_single_mlp, \
        use_our_corr, \
        use_DRT_corr, \
        eikonal_weight, \
        vh_mesh_weight, \
        color_weight, \
        corr_weight, \
        outside_border_weight, \
        zero_displacement_weight, \
        displaced_silhouette_weight, \
        displaced_normal_consistency_weight, \
        displaced_laplacian_smoothing_weight, \
        displaced_edge_weight, \
        alpha):

        super().__init__()

        self.eikonal_weight = eikonal_weight    
        self.color_weight = color_weight
        self.corr_weight = corr_weight
        self.outside_border_weight = outside_border_weight
        self.zero_displacement_weight = zero_displacement_weight
        self.displaced_silhouette_weight = displaced_silhouette_weight
        self.displaced_normal_consistency_weight = displaced_normal_consistency_weight
        self.displaced_laplacian_smoothing_weight = displaced_laplacian_smoothing_weight
        self.displaced_edge_weight = displaced_edge_weight
        self.vh_mesh_weight = vh_mesh_weight
        self.alpha = alpha
        
        self.l1_loss = nn.L1Loss(reduction='mean')
        # self.get_vgg_loss = VGGPerceptualLoss().cuda()
        self.large_step = large_step
        self.sdf_single_mlp = sdf_single_mlp
        self.use_our_corr = use_our_corr
        self.use_DRT_corr = use_DRT_corr

        mean_rgb = [0.485, 0.456, 0.406]
        std_rgb = [0.229, 0.224, 0.225]
        self.mean_rgb = torch.from_numpy(np.array(mean_rgb)).view(1, 3).float().cuda()
        self.std_rgb = torch.from_numpy(np.array(std_rgb)).view(1, 3).float().cuda()

    def get_rgb_loss(self, rgb_values, rgb_gt, composite_rgb_mask, object_mask, rgb_values_mask):
        if (composite_rgb_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values.reshape(-1, 3)[composite_rgb_mask & object_mask & rgb_values_mask, :]
        rgb_gt = rgb_gt.reshape(-1, 3)[composite_rgb_mask & object_mask & rgb_values_mask, :]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt)

        return rgb_loss

    def get_corr_loss(self, src, target, object_mask):

        mask = (~torch.isnan(src[:, 0])) & (~torch.isnan(target[:, 0])) & (~torch.isinf(target[:, 0])) & object_mask
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        src_uv = src[mask, :]
        target = target[mask, :].detach()
        target_uv = target[:, 0:2]
        target_radius = target[:, 2]
        dists = (src_uv - target_uv).norm(2, dim=1)

        u_dists_detach = torch.abs(src_uv[:, 0] - target_uv[:, 0]).detach()
        v_dists_detach = torch.abs(src_uv[:, 1] - target_uv[:, 1]).detach()

        dist_mask = (u_dists_detach > target_radius) | \
                    (v_dists_detach > target_radius)
        if dist_mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        corr_loss = dists[dist_mask].mean()

        return corr_loss

    def get_outside_border_loss(self, src, target, object_mask):

        mask = (~torch.isnan(src[:, 0])) & (torch.isinf(target[:, 0])) & object_mask
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        
        src_uv = src[mask, :]
        target_uv = (torch.ones(1, 2) * 0.5).cuda()
        target_radius = 0.5
        dists = (src_uv - target_uv).norm(2, dim=1)

        u_dists_detach = torch.abs(src_uv[:, 0] - target_uv[:, 0]).detach()
        v_dists_detach = torch.abs(src_uv[:, 1] - target_uv[:, 1]).detach()
        dist_mask = (u_dists_detach <= target_radius) & \
                    (v_dists_detach <= target_radius) 
        if dist_mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        outside_border_loss = -dists[dist_mask].mean()

        return outside_border_loss

    def get_silhouette_loss(self, silhouette, gt_silhouette):
    
        value1 = silhouette.float().reshape(-1, 1).cuda()
        value2 = gt_silhouette.float().reshape(-1, 1).cuda()
        silhouette_loss = self.l1_loss(value1, value2)
        return silhouette_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_zero_displacement_loss(self, displacement_values):
        if displacement_values.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        zero_displacement_loss = (displacement_values.norm(p=2, dim=1)).mean()
        return zero_displacement_loss

    def get_convex_loss(self, gradgrad_theta):
        if gradgrad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        gradgrad_below_zero = torch.where(gradgrad_theta > 0, gradgrad_theta, torch.zeros_like(gradgrad_theta).cuda())
        convex_loss = (gradgrad_below_zero.norm(2, dim=1)).mean()
        return convex_loss

    def get_smooth_loss_v2(self, angle):
        smooth_loss = 1.0-torch.log(1.0 + angle)
        return smooth_loss.sum()
    
    def get_smooth_loss(self, adf_face_normals):
        if adf_face_normals.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        normal_diff = torch.bmm(adf_face_normals[:, 0, :][:, None, :], adf_face_normals[:, 1, :][:, :, None]).squeeze()
        smooth_loss = -torch.log(1.0 + normal_diff)
        return smooth_loss.mean()

    def get_mask_loss(self, sdf_output, composite_rgb_mask, object_mask):
        mask = ~(composite_rgb_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def forward(self, model_outputs, ground_truth, epoch):

        rgb_gt = ground_truth['rgb'].cuda()
        rgb_gt = rgb_gt.float() / 255.0
        # rgb_gt = (rgb_gt - self.mean_rgb) / self.std_rgb
        
        patch_size = model_outputs['patch_size']
        composite_rgb_mask = model_outputs['composite_rgb_mask']
        object_mask = model_outputs['object_mask']
        rgb_values_mask = model_outputs['rgb_values_mask']
        material_ior = model_outputs['material_ior']
        ior_regularization = (material_ior - 1.52) * (material_ior - 1.52) 
        ior_regularization = 0.001 * ior_regularization

        num_pixel = rgb_gt.shape[1]
        patch_num_pixels = patch_size * patch_size

        eikonal_loss = torch.zeros(1).cuda()
        displaced_silhouette_loss = torch.zeros(1).cuda()
        displaced_normal_consistency_loss = torch.zeros(1).cuda()
        displaced_laplacian_smoothing_loss = torch.zeros(1).cuda()
        displaced_edge_loss = torch.zeros(1)
        mask_loss = torch.zeros(1)
        if model_outputs['displaced_silhouette_loss'] != None:
            displaced_silhouette_loss = model_outputs['displaced_silhouette_loss']
            displaced_laplacian_smoothing_loss = model_outputs['displaced_laplacian_smoothing_loss']
            displaced_edge_loss = model_outputs['displaced_edge_loss']
        if model_outputs['angle'] != None:
            # displaced_normal_consistency_loss = self.get_smooth_loss(model_outputs['displaced_adf_face_normals'])
            displaced_normal_consistency_loss = self.get_smooth_loss_v2(model_outputs['angle'])
        
        vh_mesh_loss_idr, _ = chamfer_distance(model_outputs['gt_vertices_tensor_idr'][None, :, :], 
                                               model_outputs['mesh_vertices_tensor'][None, :, :])
        # vh_mesh_loss_sc, _ = chamfer_distance(model_outputs['gt_vertices_tensor_sc'][None, :, :], 
        #                                    model_outputs['mesh_vertices_tensor'][None, :, :])
                                                
        if model_outputs['eikonal_points_g'] != None:
            eikonal_loss = self.get_eikonal_loss(model_outputs['eikonal_points_g'])

        second_bounce_ray_origins = model_outputs['second_bounce_ray_origins']  
        second_bounce_ray_directions = model_outputs['second_bounce_ray_directions'] 
        mean_len = model_outputs['mean_len'] 
        DRT_ray_loss = torch.zeros(1).cuda()
        if second_bounce_ray_origins != None and self.use_DRT_corr:
            corr = ground_truth['corr'].reshape(-1, 3)
            corr = corr.float().cuda()
            object_mask = model_outputs['erode_object_mask'].reshape(-1, num_pixel, 1)[:, :patch_num_pixels, :].reshape(-1)
            composite_rgb_mask = model_outputs['composite_rgb_mask'].reshape(-1, num_pixel, 1)[:, :patch_num_pixels, :].reshape(-1)
            valid = (corr[:,0] != 0)
            mask = valid & composite_rgb_mask & object_mask
            target = corr  - second_bounce_ray_origins.detach()
            target = target/target.norm(dim=1, keepdim=True)
            diff = (second_bounce_ray_directions - target)
            DRT_ray_loss = (diff[mask, :]).pow(2).mean()

        corr_loss = torch.zeros(1).cuda()
        outside_border_loss = torch.zeros(1).cuda()
        board_hit_points_uvs = model_outputs['board_hit_points_uvs']
        corr = ground_truth['corr']
        if board_hit_points_uvs is not None and self.use_our_corr:
            board_hit_points_uvs = board_hit_points_uvs.cuda()
            corr = corr.float().cuda()
            rgb_gt = rgb_gt.reshape(-1, num_pixel, 3)[:, :patch_num_pixels, :].reshape(-1, 3)
            highlight_mask = (rgb_gt[:, 0] > 220 / 255.0) & (rgb_gt[:, 1] > 220 / 255.0) & (rgb_gt[:, 2] > 220 / 255.0)
            # Here we use erode_object_mask!!!
            object_mask = model_outputs['erode_object_mask'].reshape(-1, num_pixel, 1)[:, :patch_num_pixels, :].reshape(-1)
            composite_rgb_mask = model_outputs['composite_rgb_mask'].reshape(-1, num_pixel, 1)[:, :patch_num_pixels, :].reshape(-1)
            mask = composite_rgb_mask & object_mask & (~highlight_mask)
            corr_loss = self.get_corr_loss(board_hit_points_uvs.reshape(-1, 2),
                                           corr.reshape(-1, 3).detach(),
                                           mask.detach())
            outside_border_loss = self.get_outside_border_loss(board_hit_points_uvs.reshape(-1, 2),
                                                               corr.reshape(-1, 3).detach(),
                                                               mask.detach())

        zero_displacement_loss =  torch.zeros(1)
        if model_outputs['displacement_values'] != None:
            zero_displacement_loss = self.get_zero_displacement_loss(model_outputs['displacement_values'])

        color_loss = torch.zeros(1)
        if model_outputs['rgb_values'] != None:

            object_mask = model_outputs['object_mask'].reshape(-1, num_pixel, 1)[:, :patch_num_pixels, :].reshape(-1)
            composite_rgb_mask = model_outputs['composite_rgb_mask'].reshape(-1, num_pixel, 1)[:, :patch_num_pixels, :].reshape(-1)
            rgb_values_mask = model_outputs['rgb_values_mask'].reshape(-1, num_pixel, 1)[:, :patch_num_pixels, :].reshape(-1)
            rgb_values = model_outputs['rgb_values']

            rgb_values = rgb_values.reshape(-1, num_pixel, 3)[:, :patch_num_pixels, :].reshape(-1, 3)
            rgb_gt = rgb_gt.reshape(-1, num_pixel, 3)[:, :patch_num_pixels, :].reshape(-1, 3)
            
            highlight_mask = (rgb_gt[:, 0] > 220 / 255.0) & (rgb_gt[:, 1] > 220 / 255.0) & (rgb_gt[:, 2] > 220 / 255.0)
            pixel_valid_mask = (rgb_values[:, 0] < 0.01) & (rgb_values[:, 1] < 0.01) & (rgb_values[:, 2] < 0.01)
            
            object_mask_wo_highlight = object_mask & (~highlight_mask) & (~pixel_valid_mask)
            composite_rgb_mask_wo_highlight = composite_rgb_mask & (~highlight_mask) & (~pixel_valid_mask)
            rgb_values[~object_mask_wo_highlight, :] = 0
            rgb_values[~composite_rgb_mask_wo_highlight, :] = 0
            rgb_gt[~object_mask_wo_highlight, :] = 0
            rgb_gt[~composite_rgb_mask_wo_highlight, :] = 0
            
            gauss_5x5 = kornia.filters.GaussianBlur2d((5, 5), (1.0, 1.0))
            gauss_9x9 = kornia.filters.GaussianBlur2d((9, 9), (2.0, 2.0))
            input_rgb_patch = rgb_values.reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)
            target_rgb_patch = rgb_gt.reshape(-1, patch_size, patch_size, 3).permute(0, 3, 1, 2)
            
            gauss_1x1_rgb_loss = self.get_rgb_loss(input_rgb_patch, target_rgb_patch, composite_rgb_mask_wo_highlight, object_mask_wo_highlight, rgb_values_mask) 
            
            gauss_5x5_input_rgb_patch = gauss_5x5(input_rgb_patch)
            gauss_5x5_target_rgb_patch = gauss_5x5(target_rgb_patch)
            gauss_5x5_rgb_loss = self.get_rgb_loss(gauss_5x5_input_rgb_patch, gauss_5x5_target_rgb_patch, composite_rgb_mask_wo_highlight, object_mask_wo_highlight, rgb_values_mask) 
            
            gauss_9x9_input_rgb_patch = gauss_9x9(input_rgb_patch)
            gauss_9x9_target_rgb_patch = gauss_9x9(target_rgb_patch)
            gauss_9x9_rgb_loss = self.get_rgb_loss(gauss_9x9_input_rgb_patch, gauss_9x9_target_rgb_patch, composite_rgb_mask_wo_highlight, object_mask_wo_highlight, rgb_values_mask) 
      
            pixel_loss = 0.4 * (gauss_1x1_rgb_loss + gauss_5x5_rgb_loss + gauss_9x9_rgb_loss)
            color_loss = 10.0 * pixel_loss #(pixel_loss + ssim_loss + vgg_loss)

        # print(f'self.corr_weight: {self.corr_weight}\n self.ncorr_weight: {self.outside_border_weight}\n self.color_weight: {self.color_weight}\n self.displaced_silhouette_weight: {self.displaced_silhouette_weight}\n self.displaced_normal_consistency_weight: {self.displaced_normal_consistency_weight * (mean_len * mean_len / 10.0)}\n self.displaced_laplacian_smoothing_weight: {self.displaced_laplacian_smoothing_weight}\n self.vh_mesh_weight: {self.vh_mesh_weight}\n')
        # input()
        
        eikonal_loss = self.eikonal_weight * eikonal_loss
        vh_mesh_loss = self.vh_mesh_weight * vh_mesh_loss_idr
        corr_loss = self.corr_weight * corr_loss
        DRT_ray_loss = self.corr_weight * DRT_ray_loss
        color_loss = self.color_weight * color_loss
        outside_border_loss = self.outside_border_weight * outside_border_loss 
        displaced_silhouette_loss = self.displaced_silhouette_weight * displaced_silhouette_loss
        # displaced_normal_consistency_loss = max(1, (3 - math.floor(epoch / 60))) * self.displaced_normal_consistency_weight * (mean_len * mean_len / 10.0) * displaced_normal_consistency_loss
        displaced_normal_consistency_loss = self.displaced_normal_consistency_weight * (mean_len * mean_len / 10.0) * displaced_normal_consistency_loss
        displaced_laplacian_smoothing_loss = self.displaced_laplacian_smoothing_weight * displaced_laplacian_smoothing_loss
        displaced_edge_loss = self.displaced_edge_weight * displaced_edge_loss
        
        if self.large_step:
            loss = corr_loss + \
                outside_border_loss + \
                color_loss + \
                displaced_silhouette_loss + \
                0.1 * vh_mesh_loss
        elif self.sdf_single_mlp:
            loss = corr_loss + \
                outside_border_loss + \
                displaced_silhouette_loss + \
                eikonal_loss
        else:
            loss = corr_loss + \
                outside_border_loss + \
                displaced_silhouette_loss + \
                displaced_laplacian_smoothing_loss + \
                displaced_normal_consistency_loss + \
                vh_mesh_loss + \
                color_loss
                
        return {
            'loss': loss,
            'color_loss': color_loss,
            'eikonal_loss': eikonal_loss,
            'zero_displacement_loss': zero_displacement_loss,
            'displaced_silhouette_loss': displaced_silhouette_loss,
            'displaced_normal_consistency_loss': displaced_normal_consistency_loss,
            'displaced_laplacian_smoothing_loss': displaced_laplacian_smoothing_loss,
            'displaced_edge_loss': displaced_edge_loss,
            'corr_loss': corr_loss,
            'outside_border_loss': outside_border_loss,
            'DRT_ray_loss': DRT_ray_loss,
            'vh_mesh_loss': vh_mesh_loss
        }
