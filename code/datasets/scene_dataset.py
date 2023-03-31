import os
import torch
import numpy as np

import utils.general as utils
from utils import rend_util
from utils.general import sync_print
import cv2
import tqdm
import math

class SceneDataset(torch.utils.data.Dataset):
    """Dataset for a class of objects, where each datapoint is a SceneInstanceDataset."""

    def __init__(self,
                 use_our_corr,
                 use_DRT_corr,
                 use_sdf,
                 data_dir,
                 img_res,
                 mean_rgb,
                 std_rgb,
                 stride
                 ):

        self.instance_dir = data_dir
        self.total_pixels = img_res[0] * img_res[1]
        self.img_res = img_res
        assert os.path.exists(self.instance_dir), "Data directory is empty"
        self.sampling_idx = None
        self.use_our_corr = use_our_corr
        self.use_DRT_corr = use_DRT_corr
        self.use_sdf = use_sdf

        image_dir = '{}/image/'.format(self.instance_dir)
        image_paths = sorted(utils.glob_imgs(image_dir))
        # mask_dir = '{}/project_mask/'.format(self.instance_dir)
        mask_dir = '{}/mask/'.format(self.instance_dir)
        mask_paths = sorted(utils.glob_imgs(mask_dir))
        corr_dir = '{}/corr/'.format(self.instance_dir)
        corr_paths = sorted(utils.glob_imgs(corr_dir)) 

        manual_anno_dir = '{}/mask_loss/mask/'.format(self.instance_dir)
        # manual_anno_dir = '{}/project_mask/'.format(self.instance_dir)
        manual_anno_image_paths = sorted(utils.glob_imgs(manual_anno_dir))
        
        self.n_images = len(image_paths)
        mask_paths = mask_paths[:self.n_images]

        self.cam_file = '{}/cameras.npz'.format(self.instance_dir)

        camera_dict = np.load(self.cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        self.all_intrinsics = []
        self.all_poses = []
        self.all_Ps = []
        for scale_mat, world_mat in tqdm.tqdm(zip(scale_mats, world_mats)):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            self.all_Ps.append(P)
            intrinsics, pose = rend_util.load_K_Rt_from_P(None, P)
            self.all_intrinsics.append(torch.from_numpy(intrinsics).float())
            self.all_poses.append(torch.from_numpy(pose).float())
        image_paths = image_paths[::stride]
        mask_paths = mask_paths[::stride]
        corr_paths = corr_paths[::stride]
        manual_anno_image_paths = manual_anno_image_paths[::stride]
        self.all_intrinsics = self.all_intrinsics[::stride]
        self.all_poses = self.all_poses[::stride]

        self.mask_scale = 2
        
        self.all_rgb_images = []
        self.all_object_masks = []
        self.all_erode_object_masks = []
        self.all_hard_mining_masks = []
        self.all_scaled_object_masks = []
        self.all_corr_images = []
        self.all_board_indices = []
        self.all_manual_anno_indices = []
        for path in tqdm.tqdm(image_paths):
            # rgb = rend_util.load_rgb(path, mean_rgb, std_rgb).reshape(-1, 3)
            rgb = rend_util.load_rgb_uint8(path).reshape(-1, 3)
            object_mask = torch.ones(rgb.shape[0]).bool()
            half_object_mask = torch.ones(rgb.shape[0] // self.mask_scale // self.mask_scale).bool()
            corr_image = torch.zeros(rgb.shape[0], 3) / 0.0

            # self.all_rgb_images.append(torch.from_numpy(rgb).float())
            self.all_rgb_images.append(torch.from_numpy(rgb))
            self.all_object_masks.append(object_mask)
            self.all_erode_object_masks.append(object_mask)
            self.all_hard_mining_masks.append(object_mask)
            self.all_scaled_object_masks.append(half_object_mask)
            self.all_corr_images.append(corr_image)
            self.all_board_indices.append(torch.IntTensor([-1]))
            self.all_manual_anno_indices.append(torch.IntTensor([-1]))

        if use_our_corr:
 
            image_board_mapping = np.array([])
            if os.path.exists(corr_dir + 'image_board_mapping.npy'):
                image_board_mapping = np.load(corr_dir + 'image_board_mapping.npy')
            for i in range(image_board_mapping.shape[0]):
                image_idx = int(image_board_mapping[i, 0])
                board_idx = int(image_board_mapping[i, 1])
                if image_idx < len(self.all_board_indices):
                    self.all_board_indices[image_idx] = torch.IntTensor([board_idx]) 

            for path in tqdm.tqdm(corr_paths):
                image_idx, _ = path.split('/')[-1].split('.')
                image_idx = int(image_idx)
                if image_idx < len(self.all_corr_images):
                    corr = rend_util.load_corr(path)
                    corr = corr.reshape(-1, 3)
                    corr_tensor = torch.tensor(corr, dtype=torch.float16) 
                    # corr_tensor = corr_tensor.type('torch.float16') 
                    self.all_corr_images[image_idx] = corr_tensor
                    # self.all_corr_images[image_idx] = torch.from_numpy(corr).float()
                    
        if use_DRT_corr:
            image_idx = -1
            for path in tqdm.tqdm(corr_paths):
                if not use_DRT_corr:
                    image_idx, _ = path.split('/')[-1].split('.')
                    image_idx = int(image_idx)
                else:
                    image_idx = image_idx + 1
                if image_idx < len(self.all_corr_images):
                    corr = rend_util.load_corr(path)
                    corr = corr.reshape(-1, 3)
                    corr_tensor = torch.tensor(corr, dtype=torch.float32) 
                    self.all_corr_images[image_idx] = corr_tensor

        kernel_7x7 = np.ones((7, 7), np.uint8) 
        kernel_21x21 = np.ones((21, 21), np.uint8) 
        # kernel = np.ones((79, 79), np.uint8) # for hand
        image_idx = -1
        for path in tqdm.tqdm(mask_paths):
            object_mask = rend_util.load_mask(path) 
            # Using cv2.erode() method 
            erode_object_mask = cv2.erode(object_mask.astype('uint8'), kernel_7x7) 
            hard_mining_mask = object_mask - cv2.erode(object_mask.astype('uint8'), kernel_21x21) 
            scaled_object_mask = cv2.resize(object_mask.astype('float32'), 
                                            (object_mask.shape[1] // self.mask_scale, object_mask.shape[0] // self.mask_scale), 
                                            interpolation=cv2.INTER_LINEAR)
            hard_mining_mask = cv2.resize(hard_mining_mask, 
                                          (hard_mining_mask.shape[1] // self.mask_scale, hard_mining_mask.shape[0] // self.mask_scale), 
                                          interpolation=cv2.INTER_NEAREST)
            object_mask = object_mask.reshape(-1)
            erode_object_mask = erode_object_mask.reshape(-1)
            scaled_object_mask = scaled_object_mask.reshape(-1)
            if not use_DRT_corr:
                image_idx, _ = path.split('/')[-1].split('.')
                image_idx = int(image_idx)
            else:
                image_idx = image_idx + 1
            if image_idx < len(self.all_object_masks):
                self.all_object_masks[image_idx] = torch.from_numpy(object_mask).bool()
                self.all_erode_object_masks[image_idx] = torch.from_numpy(erode_object_mask).bool()
                self.all_hard_mining_masks[image_idx] = torch.from_numpy(hard_mining_mask).bool()
                self.all_scaled_object_masks[image_idx] = torch.from_numpy(scaled_object_mask).bool()

        self.silhouette_loss_object_masks = []
        self.silhouette_loss_intrinsics = []
        self.silhouette_loss_poses = []
        self.hard_mining_masks = []
        image_idx = -1
        for path in tqdm.tqdm(manual_anno_image_paths):
            if not use_DRT_corr:
                image_idx, _ = path.split('/')[-1].split('.')
                image_idx = int(image_idx)
            else:
                image_idx = image_idx + 1
            if image_idx < len(self.all_manual_anno_indices):
                self.all_manual_anno_indices[image_idx] = torch.IntTensor([image_idx]) 
                self.silhouette_loss_object_masks.append(self.all_scaled_object_masks[image_idx])
                self.silhouette_loss_intrinsics.append(self.all_intrinsics[image_idx])
                self.silhouette_loss_poses.append(self.all_poses[image_idx])
                self.hard_mining_masks.append(self.all_hard_mining_masks[image_idx])

        self.intrinsics = []
        self.poses = []
        self.rgb_images = []
        self.object_masks = []
        self.erode_object_masks = []
        # self.scaled_object_masks = []
        self.corr_images = []
        self.board_indices = []
        self.manual_anno_indices = []
        self.Ps = []
        image_idx = -1
        for path in tqdm.tqdm(mask_paths):

            if not use_DRT_corr:
                image_idx, _ = path.split('/')[-1].split('.')
                image_idx = int(image_idx)
            else:
                image_idx = image_idx + 1

            # print("mask_image_idx: {0}".format(image_idx))

            self.rgb_images.append(self.all_rgb_images[image_idx])
            self.corr_images.append(self.all_corr_images[image_idx])
            self.board_indices.append(self.all_board_indices[image_idx])
            self.manual_anno_indices.append(self.all_manual_anno_indices[image_idx])
            self.intrinsics.append(self.all_intrinsics[image_idx])
            self.poses.append(self.all_poses[image_idx])
            self.Ps.append(self.all_Ps[image_idx])
            self.object_masks.append(self.all_object_masks[image_idx])
            self.erode_object_masks.append(self.all_erode_object_masks[image_idx])
            # self.scaled_object_masks.append(self.all_scaled_object_masks[image_idx])
        self.n_images = len(self.rgb_images)
        
        uv = np.mgrid[0:self.img_res[0], 0:self.img_res[1]].astype(np.int32)
        uv = torch.from_numpy(np.flip(uv, axis=0).copy()).float()
        self.uv = uv.reshape(2, -1).transpose(1, 0)

        scaled_uv = np.mgrid[0:self.img_res[0] // self.mask_scale, 0:self.img_res[1] // self.mask_scale].astype(np.int32)
        scaled_uv = torch.from_numpy(np.flip(scaled_uv, axis=0).copy()).float()
        self.scaled_uv = scaled_uv.reshape(2, -1).transpose(1, 0)

        self.sampling_size = -1
        self.ignore_list = np.array([4, 5, 8, 13, 20, 36, 40, 44, 48, 53, 56, 71, 79, 83, 85, 86, 87, 92, 96, 99, 101, 103, 104, 105, 107, 112, 116, 118, 119, 120, 124, 128])

    def __len__(self):
        return self.n_images

    def __getitem__(self, idx):
        
        patch_size = 10
        if self.sampling_size > 0:
            patch_size = int(math.sqrt(self.sampling_size))
        if self.sampling_size == -1:
            self.sampling_idx = None
            self.silhouette_loss_sampling_idx = torch.randperm(100)[:100]
        else:
            patch_rad = patch_size // 2
            object_mask = self.object_masks[idx]
            nonzero_idx = torch.nonzero(object_mask).squeeze()
            if nonzero_idx.shape[0] == 0:
                self.sampling_idx = None
                self.silhouette_loss_sampling_idx = None
            else:
                height = self.img_res[0]
                width = self.img_res[1]
                rand_val = torch.rand(1)
                center = nonzero_idx[(rand_val * nonzero_idx.shape[0]).long().item()]
                sample_mask = torch.zeros(height, width).bool()
                center_row = max(patch_rad, min(center // width, height - patch_rad -1))
                center_col = max(patch_rad, min(center % width, width - patch_rad - 1))
                # sync_print("center: {0}, {1}, {2}".format(center, center_row, center_col))
                sample_mask[center_row - patch_rad : center_row + patch_rad, \
                    center_col - patch_rad : center_col + patch_rad] = True
                nonzero_idx = torch.nonzero(sample_mask)
                nonzero_idx = nonzero_idx[:, 0] * width + nonzero_idx[:, 1] 
                self.sampling_idx = nonzero_idx

                height = self.img_res[0] // self.mask_scale
                width = self.img_res[1] // self.mask_scale
                scaled_object_mask = self.silhouette_loss_object_masks[idx % len(self.silhouette_loss_object_masks)]
                scaled_nonzero_idx = torch.nonzero(scaled_object_mask).squeeze()
                center = scaled_nonzero_idx[(rand_val * scaled_nonzero_idx.shape[0]).long().item()]
                sample_mask = torch.zeros(height, width).bool()
                center_row = max(patch_rad, min(center // width, height - patch_rad -1))
                center_col = max(patch_rad, min(center % width, width - patch_rad - 1))
                sample_mask[center_row - patch_rad : center_row + patch_rad, \
                    center_col - patch_rad : center_col + patch_rad] = True
                scaled_nonzero_idx = torch.nonzero(sample_mask)
                scaled_nonzero_idx = scaled_nonzero_idx[:, 0] * width + scaled_nonzero_idx[:, 1] 
                self.silhouette_loss_sampling_idx = scaled_nonzero_idx
                
                self.random_sampling_idx = scaled_nonzero_idx#torch.randperm(width * height)[:self.sampling_size]

        sample = {
            "intrinsics": self.intrinsics[idx],
            "pose": self.poses[idx],
            "silhouette_loss_object_mask": self.silhouette_loss_object_masks[idx % len(self.silhouette_loss_object_masks)],
            "hard_mining_mask": self.hard_mining_masks[idx % len(self.silhouette_loss_object_masks)],
            "silhouette_loss_intrinsics": self.silhouette_loss_intrinsics[idx % len(self.silhouette_loss_object_masks)],
            "silhouette_loss_pose": self.silhouette_loss_poses[idx % len(self.silhouette_loss_object_masks)],
            "mask_scale": torch.IntTensor([self.mask_scale]),
            "patch_size": torch.IntTensor([patch_size]),
            "board_indices": self.board_indices[idx],
            "manual_anno_indices": self.manual_anno_indices[idx],
            "silhouette_loss_sampling_idx": self.silhouette_loss_sampling_idx  
        }
        ground_truth = {
            
        }

        if self.sampling_size != -1:
            ground_truth["rgb"] = self.rgb_images[idx][self.sampling_idx, :]
            ground_truth["corr"] = self.corr_images[idx][self.sampling_idx]
            sample["object_mask"] = self.object_masks[idx][self.sampling_idx]
            sample["erode_object_mask"] = self.erode_object_masks[idx][self.sampling_idx]
            sample["uv"] = self.uv[self.sampling_idx, :]
            # sample["idr_mask_loss_uv"] = self.scaled_uv[self.silhouette_loss_sampling_idx, :]
            # sample["idr_mask_loss_object_mask"] = sample["silhouette_loss_object_mask"][self.silhouette_loss_sampling_idx]
            sample["idr_mask_loss_uv"] = self.scaled_uv[self.random_sampling_idx, :]
            sample["idr_mask_loss_object_mask"] = sample["silhouette_loss_object_mask"][self.random_sampling_idx]
    
        else:
            ground_truth["rgb"] = self.rgb_images[idx]
            ground_truth["corr"] = self.corr_images[idx]
            sample["object_mask"] = self.object_masks[idx]
            sample["erode_object_mask"] = self.erode_object_masks[idx]
            sample["uv"] = self.uv
            sample["idr_mask_loss_uv"] = self.scaled_uv
            sample["idr_mask_loss_object_mask"] = sample["silhouette_loss_object_mask"]

        return idx, sample, ground_truth

    def collate_fn(self, batch_list):

        # sync_print('batch_list len: {0}'.format(len(batch_list)))
        # get list of dictionaries and returns input, ground_true as dictionary for all batch instances
        batch_list = zip(*batch_list)

        all_parsed = []
        for entry in batch_list:
            if type(entry[0]) is dict:
                # make them all into a new dict
                ret = {}
                for k in entry[0].keys():
                    ret[k] = torch.stack([obj[k] for obj in entry])
                all_parsed.append(ret)
            else:
                all_parsed.append(torch.LongTensor(entry))

        return tuple(all_parsed)

    def get_scale_mat(self):
        return np.load(self.cam_file)['scale_mat_0']

    def get_gt_pose(self, scaled=False):
        # Load gt pose without normalization to unit sphere
        camera_dict = np.load(self.cam_file)
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        pose_all = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat
            if scaled:
                P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            pose_all.append(torch.from_numpy(pose).float())

        return torch.cat([p.float().unsqueeze(0) for p in pose_all], 0)

    def get_pose_init(self):
        # get noisy initializations obtained with the linear method
        cam_file = '{0}/cameras_linear_init.npz'.format(self.instance_dir)
        camera_dict = np.load(cam_file)
        scale_mats = [camera_dict['scale_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]
        world_mats = [camera_dict['world_mat_%d' % idx].astype(np.float32) for idx in range(self.n_images)]

        init_pose = []
        for scale_mat, world_mat in zip(scale_mats, world_mats):
            P = world_mat @ scale_mat
            P = P[:3, :4]
            _, pose = rend_util.load_K_Rt_from_P(None, P)
            init_pose.append(pose)
        init_pose = torch.cat([torch.Tensor(pose).float().unsqueeze(0) for pose in init_pose], 0).cuda()
        init_quat = rend_util.rot_to_quat(init_pose[:, :3, :3])
        init_quat = torch.cat([init_quat, init_pose[:, :3, 3]], 1)

        return init_quat

