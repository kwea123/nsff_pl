import cv2
import torch
from torch.utils.data import Dataset
import glob
import numpy as np
import os
from collections import defaultdict
from PIL import Image
from torchvision import transforms as T

from . import ray_utils, depth_utils, colmap_utils, flowlib


class MonocularDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(512, 288),
                 start_end=(0, 30), cache_dir=None):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.cam_train = [0]
        self.cam_test = 1
        self.start_frame = start_end[0]
        self.end_frame = start_end[1]
        self.cache_dir = cache_dir
        self.define_transforms()
        self.read_meta()

    def read_meta(self):
        # read inputs
        self.image_paths = \
            sorted(glob.glob(os.path.join(self.root_dir, 'images/*')))[self.start_frame:self.end_frame]
        self.disp_paths = \
            sorted(glob.glob(os.path.join(self.root_dir, 'disp/*.npy')))[self.start_frame:self.end_frame]
        self.mask_paths = \
            sorted(glob.glob(os.path.join(self.root_dir, 'motion_masks/*')))[self.start_frame:self.end_frame]
        self.flow_fw_paths = \
            sorted(glob.glob(os.path.join(self.root_dir, 'flow_i1/*_fwd.npz')))[self.start_frame:self.end_frame] + ['dummy']
        self.flow_bw_paths = \
            ['dummy'] + sorted(glob.glob(os.path.join(self.root_dir, 'flow_i1/*_bwd.npz')))[self.start_frame:self.end_frame]
        self.N_frames = len(self.image_paths)

        poses_bounds = np.load(os.path.join(self.root_dir, 'poses_bounds.npy'))
        poses = poses_bounds[self.start_frame:self.end_frame, :15].reshape(-1, 3, 5) # (N_frames, 3, 5)
        self.bounds = poses_bounds[self.start_frame:self.end_frame, -2:] # (N_frames, 2)

        H, W, f = poses[0, :, -1]
        self.K = np.array([[f, 0, W/2],
                           [0, f, H/2],
                           [0,  0,  1]], dtype=np.float32)
        self.K[0] *= self.img_wh[0]/W
        self.K[1] *= self.img_wh[1]/H

        poses = np.concatenate([poses[..., 1:2], -poses[..., :1], poses[..., 2:4]], -1) # (N_frames, 3, 4)
        self.poses = colmap_utils.center_poses(poses)

        near_original = self.bounds.min()
        scale_factor = np.percentile(self.bounds[:, 0], 5) * 0.9
        self.bounds /= scale_factor
        self.poses[..., 3] /= scale_factor

        # create projection matrix, used to compute optical flow
        bottom = np.zeros((self.N_frames, 1, 4))
        bottom[..., -1] = 1
        rt = np.linalg.inv(np.concatenate([self.poses, bottom], 1))[:, :3]
        rt[:, 1:] *= -1 # "right up back" to "right down forward" for cam projection
        self.Ps = self.K @ rt
        self.Ps = torch.FloatTensor(self.Ps).unsqueeze(0) # (1, N_frames, 3, 4)
        self.Ks = torch.FloatTensor(self.K).unsqueeze(0) # (1, 3, 3)

        # Step 4: create ray buffers
        if self.split == 'train':
            # same ray directions for all cameras
            directions, uv = ray_utils.get_ray_directions(
                                self.img_wh[1], self.img_wh[0], self.K, return_uv=True)
            if self.cache_dir:
                self.rays_dict = torch.load(os.path.join(self.cache_dir, 'rays_dict.pt'))
                self.rgbs_dict = torch.load(os.path.join(self.cache_dir, 'rgbs_dict.pt'))
            else:
                self.rays_dict = {'static': {}, 'dynamic': {}}
                self.rgbs_dict = {'static': {}, 'dynamic': {}}
                for i, image_path in enumerate(self.image_paths):
                    c2w = torch.FloatTensor(self.poses[i])

                    img = Image.open(image_path).convert('RGB').resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img).view(3, -1).T # (h*w, 3) RGB
                    if self.mask_paths:
                        mask = Image.open(self.mask_paths[i]).convert('L')
                        mask = mask.resize(self.img_wh, Image.NEAREST)
                        mask = 1-self.transform(mask).flatten() # (h*w)
                        self.rgbs_dict['static'][i] = img[mask>0]
                        self.rgbs_dict['dynamic'][i] = img[mask==0]
                    else:
                        self.rgbs_dict['static'][i] = img

                    rays_o, rays_d = ray_utils.get_rays(directions, c2w) # both (h*w, 3)
                    shift_near = -min(-1.0, self.poses[i, 2, 3])
                    rays_o, rays_d = ray_utils.get_ndc_rays(self.K, 1.0, 
                                                            shift_near, rays_o, rays_d)

                    rays_t = i * torch.ones(len(rays_o), 1) # (h*w, 1)
                    rays = [rays_o, rays_d, rays_t]

                    if self.disp_paths:
                        disp = np.load(self.disp_paths[i])
                        disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
                        disp = torch.FloatTensor(disp).reshape(-1, 1) # (h*w, 1)
                        rays += [disp]

                    if self.mask_paths:
                        rays_mask = mask.unsqueeze(-1) # 0:static, 1:dynamic
                        rays += [rays_mask]

                    if self.flow_fw_paths:
                        if i < self.N_frames-1:
                            flow_fw = np.load(self.flow_fw_paths[i])['flow']
                            flow_fw = flowlib.resize_flow(flow_fw, self.img_wh[0], self.img_wh[1])
                            flow_fw = torch.FloatTensor(flow_fw.reshape(-1, 2))
                        else:
                            flow_fw = torch.zeros(len(rays_o), 2)

                        if i >= 1:
                            flow_bw = np.load(self.flow_bw_paths[i])['flow']
                            flow_bw = flowlib.resize_flow(flow_bw, self.img_wh[0], self.img_wh[1])
                            flow_bw = torch.FloatTensor(flow_bw.reshape(-1, 2))
                        else:
                            flow_bw = torch.zeros(len(rays_o), 2)
                        rays += [uv+flow_fw, uv+flow_bw]

                    rays = torch.cat(rays, 1) # (h*w, at most 3+3+1+1+1+2+2=13)

                    if self.mask_paths:
                        self.rays_dict['static'][i] = rays[mask>0]
                        self.rays_dict['dynamic'][i] = rays[mask==0]
                    else:
                        self.rays_dict['static'][i] = rays

            # combine all static and dynamic data
            self.rays_static = torch.cat([self.rays_dict['static'][t]
                                          for t in range(self.N_frames)], 0)
            self.rgbs_static = torch.cat([self.rgbs_dict['static'][t]
                                          for t in range(self.N_frames)], 0)
            self.N_rays_static = len(self.rays_static)

            if self.mask_paths:
                self.rays_dynamic = torch.cat([self.rays_dict['dynamic'][t]
                                               for t in range(self.N_frames)], 0)
                self.rgbs_dynamic = torch.cat([self.rgbs_dict['dynamic'][t]
                                               for t in range(self.N_frames)], 0)
                self.N_rays_dynamic = len(self.rays_dynamic)

        elif self.split == 'val':
            self.val_id = self.N_frames//2
            print('val image is', self.image_paths[self.val_id])

        else:
            self.poses_test = self.poses.copy()
            self.image_paths_test = self.image_paths

    def define_transforms(self):
        self.transform = T.ToTensor()

    def get_data(self, output_transient): # called in train.py
        if self.mask_paths and output_transient:
            rays = torch.cat([self.rays_static, self.rays_dynamic])
            self.rgbs = torch.cat([self.rgbs_static, self.rgbs_dynamic])
        else:
            rays = self.rays_static
            self.rgbs = self.rgbs_static
        self.rays = rays[..., :6] # (N, 6)
        self.ts = rays[..., 6].long() # (N)
        self.cam_ids = torch.zeros_like(self.ts) # (N)
        if self.disp_paths:
            self.disps = rays[:, 7]
        if self.mask_paths:
            self.rays_mask = rays[..., 8] # (N)
        if self.output_transient_flow:
            self.uv_fw = rays[..., 9:11] # (N, 2) uv + forward flow
            self.uv_bw = rays[..., 11:13] # (N, 2) uv + backward flow

    def __len__(self):
        if self.split == 'train':
            if self.batch_from_same_image: return 5000
            else: return len(self.rays)
        if self.split == 'val': return 1
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train': # use data in the buffers
            if self.batch_from_same_image: # random rays from the same image
                t = np.random.choice(self.N_frames)
                if self.mask_paths:
                    rand_idx = np.random.choice(len(self.rays_dict['static'][t])+
                                                len(self.rays_dict['dynamic'][t]),
                                                self.batch_size)
                    rays = torch.cat([self.rays_dict['static'][t],
                                      self.rays_dict['dynamic'][t]], 0)[rand_idx]
                    rgbs = torch.cat([self.rgbs_dict['static'][t],
                                      self.rgbs_dict['dynamic'][t]], 0)[rand_idx]
                else:
                    rand_idx = np.random.choice(self.img_wh[0]*self.img_wh[1], self.batch_size)
                    rays = self.rays_dict['static'][t][rand_idx]
                    rgbs = self.rgbs_dict['static'][t][rand_idx]
                sample = {'rays': rays[..., :6],
                          'rgbs': rgbs,
                          'ts': rays[..., 6].long(),
                          'cam_ids': 0*rays[..., 6].long()}
                if self.disp_paths:
                    sample['disps'] = rays[..., 7]
                if self.mask_paths:
                    sample['rays_mask'] = rays[..., 8]
                if self.output_transient_flow:
                    sample['uv_fw'] = rays[..., 9:11]
                    sample['uv_bw'] = rays[..., 11:13]
            else: # random rays from all images
                sample = {'rays': self.rays[idx],
                          'rgbs': self.rgbs[idx],
                          'ts': self.ts[idx],
                          'cam_ids': self.cam_ids[idx]}
                if self.disp_paths:
                    sample['disps'] = self.disps[idx]
                if self.mask_paths:
                    sample['rays_mask'] = self.rays_mask[idx]
                if self.output_transient_flow:
                    sample['uv_fw'] = self.uv_fw[idx]
                    sample['uv_bw'] = self.uv_bw[idx]
        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.poses[self.val_id])
                time = self.val_id % self.N_frames
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])
                if self.split == 'test': time = idx
                else: time = 0

            directions = ray_utils.get_ray_directions(self.img_wh[1], self.img_wh[0], self.K)
            rays_o, rays_d = ray_utils.get_rays(directions, c2w)
            shift_near = -min(-1.0, c2w[2, 3])
            rays_o, rays_d = ray_utils.get_ndc_rays(self.K, 1.0, 
                                                    shift_near, rays_o, rays_d)

            rays_t = time * torch.ones(len(rays_o), dtype=torch.long) # (h*w)

            rays = torch.cat([rays_o, rays_d], 1) # (h*w, 6)

            sample = {'rays': rays, 'ts': rays_t, 'c2w': c2w}

            if self.split in ['val', 'test']:
                sample['cam_ids'] = 0
                id_ = self.val_id if self.split == 'val' else idx
                img = Image.open(self.image_paths[id_]).convert('RGB')
                img = img.resize(self.img_wh, Image.LANCZOS)
                img = self.transform(img) # (3, h, w)
                img = img.view(3, -1).T # (h*w, 3)
                sample['rgbs'] = img
                if self.disp_paths:
                    disp = np.load(self.disp_paths[id_])
                    disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
                    sample['disp'] = torch.FloatTensor(disp.flatten())
                if self.mask_paths:
                    mask = Image.open(self.mask_paths[id_]).convert('L')
                    mask = mask.resize(self.img_wh, Image.NEAREST)
                    mask = 1-self.transform(mask).flatten() # (h*w)
                    sample['mask'] = mask
                if self.flow_fw_paths:
                    if time < self.N_frames-1:
                        flow_fw = np.load(self.flow_fw_paths[id_])['flow']
                        flow_fw = flowlib.resize_flow(flow_fw, self.img_wh[0], self.img_wh[1])
                        sample['flow_fw'] = flow_fw
                    else:
                        sample['flow_fw'] = torch.zeros(self.img_wh[1], self.img_wh[0], 2)

                    if time >= 1:
                        flow_bw = np.load(self.flow_bw_paths[id_])['flow']
                        flow_bw = flowlib.resize_flow(flow_bw, self.img_wh[0], self.img_wh[1])
                        sample['flow_bw'] = flow_bw
                    else:
                        sample['flow_bw'] = torch.zeros(self.img_wh[1], self.img_wh[0], 2)

        return sample