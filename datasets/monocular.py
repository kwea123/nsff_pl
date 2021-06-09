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
            sorted(glob.glob(os.path.join(self.root_dir, 'disps/*')))[self.start_frame:self.end_frame]
        self.mask_paths = \
            sorted(glob.glob(os.path.join(self.root_dir, 'masks/*')))[self.start_frame:self.end_frame]
        self.flow_fw_paths = \
            sorted(glob.glob(os.path.join(self.root_dir, 'flow_fw/*.flo')))[self.start_frame:self.end_frame] + ['dummy']
        self.flow_bw_paths = \
            ['dummy'] + sorted(glob.glob(os.path.join(self.root_dir, 'flow_bw/*.flo')))[self.start_frame:self.end_frame]
        self.N_frames = len(self.image_paths)

        camdata = colmap_utils.read_cameras_binary(os.path.join(self.root_dir,
                                                                'sparse/0/cameras.bin'))
        H = camdata[1].height
        W = camdata[1].width
        f, cx, cy, _ = camdata[1].params

        self.K = np.array([[f, 0, cx],
                           [0, f, cy],
                           [0,  0, 1]], dtype=np.float32)
        self.K[0] *= self.img_wh[0]/W
        self.K[1] *= self.img_wh[1]/H

        # read extrinsics
        imdata = colmap_utils.read_images_binary(os.path.join(self.root_dir,
                                                              'sparse/0/images.bin'))
        perm = np.argsort([imdata[k].name for k in imdata])
        w2c_mats = []
        bottom = np.array([0, 0, 0, 1.]).reshape(1, 4)
        for k in imdata:
            im = imdata[k]
            R = im.qvec2rotmat()
            t = im.tvec.reshape(3, 1)
            w2c_mats += [np.concatenate([np.concatenate([R, t], 1), bottom], 0)]
        w2c_mats = np.stack(w2c_mats, 0)[perm]
        poses = np.linalg.inv(w2c_mats)[self.start_frame:self.end_frame, :3] # (N_images, 3, 4)

        # read bounds
        pts3d = colmap_utils.read_points3d_binary(os.path.join(self.root_dir,
                                                               'sparse/0/points3D.bin'))
        xyz_world = [pts3d[p_id].xyz for p_id in pts3d]
        xyz_world = np.concatenate([np.array(xyz_world), np.ones((len(xyz_world), 1))], -1)
        xyz_cam0 = (xyz_world @ w2c_mats[self.start_frame].T)[:, :3] # xyz in the first frame
        xyz_cam0 = xyz_cam0[xyz_cam0[:, 2]>0]
        self.nearest_depth = np.percentile(xyz_cam0[:, 2], 0.1)

        # Step 2: correct poses
        # change "right down front" of COLMAP to "right up back"
        self.poses = np.concatenate([poses[..., 0:1], -poses[..., 1:3], poses[..., 3:4]], -1)
        self.poses = colmap_utils.center_poses(self.poses)

        # Step 3: correct scale
        self.scale_factor = self.nearest_depth
        self.poses[..., 3] /= self.scale_factor

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
                        mask = self.transform(mask).flatten() # (h*w)
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
                        disp = cv2.imread(self.disp_paths[i], cv2.IMREAD_ANYDEPTH).astype(np.float32)
                        disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
                        disp = torch.FloatTensor(disp).reshape(-1, 1) # (h*w, 1)
                        rays += [disp]

                    if self.mask_paths:
                        rays_mask = mask.unsqueeze(-1) # 0:static, 1:dynamic
                        rays += [rays_mask]

                    if self.flow_fw_paths:
                        if i < self.N_frames-1:
                            flow_fw = flowlib.read_flow(self.flow_fw_paths[i])
                            flow_fw = flowlib.resize_flow(flow_fw, self.img_wh[0], self.img_wh[1])
                            flow_fw = torch.FloatTensor(flow_fw.reshape(-1, 2))
                        else:
                            flow_fw = torch.zeros(len(rays_o), 2)

                        if i >= 1:
                            flow_bw = flowlib.read_flow(self.flow_bw_paths[i])
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
                    disp = cv2.imread(self.disp_paths[id_], cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
                    sample['disp'] = torch.FloatTensor(disp.flatten())
                if self.mask_paths:
                    mask = Image.open(self.mask_paths[id_]).convert('L')
                    mask = mask.resize(self.img_wh, Image.NEAREST)
                    mask = self.transform(mask).flatten() # (h*w)
                    sample['mask'] = mask
                if self.flow_fw_paths:
                    if time < self.N_frames-1:
                        flow_fw = flowlib.read_flow(self.flow_fw_paths[id_])
                        flow_fw = flowlib.resize_flow(flow_fw, self.img_wh[0], self.img_wh[1])
                        sample['flow_fw'] = flow_fw
                    else:
                        sample['flow_fw'] = torch.zeros(self.img_wh[1], self.img_wh[0], 2)

                    if time >= 1:
                        flow_bw = flowlib.read_flow(self.flow_bw_paths[id_])
                        flow_bw = flowlib.resize_flow(flow_bw, self.img_wh[0], self.img_wh[1])
                        sample['flow_bw'] = flow_bw
                    else:
                        sample['flow_bw'] = torch.zeros(self.img_wh[1], self.img_wh[0], 2)

        return sample