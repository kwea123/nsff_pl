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
from buffer.replay_buffer import PrioritizedReplayBuffer


class MonocularDataset(Dataset):
    def __init__(self, root_dir, split='train', img_wh=(512, 288),
                 start_end=(0, 30), cache_dir=None, prioritized_replay=False):
        self.root_dir = root_dir
        self.split = split
        self.img_wh = img_wh
        self.cam_train = [0]
        self.cam_test = 1
        self.start_frame = start_end[0]
        self.end_frame = start_end[1]
        self.cache_dir = cache_dir
        self.prioritized_replay = prioritized_replay
        if self.prioritized_replay:
            self.beta = 1.0 # maybe vary this according to steps
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
        self.nearest_depth = np.percentile(xyz_cam0[:, 2], 1) * 0.75

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
            directions, uv = ray_utils.get_ray_directions(
                                self.img_wh[1], self.img_wh[0], self.K, return_uv=True)
            if self.cache_dir:
                self.rays_dict = torch.load(os.path.join(self.cache_dir, 'rays_dict.pt'))
            else:
                self.rays_dict = {'static': {}, 'dynamic': {}}
                for t in range(self.N_frames):
                    img = Image.open(self.image_paths[t]).convert('RGB')
                    img = img.resize(self.img_wh, Image.LANCZOS)
                    img = self.transform(img).view(3, -1).T # (h*w, 3) RGB

                    c2w = torch.FloatTensor(self.poses[t])
                    rays_o, rays_d = ray_utils.get_rays(directions, c2w) # both (h*w, 3)
                    shift_near = -min(-1.0, self.poses[t, 2, 3])
                    rays_o, rays_d = ray_utils.get_ndc_rays(self.K, 1.0, 
                                                            shift_near, rays_o, rays_d)

                    rays_t = t * torch.ones(len(rays_o), 1) # (h*w, 1)

                    disp = cv2.imread(self.disp_paths[t], cv2.IMREAD_ANYDEPTH).astype(np.float32)
                    disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
                    disp = torch.FloatTensor(disp).reshape(-1, 1) # (h*w, 1)

                    mask = Image.open(self.mask_paths[t]).convert('L')
                    mask = mask.resize(self.img_wh, Image.NEAREST)
                    mask = self.transform(mask).flatten() # (h*w)
                    rays_mask = mask.unsqueeze(-1) # 0:static, 1:dynamic

                    if t < self.N_frames-1:
                        flow_fw = flowlib.read_flow(self.flow_fw_paths[t])
                        flow_fw = flowlib.resize_flow(flow_fw, self.img_wh[0], self.img_wh[1])
                        flow_fw = torch.FloatTensor(flow_fw.reshape(-1, 2))
                    else:
                        flow_fw = torch.zeros(len(rays_o), 2)

                    if t >= 1:
                        flow_bw = flowlib.read_flow(self.flow_bw_paths[t])
                        flow_bw = flowlib.resize_flow(flow_bw, self.img_wh[0], self.img_wh[1])
                        flow_bw = torch.FloatTensor(flow_bw.reshape(-1, 2))
                    else:
                        flow_bw = torch.zeros(len(rays_o), 2)

                    rays = [rays_o, rays_d, img, rays_t, disp, rays_mask, uv+flow_fw, uv+flow_bw]
                    rays = torch.cat(rays, 1) # (h*w, 3+3+3+1+1+1+2+2=16)

                    self.rays_dict['static'][t] = rays[mask>0]
                    self.rays_dict['dynamic'][t] = rays[mask==0]

            if self.prioritized_replay:
                self.replay_buffers = \
                    [PrioritizedReplayBuffer(self.img_wh[0]*self.img_wh[1], alpha=0.8)
                     for _ in range(self.N_frames)]
                # add all pixels with same priority
                for t in range(self.N_frames):
                    for ray in self.rays_dict['static']:
                        self.replay_buffers[t].add(ray)
                    for ray in self.rays_dict['dynamic']:
                        self.replay_buffers[t].add(ray)

        elif self.split == 'val':
            self.val_id = self.N_frames//2
            print('val image is', self.image_paths[self.val_id])

        elif self.split == 'test':
            self.poses_test = self.poses.copy()
            self.image_paths_test = self.image_paths

        elif self.split.startswith('test_fixview'):
            # fix to target view and change time
            target_idx = int(self.split.split('_')[-1])
            self.poses_test = np.tile(self.poses[target_idx], (self.N_frames, 1, 1))

        elif self.split.startswith('test_spiral'):
            if self.split == 'test_spiral': # spiral on the whole sequence
                max_trans = np.percentile(np.abs(np.diff(self.poses[:, 0, 3])), 10)
                radii = np.array([max_trans, max_trans, 0])
                self.poses_test = colmap_utils.create_spiral_poses(
                                    self.poses, radii, n_poses=6*self.N_frames)
            else: # spiral on the target idx
                target_idx = int(self.split.split('_')[-1])
                max_trans = np.abs(self.poses[0, 0, 3]-self.poses[-1, 0, 3])/5
                self.poses_test = colmap_utils.create_wander_path(
                                    self.poses[target_idx], max_trans=max_trans, n_poses=60)

    def define_transforms(self):
        self.transform = T.ToTensor()

    def __len__(self):
        if self.split == 'train': return 5000
        if self.split == 'val': return 1
        return len(self.poses_test)

    def __getitem__(self, idx):
        if self.split == 'train':
            t = np.random.choice(self.N_frames)
            if self.prioritized_replay:
                rays, weights, batch_idxs = \
                    self.replay_buffer[t].sample(self.batch_size, beta=self.beta)
            else:
                # st_samples = int(self.batch_size*0.2)
                # st_rand_idx = np.random.choice(len(self.rays_dict['static'][t]),
                #                                st_samples)
                # dy_rand_idx = np.random.choice(len(self.rays_dict['dynamic'][t]),
                #                                self.batch_size-st_samples)
                # rays = torch.cat([self.rays_dict['static'][t][st_rand_idx],
                #                   self.rays_dict['dynamic'][t][dy_rand_idx]], 0)
                rand_idx = np.random.choice(len(self.rays_dict['static'][t])+
                                            len(self.rays_dict['dynamic'][t]),
                                            self.batch_size)
                rays = torch.cat([self.rays_dict['static'][t],
                                  self.rays_dict['dynamic'][t]], 0)[rand_idx]
            sample = {'rays': rays[:, :6],
                      'rgbs': rays[:, 6:9],
                      'ts': rays[:, 9].long(),
                      'cam_ids': 0*rays[:, 9].long(),
                      'disps': rays[:, 10],
                      'rays_mask': rays[:, 11],
                      'uv_fw': rays[:, 12:14],
                      'uv_bw': rays[:, 14:16]}
            if self.prioritized_replay:
                sample['weights'] = weights # FloatTensor
                sample['batch_idxs'] = batch_idxs # list
        else:
            if self.split == 'val':
                c2w = torch.FloatTensor(self.poses[self.val_id])
                time = self.val_id % self.N_frames
            else:
                c2w = torch.FloatTensor(self.poses_test[idx])
                if self.split == 'test':
                    time = idx
                elif self.split.startswith('test_spiral'):
                    if self.split == 'test_spiral': 
                        time = int(idx/len(self.poses_test)*self.N_frames)
                    else:
                        time = int(self.split.split('_')[-1])
                elif self.split.startswith('test_fixview'):
                    time = idx
                else: time = 0

            directions = ray_utils.get_ray_directions(self.img_wh[1], self.img_wh[0], self.K)
            rays_o, rays_d = ray_utils.get_rays(directions, c2w)
            shift_near = -min(-1.0, c2w[2, 3])
            rays_o, rays_d = ray_utils.get_ndc_rays(self.K, 1.0, 
                                                    shift_near, rays_o, rays_d)

            rays_t = time * torch.ones(len(rays_o), dtype=torch.long) # (h*w)

            rays = torch.cat([rays_o, rays_d], 1) # (h*w, 6)

            sample = {'rays': rays, 'ts': rays_t, 'c2w': c2w}

            sample['cam_ids'] = 0
            id_ = self.val_id if self.split == 'val' else time
            img = Image.open(self.image_paths[id_]).convert('RGB')
            img = img.resize(self.img_wh, Image.LANCZOS)
            img = self.transform(img) # (3, h, w)
            img = img.view(3, -1).T # (h*w, 3)
            sample['rgbs'] = img

            disp = cv2.imread(self.disp_paths[id_], cv2.IMREAD_ANYDEPTH).astype(np.float32)
            disp = cv2.resize(disp, self.img_wh, interpolation=cv2.INTER_NEAREST)
            sample['disp'] = torch.FloatTensor(disp.flatten())

            mask = Image.open(self.mask_paths[id_]).convert('L')
            mask = mask.resize(self.img_wh, Image.NEAREST)
            mask = self.transform(mask).flatten() # (h*w)
            sample['mask'] = mask

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