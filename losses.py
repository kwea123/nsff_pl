import torch
from torch import nn
from kornia.filters import filter2d
from einops import reduce, rearrange, repeat
from datasets import ray_utils


def shiftscale_invariant_depthloss(depth, disp):
    """
    Computes the shift- scale- invariant depth loss as proposed in
    https://arxiv.org/pdf/1907.01341.pdf.

    Inputs:
        depth: (N) depth in NDC space.
        disp: (N) disparity in Euclidean space, produced by image-based method.

    Outputs:
        loss: (N)
    """
    t_pred = torch.median(depth)
    s_pred = torch.mean(torch.abs(depth-t_pred))
    t_gt = torch.median(-disp)
    s_gt = torch.mean(torch.abs(-disp-t_gt))

    pred_depth_n = (depth-t_pred)/s_pred
    gt_depth_n = (-disp-t_gt)/s_gt
    loss = (pred_depth_n-gt_depth_n)**2
    return loss


class NeRFWLoss(nn.Module):
    """
    col_l: coarse and fine color loss
    disp_l: monodepth loss
    entropy_l: entropy loss to encourage weights to be concentrated
    cross_entropy_l: encourage static to have different peaks than dynamic
                     @thickness specifies how many intervals to separate the peaks


    ------ flow losses cf. https://arxiv.org/pdf/2011.13084.pdf -------------------
    ------                 https://www.cs.cornell.edu/~zl548/NSFF/NSFF_supp.pdf ---
    L_pho: (Eq 8)
        pho_l: forward+backward color loss
    L_cyc: (Eq 9)
        cyc_l: cycle forward+backward flow loss
    L_geo: (Eq 10, Eq 4-7 in supplementary)
        flow_fw_l: consistent forward 2D-3D flow loss
        flow_bw_l: consistent backward 2D-3D flow loss
    L_reg: (Eq 1, 2, 3 in supplementary)
        reg_min_l: small flow loss
        reg_temp_sm_l: linear flow loss
        reg_sp_sm_l: spatial smoothness flow loss
    """
    def __init__(self,
                 lambda_geo=0.04, lambda_reg=0.1,
                 thickness=1,
                 topk=1.0):
        super().__init__()
        self.lambda_geo_d = self.lambda_geo_f = lambda_geo
        self.lambda_reg = lambda_reg
        self.lambda_ent = 1e-3
        self.z_far = 0.95
        self.thickness_filter = torch.ones(1, 1, max(thickness, 1))

        self.topk = topk

        # self.Ks camera intrinsics (1, 3, 3) and
        # self.Ps world to image projection matrices (1, N_frames, 3, 4) and
        # self.max_t (N_frames-1)
        # are registered as buffers in train.py !

    def forward(self, inputs, targets, **kwargs):
        ret = {}
        ret['col_l'] = reduce((inputs['rgb_fine']-targets['rgbs'])**2,
                              'n1 c -> n1', 'mean')
        if 'rgb_coarse' in inputs:
            ret['col_l'] += 0.1 * reduce((inputs['rgb_coarse']-targets['rgbs'])**2,
                                         'n1 c -> n1', 'mean')
        ret['disp_l'] = self.lambda_geo_d * \
            shiftscale_invariant_depthloss(inputs['depth_fine'], targets['disps'])
        if 'depth_coarse' in inputs:
            ret['disp_l'] += self.lambda_geo_d * \
                shiftscale_invariant_depthloss(inputs['depth_coarse'], targets['disps'])

        if kwargs['output_transient_flow']:
            ret['entropy_l'] = self.lambda_ent * \
                reduce(-inputs['transient_weights_fine']*
                       torch.log(inputs['transient_weights_fine']+1e-8), 'n1 n2 -> n1', 'sum')
            # linearly increase the weight from 0 to lambda_ent/5 in 10 epochs
            cross_entropy_w = self.lambda_ent/5 * min(kwargs['epoch']/10, 1.0)
            # dilate transient_weight with @thickness window
            tr_w = inputs['transient_weights_fine'].detach() # (N_rays, N_samples)
            tr_w = rearrange(tr_w, 'n1 n2 -> 1 1 n1 n2')
            tr_w = filter2d(tr_w, self.thickness_filter, 'constant') # 0-pad
            tr_w = rearrange(tr_w, '1 1 n1 n2 -> n1 n2')
            ret['cross_entropy_l'] = cross_entropy_w * \
                reduce(tr_w*torch.log(inputs['static_weights_fine']+1e-8), 'n1 n2 -> n1', 'sum')

            Ks = self.Ks[targets['cam_ids']] # (N_rays, 3, 3)
            xyz_fw_w = ray_utils.ndc2world(inputs['xyz_fw'], Ks) # (N_rays, 3)
            xyz_bw_w = ray_utils.ndc2world(inputs['xyz_bw'], Ks) # (N_rays, 3)

            ts_fw = torch.clamp(targets['ts']+1, max=self.max_t)
            Ps_fw = self.Ps[targets['cam_ids'], ts_fw] # (N_rays, 3, 4)
            uvd_fw = Ps_fw[:, :3, :3] @ xyz_fw_w.unsqueeze(-1) + Ps_fw[:, :3, 3:]
            uv_fw = uvd_fw[:, :2, 0] / (torch.abs(uvd_fw[:, 2:, 0])+1e-8)

            ts_bw = torch.clamp(targets['ts']-1, min=0)
            Ps_bw = self.Ps[targets['cam_ids'], ts_bw] # (N_rays, 3, 4)
            uvd_bw = Ps_bw[:, :3, :3] @ xyz_bw_w.unsqueeze(-1) + Ps_bw[:, :3, 3:]
            uv_bw = uvd_bw[:, :2, 0] / (torch.abs(uvd_bw[:, 2:, 0])+1e-8)

            # disable geo loss for the first and last frames (no gt for fw/bw)
            # also projected depth must > 0 (must be in front of the camera)
            valid_geo_fw = (uvd_fw[:, 2, 0]>0)&(targets['ts']<self.max_t)
            valid_geo_bw = (uvd_bw[:, 2, 0]>0)&(targets['ts']>0)
            if valid_geo_fw.any():
                ret['flow_fw_l'] = self.lambda_geo_f/2 * \
                    torch.abs(uv_fw[valid_geo_fw]-targets['uv_fw'][valid_geo_fw])
                ret['flow_fw_l'] = reduce(ret['flow_fw_l'], 'n1 c -> n1', 'mean')
            if valid_geo_bw.any():
                ret['flow_bw_l'] = self.lambda_geo_f/2 * \
                    torch.abs(uv_bw[valid_geo_bw]-targets['uv_bw'][valid_geo_bw])
                ret['flow_bw_l'] = reduce(ret['flow_bw_l'], 'n1 c -> n1', 'mean')

            pho_w = cyc_w = 1.0#min(kwargs['epoch']/2, 1.0)
            ret['pho_l'] = pho_w * \
                inputs['disocc_fw']*(inputs['rgb_fw']-targets['rgbs'])**2 / \
                inputs['disocc_fw'].mean()
            ret['pho_l']+= pho_w * \
                inputs['disocc_bw']*(inputs['rgb_bw']-targets['rgbs'])**2 / \
                inputs['disocc_bw'].mean()
            ret['pho_l'] = reduce(ret['pho_l'], 'n1 c -> n1', 'mean')

            ret['cyc_l'] = cyc_w * \
                inputs['disoccs_fw']*torch.abs(inputs['xyzs_fw_bw']-inputs['xyzs_fine']) / \
                inputs['disoccs_fw'].mean()
            ret['cyc_l']+= cyc_w * \
                inputs['disoccs_bw']*torch.abs(inputs['xyzs_bw_fw']-inputs['xyzs_fine']) / \
                inputs['disoccs_bw'].mean()
            ret['cyc_l'] = reduce(ret['cyc_l'], 'n1 n2 c -> n1', 'mean')

            N = inputs['xyzs_fine'].shape[1]
            xyzs_w = ray_utils.ndc2world(inputs['xyzs_fine'][:, :int(N*self.z_far)], Ks)
            xyzs_fw_w = ray_utils.ndc2world(inputs['xyzs_fw'][:, :int(N*self.z_far)], Ks)
            xyzs_bw_w = ray_utils.ndc2world(inputs['xyzs_bw'][:, :int(N*self.z_far)], Ks)
            ret['reg_temp_sm_l'] = self.lambda_reg * torch.abs(xyzs_fw_w+xyzs_bw_w-2*xyzs_w)
            ret['reg_temp_sm_l'] = reduce(ret['reg_temp_sm_l'], 'n1 n2 c -> n1', 'mean')
            ret['reg_min_l'] = self.lambda_reg * (torch.abs(xyzs_fw_w-xyzs_w)+
                                                  torch.abs(xyzs_bw_w-xyzs_w))
            ret['reg_min_l'] = reduce(ret['reg_min_l'], 'n1 n2 c -> n1', 'mean')

            d = torch.norm(xyzs_w[:, 1:]-xyzs_w[:, :-1], dim=-1, keepdim=True)
            sp_w = torch.exp(-2*d) # weight decreases as the distance increases
            sf_fw_w = xyzs_fw_w-xyzs_w # forward scene flow in world coordinate
            sf_bw_w = xyzs_bw_w-xyzs_w # backward scene flow in world coordinate
            ret['reg_sp_sm_l'] = self.lambda_reg * \
                (torch.abs(sf_fw_w[:, 1:]-sf_fw_w[:, :-1])*sp_w+
                 torch.abs(sf_bw_w[:, 1:]-sf_bw_w[:, :-1])*sp_w)
            ret['reg_sp_sm_l'] = reduce(ret['reg_sp_sm_l'], 'n1 n2 c -> n1', 'mean')

        for k, loss in ret.items():
            if 'weights' in kwargs: # use prioritized weights of each ray
                loss = loss * kwargs['weights']
            if self.topk < 1: # use hard example mining
                num_hard_samples = int(self.topk * loss.numel())
                loss, _ = torch.topk(loss.flatten(), num_hard_samples)

            ret[k] = loss.mean()

        return ret

loss_dict = {'nerfw': NeRFWLoss}