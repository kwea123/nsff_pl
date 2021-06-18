import torch
from torch import nn
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
    tr_far_l: minimize transient sigma at far regions
    entropy_l: entropy loss to encourage weights to be concentrated
    cross_entropy_l: encourage static to have different peaks than dynamic


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
                 topk=1.0):
        super().__init__()
        self.lambda_geo_d = self.lambda_geo_f = lambda_geo
        self.lambda_reg = lambda_reg
        self.lambda_ent = 1e-3
        self.z_far = 0.95

        self.topk = topk

        # self.Ks camera intrinsics (1, 3, 3) and
        # self.Ps world to image projection matrices (1, N_frames, 3, 4) and
        # self.max_t (N_frames-1)
        # are registered as buffers in train.py !

    def forward(self, inputs, targets, **kwargs):
        # if 'weights' in kwargs...
        ret = {}
        ret['col_l'] = (inputs['rgb_fine']-targets['rgbs'])**2
        ret['disp_l'] = self.lambda_geo_d * \
            shiftscale_invariant_depthloss(inputs['depth_fine'], targets['disps'])

        if kwargs['output_transient_flow']:
            ret['entropy_l'] = self.lambda_ent * \
                torch.sum(-inputs['transient_weights_fine']*
                          torch.log(inputs['transient_weights_fine']+1e-8), -1)
            # linearly increase the weight from 0 to lambda_ent in 20 epochs
            cross_entropy_w = self.lambda_ent * min(kwargs['epoch']/20, 1.0)
            ret['cross_entropy_l'] = cross_entropy_w * \
                torch.sum(inputs['transient_weights_fine'].detach()*
                          torch.log(inputs['static_weights_fine']+1e-8), -1)

            Ks = self.Ks[targets['cam_ids']] # (N_rays, 3, 3)
            xyz_fw_w = ray_utils.ndc2world(inputs['xyz_fw'], Ks) # (N_rays, 3)
            xyz_bw_w = ray_utils.ndc2world(inputs['xyz_bw'], Ks) # (N_rays, 3)

            ts_fw = torch.clamp(targets['ts']+1, max=self.max_t)
            Ps_fw = self.Ps[targets['cam_ids'], ts_fw] # (N_rays, 3, 4)
            uvd_fw = Ps_fw[:, :3, :3] @ xyz_fw_w.unsqueeze(-1) + Ps_fw[:, :3, 3:]
            uv_fw = uvd_fw[:, :2, 0] / torch.abs(uvd_fw[:, 2:, 0])+1e-8

            ts_bw = torch.clamp(targets['ts']-1, min=0)
            Ps_bw = self.Ps[targets['cam_ids'], ts_bw] # (N_rays, 3, 4)
            uvd_bw = Ps_bw[:, :3, :3] @ xyz_bw_w.unsqueeze(-1) + Ps_bw[:, :3, 3:]
            uv_bw = uvd_bw[:, :2, 0] / torch.abs(uvd_bw[:, 2:, 0])+1e-8 # (N_rays, 2)

            # disable geo loss for the first and last frames (no gt for fw/bw)
            # also projected depth must > 0 (must be in front of the camera)
            valid_geo_fw = (uvd_fw[:, 2, 0]>0)&(targets['ts']<self.max_t)
            valid_geo_bw = (uvd_bw[:, 2, 0]>0)&(targets['ts']>0)
            if valid_geo_fw.any():
                ret['flow_fw_l'] = self.lambda_geo_f/2 * \
                    torch.abs(uv_fw[valid_geo_fw]-targets['uv_fw'][valid_geo_fw])
            if valid_geo_bw.any():
                ret['flow_bw_l'] = self.lambda_geo_f/2 * \
                    torch.abs(uv_bw[valid_geo_bw]-targets['uv_bw'][valid_geo_bw])

            ret['pho_l'] = \
                inputs['transient_disocc_fw']*(inputs['rgb_fw']-targets['rgbs'])**2 / \
                inputs['transient_disocc_fw'].mean()
            ret['pho_l'] += \
                inputs['transient_disocc_bw']*(inputs['rgb_bw']-targets['rgbs'])**2 / \
                inputs['transient_disocc_bw'].mean()

            ret['cyc_l'] = \
                inputs['transient_disoccs_fw']*torch.abs(inputs['xyzs_fw_bw']-inputs['xyzs_fine']) / \
                inputs['transient_disoccs_fw'].mean()
            ret['cyc_l'] += \
                inputs['transient_disoccs_bw']*torch.abs(inputs['xyzs_bw_fw']-inputs['xyzs_fine']) / \
                inputs['transient_disoccs_bw'].mean()
            ret['disocc_l'] = self.lambda_reg * (1-inputs['transient_disocc_fw']+
                                                 1-inputs['transient_disocc_bw'])

            N = inputs['xyzs_fine'].shape[1]
            xyzs_w = ray_utils.ndc2world(inputs['xyzs_fine'][:, :int(N*self.z_far)], Ks)
            xyzs_fw_w = ray_utils.ndc2world(inputs['xyzs_fw'][:, :int(N*self.z_far)], Ks)
            xyzs_bw_w = ray_utils.ndc2world(inputs['xyzs_bw'][:, :int(N*self.z_far)], Ks)
            ret['reg_temp_sm_l'] = self.lambda_reg * torch.abs(xyzs_fw_w+xyzs_bw_w-2*xyzs_w)
            ret['reg_min_l'] = self.lambda_reg * (torch.abs(xyzs_fw_w-xyzs_w)+
                                                  torch.abs(xyzs_bw_w-xyzs_w))

            d = torch.norm(xyzs_w[:, 1:]-xyzs_w[:, :-1], dim=-1, keepdim=True)
            sp_w = torch.exp(-2*d) # weight decreases as the distance increases
            sf_fw_w = xyzs_fw_w-xyzs_w # forward scene flow in world coordinate
            sf_bw_w = xyzs_bw_w-xyzs_w # backward scene flow in world coordinate
            ret['reg_sp_sm_l'] = self.lambda_reg * \
                (torch.abs(sf_fw_w[:, 1:]-sf_fw_w[:, :-1])*sp_w+
                 torch.abs(sf_bw_w[:, 1:]-sf_bw_w[:, :-1])*sp_w)

        for k, loss in ret.items():
            if self.topk < 1:
                num_hard_samples = int(self.topk * loss.numel())
                loss, _ = torch.topk(loss.flatten(), num_hard_samples)
            ret[k] = loss.mean()

        return ret

loss_dict = {'nerfw': NeRFWLoss}