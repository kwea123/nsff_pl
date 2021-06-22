import torch
from torch import nn

class PosEmbedding(nn.Module):
    def __init__(self, max_logscale, N_freqs, logscale=True):
        """
        Defines a function that embeds x to (sin(2^k x), cos(2^k x), ...)
        """
        super().__init__()
        self.funcs = [torch.sin, torch.cos]

        if logscale:
            self.freqs = 2**torch.linspace(0, max_logscale, N_freqs)
        else:
            self.freqs = torch.linspace(1, 2**max_logscale, N_freqs)

    def forward(self, x):
        """
        Inputs:
            x: (B, 3)

        Outputs:
            out: (B, 6*N_freqs+3)
        """
        out = [x]
        for freq in self.freqs:
            for func in self.funcs:
                out += [func(freq*x)]

        return torch.cat(out, -1)


class NeRF(nn.Module):
    def __init__(self, typ,
                 D=8, W=256, skips=[4],
                 in_channels_xyz=63, 
                 use_viewdir=True, in_channels_dir=27,
                 encode_appearance=False, in_channels_a=48,
                 encode_transient=False, in_channels_t=16,
                 output_flow=False, flow_scale=0.2):
        """
        ---Parameters for the original NeRF---
        D: number of layers for density (sigma) encoder
        W: number of hidden units in each layer
        skips: add skip connection in the Dth layer
        in_channels_xyz: number of input channels for xyz (3+3*10*2=63 by default)
        use_viewdir: whether to use view dependency for static network
        in_channels_dir: number of input channels for direction (3+3*4*2=27 by default)

        ---Parameters for NeRF-W (transient is used in fine model only as per section 4.3)---
        ---cf. Figure 3 of the paper---
        encode_appearance: whether to add appearance encoding as input (NeRF-A)
        in_channels_a: appearance embedding dimension. n^(a) in the paper
        encode_transient: whether to add transient encoding as input (NeRF-U)
        in_channels_t: transient embedding dimension. n^(tau) in the paper
        flow_scale: how much scale to multiply to flow output (in NDC)
        """
        super().__init__()
        self.typ = typ
        self.D = D
        self.W = W
        self.skips = skips
        self.in_channels_xyz = in_channels_xyz
        self.use_viewdir = use_viewdir
        self.in_channels_dir = in_channels_dir

        self.encode_appearance = False if typ=='coarse' else encode_appearance
        self.in_channels_a = in_channels_a if encode_appearance else 0
        self.encode_transient = encode_transient
        self.in_channels_t = in_channels_t if encode_transient else 0
        self.output_flow = self.encode_transient and output_flow

        # xyz encoding layers
        for i in range(D):
            if i == 0:
                layer = nn.Linear(in_channels_xyz, W)
            elif i in skips:
                layer = nn.Linear(W+in_channels_xyz, W)
            else:
                layer = nn.Linear(W, W)
            layer = nn.Sequential(layer, nn.ReLU(True))
            setattr(self, f"static_xyz_encoding_{i+1}", layer)
        self.static_xyz_encoding_final = nn.Linear(W, W)

        if self.use_viewdir:
            self.static_dir_encoding = nn.Sequential(
                        nn.Linear(W+in_channels_dir+self.in_channels_a, W), nn.ReLU(True))

        # static output layers
        self.static_sigma = nn.Linear(W, 1)
        self.static_rgb = nn.Sequential(nn.Linear(W, 3), nn.Sigmoid())

        if self.encode_transient:
            for i in range(D):
                if i == 0:
                    layer = nn.Linear(in_channels_xyz+in_channels_t, W)
                elif i in skips:
                    layer = nn.Linear(W+in_channels_xyz+in_channels_t, W)
                else:
                    layer = nn.Linear(W, W)
                layer = nn.Sequential(layer, nn.ReLU(True))
                setattr(self, f"transient_xyz_encoding_{i+1}", layer)
            self.transient_xyz_encoding_final = nn.Linear(W, W)

            # if self.use_viewdir:
            #     self.transient_dir_encoding = nn.Sequential(
            #             nn.Linear(W+in_channels_dir+self.in_channels_a, W), nn.ReLU(True))

            # transient output layers
            self.transient_sigma = nn.Linear(W, 1)
            self.transient_rgb = nn.Sequential(nn.Linear(W, 3), nn.Sigmoid())
            if typ == 'fine' and self.output_flow:
                self.flow_scale = flow_scale
                # predict forward and backward flows
                self.transient_flow_fw = nn.Sequential(nn.Linear(W, 3), nn.Tanh())
                self.transient_flow_bw = nn.Sequential(nn.Linear(W, 3), nn.Tanh())

    def forward(self, x, sigma_only=False, 
                output_static=True, output_transient=True,
                output_transient_flow=[]):
        """
        Encodes input (xyz+dir) to rgb+sigma (not ready to render yet).
        For rendering this ray, please see rendering.py
        Inputs:
            x: the embedded vector of position (+ direction + appearance + transient)
            sigma_only: whether to infer sigma only, for coarse model
            output_transient: whether to infer the transient component.
            output_transient_flow: [] or ['fw'] or ['bw'] or ['fw', 'bw'] or ['fw', 'bw', 'disocc']
        Outputs (concatenated):
            if sigma_ony:
                if not output_transient:
                    static_sigma
                else:
                    static_sigma, transient_sigma
            elif output_transient:
                if not output_transient_flow:
                    static_rgb, static_sigma, transient_rgb, transient_sigma
                else:
                    above, transient_flow_fw, transient_flow_bw, transient_disocc
            else:
                static_rgb, static_sigma
        """
        if sigma_only:
            if output_transient:
                input_xyz, input_t = \
                    torch.split(x, [self.in_channels_xyz,
                                    self.in_channels_t], 1)
            else:
                input_xyz = x
        elif output_transient:
            input_xyz, input_dir, input_a, input_t = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir,
                                self.in_channels_a,
                                self.in_channels_t], 1)
        else:
            input_xyz, input_dir, input_a = \
                torch.split(x, [self.in_channels_xyz,
                                self.in_channels_dir,
                                self.in_channels_a], 1)
        
        if output_static:
            xyz_ = input_xyz
            for i in range(self.D):
                if i in self.skips:
                    xyz_ = torch.cat([input_xyz, xyz_], 1)
                xyz_ = getattr(self, f"static_xyz_encoding_{i+1}")(xyz_)

            static_sigma = self.static_sigma(xyz_) # (B, 1)
            if sigma_only: # static and transient sigmas
                if not output_transient:
                    return static_sigma
                xyz_ = torch.cat([input_xyz, input_t], 1)
                for i in range(self.D):
                    if i in self.skips:
                        xyz_ = torch.cat([input_xyz, input_t, xyz_], 1)
                    xyz_ = getattr(self, f"transient_xyz_encoding_{i+1}")(xyz_)
                transient_xyz_encoding_final = self.transient_xyz_encoding_final(xyz_)
                transient_sigma = self.transient_sigma(transient_xyz_encoding_final)
                return torch.cat([static_sigma, transient_sigma], 1)

            feat_final = self.static_xyz_encoding_final(xyz_)
            if self.use_viewdir:
                dir_encoding_input = torch.cat([feat_final, input_dir, input_a], 1)
                feat_final = self.static_dir_encoding(dir_encoding_input)
            static_rgb = self.static_rgb(feat_final) # (B, 3)
            static = torch.cat([static_rgb, static_sigma], 1) # (B, 4)

            if not output_transient:
                return static

        xyz_ = torch.cat([input_xyz, input_t], 1)
        for i in range(self.D):
            if i in self.skips:
                xyz_ = torch.cat([input_xyz, input_t, xyz_], 1)
            xyz_ = getattr(self, f"transient_xyz_encoding_{i+1}")(xyz_)
        feat_final = self.transient_xyz_encoding_final(xyz_)
        transient_sigma = self.transient_sigma(feat_final)
        # if self.use_viewdir:
        #     dir_encoding_input = torch.cat([feat_final, input_dir, input_a], 1)
        #     feat_final = self.transient_dir_encoding(dir_encoding_input)
        transient_rgb = self.transient_rgb(feat_final)

        transient_list = [transient_rgb, transient_sigma] # (B, 4)
        if 'fw' in output_transient_flow:
            transient_list += [self.flow_scale * self.transient_flow_fw(feat_final)]
        if 'bw' in output_transient_flow:
            transient_list += [self.flow_scale * self.transient_flow_bw(feat_final)]

        transient = torch.cat(transient_list, 1) # (B, 12)
        if output_static:
            return torch.cat([static, transient], 1) # (B, 16)
        return transient
