import torch
from einops import rearrange, reduce, repeat
from datasets import ray_utils

# for frame interpolation
from kornia import create_meshgrid
from .softsplat import FunctionSoftsplat


def sample_pdf(bins, weights, N_importance, det=False, eps=1e-5):
    """
    Sample @N_importance samples from @bins with distribution defined by @weights.
    Inputs:
        bins: (N_rays, N_samples_+1) where N_samples_ is "the number of coarse samples per ray - 2"
        weights: (N_rays, N_samples_)
        N_importance: the number of samples to draw from the distribution
        det: deterministic or not
        eps: a small number to prevent division by zero
    Outputs:
        samples: the sampled samples
    """
    N_rays, N_samples_ = weights.shape
    weights = weights + eps # prevent division by zero (don't do inplace op!)
    pdf = weights / reduce(weights, 'n1 n2 -> n1 1', 'sum') # (N_rays, N_samples_)
    cdf = torch.cumsum(pdf, -1) # (N_rays, N_samples), cumulative distribution function
    cdf = torch.cat([torch.zeros_like(cdf[: ,:1]), cdf], -1)  # (N_rays, N_samples_+1) 
                                                               # padded to 0~1 inclusive

    if det:
        u = torch.linspace(0, 1, N_importance, device=bins.device)
        u = u.expand(N_rays, N_importance)
    else:
        u = torch.rand(N_rays, N_importance, device=bins.device)
    u = u.contiguous()

    inds = torch.searchsorted(cdf, u, right=True)
    below = torch.clamp_min(inds-1, 0)
    above = torch.clamp_max(inds, N_samples_)

    inds_sampled = rearrange(torch.stack([below, above], -1), 'n1 n2 c -> n1 (n2 c)', c=2)
    cdf_g = rearrange(torch.gather(cdf, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)
    bins_g = rearrange(torch.gather(bins, 1, inds_sampled), 'n1 (n2 c) -> n1 n2 c', c=2)

    denom = cdf_g[...,1]-cdf_g[...,0]
    denom[denom<eps] = 1 # denom equals 0 means a bin has weight 0, in which case it will not be sampled
                         # anyway, therefore any value for it is fine (set to 1 here)

    samples = bins_g[...,0] + (u-cdf_g[...,0])/denom * (bins_g[...,1]-bins_g[...,0])
    return samples


def render_rays(models,
                embeddings,
                rays,
                ts,
                max_t,
                N_samples=64,
                perturb=0,
                noise_std=0,
                N_importance=0,
                chunk=1024*32,
                test_time=False,
                **kwargs):
    """
    Render rays by computing the output of @model applied on @rays
    Inputs:
        models: list of NeRF models (coarse and fine) defined in nerf.py
        embeddings: list of embedding models of origin and direction defined in nerf.py
        rays: (N_rays, 3+3), ray origins and directions
        ts: (N_rays) or None, ray time (None if not output_transient)
        max_t: int, max ray time (self.N_frames-1 in datasets/lightfield.py)
        N_samples: number of coarse samples per ray
        perturb: factor to perturb the sampling position on the ray (for coarse model only)
        noise_std: factor to perturb the model's prediction of sigma
        N_importance: number of fine samples per ray
        chunk: the chunk size in batched inference
        test_time: whether it is test (inference only) or not. If True, it will not do inference
                   on coarse rgb to save time
    Outputs:
        result: dictionary containing final rgb and depth maps for coarse and fine models
    """

    def inference(results, model, xyz, zs, test_time=False, **kwargs):
        """
        Helper function that performs model inference.
        Inputs:
            results: a dict storing all results
            model: NeRF model (coarse or fine)
            xyz: (N_rays, N_samples_, 3) sampled positions
                  N_samples_ is the number of sampled points in each ray;
                             = N_samples for coarse model
                             = N_samples+N_importance for fine model
                             +1 if add new objects in kwargs
            zs: (N_rays, N_samples_) depths of the sampled positions
            test_time: test time or not
        """

        def render_transient_warping(xyz, t_embedded, flow):
            """
            Helper function that performs forward or backward warping for dynamic scenes.
            static sigma and rgbs of the CURRENT time are used to composite the result.
            Inputs:
                xyz: warped xyz
                t_embedded: embedded time for the warping time instance (t+i)
                flow: 'fw' or 'bw', the flow for the warped xyz

            Outputs:
                rgb_map_warped: (N_rays, 3) warped rendering
                transient_flows_: (N_rays, N_samples_, 3) warped points' fw/bw flow
            """
            out_chunks = []
            for i in range(0, B, chunk):
                inputs = [embedding_xyz(xyz[i:i+chunk]), dir_embedded_[i:i+chunk]]
                if model.encode_appearance: inputs += [a_embedded_[i:i+chunk]]
                inputs+= [t_embedded[i:i+chunk]]
                out_chunks += [model(torch.cat(inputs, 1),
                                     output_static=False,
                                     output_transient=True,
                                     output_transient_flow=[flow])]
            out = torch.cat(out_chunks, 0)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_)
            transient_rgbs_w = out[..., :3]
            transient_sigmas_w = out[..., 3]
            transient_flows_w = out[..., 4:7]
            transient_flows_w[zs>z_far] = 0

            noise = torch.randn_like(transient_sigmas_w) * noise_std
            transient_alphas_w = 1-torch.exp(-deltas*act(transient_sigmas_w+noise))
            alphas_w = 1-(1-static_alphas)*(1-transient_alphas_w)
            alphas_w_shifted = torch.cat([torch.ones_like(alphas_w[:, :1]), 1-alphas_w], -1)
            transmittance_w = torch.cumprod(alphas_w_shifted[:, :-1], -1)
            static_weights_w = rearrange(static_alphas*transmittance_w, 'n1 n2 -> n1 n2 1')
            transient_weights_w = rearrange(transient_alphas_w*transmittance_w, 'n1 n2 -> n1 n2 1')
            static_rgb_map_w = reduce(static_weights_w*static_rgbs, 'n1 n2 c -> n1 c', 'sum')
            transient_rgb_map_w = \
                reduce(transient_weights_w*transient_rgbs_w, 'n1 n2 c -> n1 c', 'sum')
            rgb_map_w = static_rgb_map_w + transient_rgb_map_w
            return rgb_map_w, transient_flows_w

        typ = model.typ
        results[f'zs_{typ}'] = zs
        results[f'xyzs_{typ}'] = xyz
        N_samples_ = xyz.shape[1]
        xyz_ = rearrange(xyz, 'n1 n2 c -> (n1 n2) c', c=3)

        # Perform model inference to get rgb and raw sigma
        B = xyz_.shape[0]
        out_chunks = []
        if typ=='coarse' and test_time:
            if output_transient:
                t_embedded_ = repeat(t_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            for i in range(0, B, chunk):
                inputs = [embedding_xyz(xyz_[i:i+chunk])]
                if output_transient: inputs += [t_embedded_[i:i+chunk]]
                out_chunks += [model(torch.cat(inputs, 1), sigma_only=True,
                                     output_transient=output_transient)]
            out = torch.cat(out_chunks, 0)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_)
            static_sigmas = out[..., 0]
            if output_transient: transient_sigmas = out[..., 1]
        else: # infer rgb and sigma and others
            dir_embedded_ = repeat(dir_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            if model.encode_appearance:
                a_embedded_ = repeat(a_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            if output_transient:
                t_embedded_ = repeat(t_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
            for i in range(0, B, chunk):
                inputs = [embedding_xyz(xyz_[i:i+chunk]), dir_embedded_[i:i+chunk]]
                if model.encode_appearance: inputs += [a_embedded_[i:i+chunk]]
                if output_transient: inputs += [t_embedded_[i:i+chunk]]
                out_chunks += [model(torch.cat(inputs, 1),
                                     output_transient=output_transient,
                                     output_transient_flow=output_transient_flow)]

            out = torch.cat(out_chunks, 0)
            out = rearrange(out, '(n1 n2) c -> n1 n2 c', n1=N_rays, n2=N_samples_)
            results[f'static_rgbs_{typ}'] = static_rgbs = out[..., :3]
            static_sigmas = out[..., 3]
            if output_transient:
                results[f'transient_rgbs_{typ}'] = transient_rgbs = out[..., 4:7]
                transient_sigmas = out[..., 7]
                if output_transient_flow: # only [] or ['fw', 'bw'] or ['fw', 'bw', 'disocc'] !
                    results['transient_flows_fw'] = transient_flows_fw = out[..., 8:11]
                    results['transient_flows_bw'] = transient_flows_bw = out[..., 11:14]
                    transient_flows_fw[zs>z_far] = 0
                    transient_flows_bw[zs>z_far] = 0
                if 'disocc' in output_transient_flow:
                    transient_disoccs_fw = out[..., 14]
                    transient_disoccs_bw = out[..., 15]

        # set invisible transient_sigmas to a very negative value
        if test_time and output_transient and 'dataset' in kwargs:
            dataset = kwargs['dataset']
            K = dataset.Ks[0].to(xyz.device)
            visibilities = torch.zeros(len(xyz_), device=xyz.device)
            xyz_w = ray_utils.ndc2world(xyz_, K)
            for i in range(len(dataset.cam_train)):
                ray_utils.compute_world_visiblility(visibilities,
                    xyz_w, K, dataset.img_wh[1], dataset.img_wh[0],
                    torch.FloatTensor(dataset.poses[i*dataset.N_frames+ts[0]]).to(xyz.device))
            transient_sigmas[visibilities.view_as(transient_sigmas)==0] = -10

        deltas = zs[:, 1:] - zs[:, :-1] # (N_rays, N_samples_-1)
        deltas = torch.cat([deltas, 100*torch.ones_like(deltas[:, :1])], -1)

        noise = torch.randn_like(static_sigmas) * noise_std
        results[f'static_sigmas_{typ}'] = static_sigmas = act(static_sigmas+noise)
        alphas = 1-torch.exp(-deltas*static_sigmas)

        if output_transient:
            static_alphas = alphas
            noise = torch.randn_like(transient_sigmas) * noise_std
            results[f'transient_sigmas_{typ}'] = transient_sigmas = act(transient_sigmas+noise)
            transient_alphas = 1-torch.exp(-deltas*transient_sigmas)
            alphas = 1-(1-static_alphas)*(1-transient_alphas)

            if (not test_time) and output_transient_flow: # render with flowed-xyzs
                results['xyzs_fw'] = xyz_fw = xyz + transient_flows_fw
                xyz_fw_ = rearrange(xyz_fw, 'n1 n2 c -> (n1 n2) c', c=3)
                tp1_embedded = embeddings['t'](torch.clamp(ts+1, max=max_t)) # t+1
                tp1_embedded_ = repeat(tp1_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
                results['rgb_fw'], transient_flows_fw_bw = \
                    render_transient_warping(xyz_fw_, tp1_embedded_, 'bw')
                results['xyzs_bw'] = xyz_bw = xyz + transient_flows_bw
                xyz_bw_ = rearrange(xyz_bw, 'n1 n2 c -> (n1 n2) c', c=3)
                tm1_embedded = embeddings['t'](torch.clamp(ts-1, min=0)) # t-1
                tm1_embedded_ = repeat(tm1_embedded, 'n1 c -> (n1 n2) c', n2=N_samples_)
                results['rgb_bw'], transient_flows_bw_fw = \
                    render_transient_warping(xyz_bw_, tm1_embedded_, 'fw')
                # to compute fw-bw consistency
                results['xyzs_fw_bw'] = xyz_fw + transient_flows_fw_bw
                results['xyzs_bw_fw'] = xyz_bw + transient_flows_fw_bw

        alphas_shifted = torch.cat([torch.ones_like(alphas[:, :1]), 1-alphas], -1)
        transmittance = torch.cumprod(alphas_shifted[:, :-1], -1)

        if output_transient:
            static_weights = static_alphas * transmittance
            transient_weights = transient_alphas * transmittance

        weights = alphas * transmittance # (N_rays, N_samples_)
        weights_ = rearrange(weights, 'n1 n2 -> n1 n2 1')

        if output_transient:
            results[f'static_weights_{typ}'] = static_weights
            results[f'transient_weights_{typ}'] = transient_weights
            results[f'static_alphas_{typ}'] = static_alphas
            results[f'transient_alphas_{typ}'] = transient_alphas
            results[f'weights_{typ}'] = weights
        else:
            results[f'static_weights_{typ}'] = weights
            results[f'static_alphas_{typ}'] = alphas
        if test_time:
            results[f'xyz_{typ}'] = reduce(weights_*xyz, 'n1 n2 c -> n1 c', 'sum')
            if typ == 'coarse':
                return

        results[f'depth_{typ}'] = reduce(weights*zs, 'n1 n2 -> n1', 'sum')
        if output_transient:
            static_rgb_map = reduce(rearrange(static_weights, 'n1 n2 -> n1 n2 1')*static_rgbs,
                                    'n1 n2 c -> n1 c', 'sum')
            transient_weights_ = rearrange(transient_weights, 'n1 n2 -> n1 n2 1')
            transient_rgb_map = reduce(transient_weights_*transient_rgbs, 'n1 n2 c -> n1 c', 'sum')
            results[f'rgb_{typ}'] = static_rgb_map + transient_rgb_map
            results[f'transient_alpha_{typ}'] = reduce(transient_weights, 'n1 n2 -> n1', 'sum')

            # Compute also depth and rgb when only one field exists.
            # The result is different from when both fields exist, since the transimttance
            # will change.
            static_alphas_shifted = \
                torch.cat([torch.ones_like(static_alphas[:, :1]), 1-static_alphas], -1)
            static_transmittance = torch.cumprod(static_alphas_shifted[:, :-1], -1)
            results[f'_static_weights_{typ}'] = \
                _static_weights = static_alphas * static_transmittance
            _static_weights_ = rearrange(_static_weights, 'n1 n2 -> n1 n2 1')
            results[f'_static_rgb_{typ}'] = \
                reduce(_static_weights_*static_rgbs, 'n1 n2 c -> n1 c', 'sum')
            results[f'_static_depth_{typ}'] = \
                reduce(_static_weights*zs, 'n1 n2 -> n1', 'sum')

            if output_transient_flow:
                results['xyz_fine'] = reduce(weights_*xyz, 'n1 n2 c-> n1 c', 'sum')
                results['transient_flow_fw'] = \
                    reduce(transient_weights_*transient_flows_fw, 'n1 n2 c -> n1 c', 'sum')
                results['xyz_fw'] = results['xyz_fine']+results['transient_flow_fw']
                results['transient_flow_bw'] = \
                    reduce(transient_weights_*transient_flows_bw, 'n1 n2 c -> n1 c', 'sum')
                results['xyz_bw'] = results['xyz_fine']+results['transient_flow_bw']

                if (not test_time) and 'disocc' in output_transient_flow:
                    results['transient_disocc_fw'] = \
                        reduce(weights*transient_disoccs_fw, 'n1 n2 -> n1 1', 'sum')
                    results['transient_disoccs_fw'] = \
                        rearrange(transient_disoccs_fw, 'n1 n2 -> n1 n2 1')
                    results['transient_disocc_bw'] = \
                        reduce(weights*transient_disoccs_bw, 'n1 n2 -> n1 1', 'sum')
                    results['transient_disoccs_bw'] = \
                        rearrange(transient_disoccs_bw, 'n1 n2 -> n1 n2 1')

            if test_time:
                # # compute transient weight when it exists solely.
                # transient_alphas_shifted = \
                #     torch.cat([torch.ones_like(transient_alphas[:, :1]), 1-transient_alphas], -1)
                # transient_transmittance = torch.cumprod(transient_alphas_shifted[:, :-1], -1)
                # _transient_weights = transient_alphas * transient_transmittance
                # _transient_weights_ = rearrange(_transient_weights, 'n1 n2 -> n1 n2 1')
                # results[f'_transient_depth_{typ}'] = \
                #     reduce(_transient_weights*zs, 'n1 n2 -> n1', 'sum')
                results[f'transient_rgb_{typ}'] = transient_rgb_map + \
                    0.8*(1-rearrange(results['transient_alpha_fine'], 'n1 -> n1 1')) # gray bg
                # results[f'transient_depth_{typ}'] = \
                #     reduce(transient_weights*zs, 'n1 n2 -> n1', 'sum')
                
        else: # no transient field
            results[f'rgb_{typ}'] = reduce(weights_*static_rgbs, 'n1 n2 c -> n1 c', 'sum')

        return


    results = {}
    N_rays = rays.shape[0]
    act = torch.nn.Softplus() # sigma activation function
    rays_o, rays_d = rays[:, 0:3], rays[:, 3:6] # both (N_rays, 3)
    embedding_xyz, embedding_dir = embeddings['xyz'], embeddings['dir']
    dir_embedded = embedding_dir(kwargs.get('view_dir', rays_d))

    rays_o = rearrange(rays_o, 'n1 c -> n1 1 c')
    rays_d = rearrange(rays_d, 'n1 c -> n1 1 c')

    # coarse sample depths (same for static and transient)
    zs = torch.linspace(0, 1, N_samples, device=rays.device).expand(N_rays, N_samples)
    zs_mid = 0.5 * (zs[: ,:-1]+zs[: ,1:]) # (N_rays, N_samples-1) interval mid points
    z_far = 0.95 # explicitly zero the flow if z exceeds this value
    
    if perturb > 0: # perturb sample depths
        # get intervals between samples
        upper = torch.cat([zs_mid, zs[: ,-1:]], -1)
        lower = torch.cat([zs[: ,:1], zs_mid], -1)
        
        perturb_rand = perturb * torch.rand_like(zs)
        zs = lower + (upper - lower) * perturb_rand

    if N_importance > 0: # coarse to fine
        model = models['coarse']
        output_transient = kwargs.get('output_transient', True) and model.encode_transient
        output_transient_flow = [] # no flow for coarse model
        if output_transient:
            t_embedded = kwargs['t_embedded'] if 't_embedded' in kwargs else embeddings['t'](ts)
        xyz_coarse = rays_o + rays_d * rearrange(zs, 'n1 n2 -> n1 n2 1')
        inference(results, model, xyz_coarse, zs, test_time, **kwargs)

        zs_static = \
            sample_pdf(zs_mid, results['static_weights_coarse'][:, 1:-1].detach(),
                       N_importance, det=perturb==0)
        zs_list = [zs, zs_static]
        if test_time: results['static_zs_fine'] = zs_static

        if output_transient:
            zs_transient = \
                sample_pdf(zs_mid, results['transient_weights_coarse'][:, 1:-1].detach(),
                            N_importance, det=perturb==0)
            zs_list += [zs_transient]
            if test_time: results['transient_zs_fine'] = zs_transient

        zs = torch.sort(torch.cat(zs_list, -1), -1)[0]

    model = models['fine']
    if model.encode_appearance:
        a_embedded = kwargs['a_embedded'] if 'a_embedded' in kwargs else embeddings['a'](ts)
    if N_importance == 0:
        output_transient = kwargs.get('output_transient', True) and model.encode_transient
        if output_transient:
            t_embedded = kwargs['t_embedded'] if 't_embedded' in kwargs else embeddings['t'](ts)
    output_transient_flow = \
        [] if not output_transient else kwargs.get('output_transient_flow', [])
    xyz_fine = rays_o + rays_d * rearrange(zs, 'n1 n2 -> n1 n2 1')
    inference(results, model, xyz_fine, zs, test_time, **kwargs)

    return results


def interpolate(results_t, results_tp1, dt, K, c2w, img_wh):
    """
    Interpolate between two results t and t+1 to produce t+dt, dt in (0, 1).
    For each sample on the ray (the sample points lie on the same distances, so they
    actually form planes), compute the optical flow on this plane, then use softsplat
    to splat the flows. Finally use MPI technique to compute the composite image.
    Used in test time only.

    Inputs:
        results_t, results_tp1: dictionaries of the @render_rays function.
        dt: float in (0, 1)
        K: (3, 3) intrinsics matrix (MUST BE THE SAME for results_t and results_tp1!)
        c2w: (3, 4) current pose (MUST BE THE SAME for results_t and results_tp1!)
        img_wh: image width and height

    Outputs:
        (img_wh[1], img_wh[0], 3) rgb interpolation result
    """
    device = results_t['xyzs_fine'].device
    N_rays, N_samples = results_t['xyzs_fine'].shape[:2]
    w, h = img_wh
    ret_rgba = torch.zeros((h, w, 4), device=device)

    c2w_ = torch.eye(4)
    c2w_[:3] = c2w
    w2c = torch.inverse(c2w_)[:3]
    w2c[1:] *= -1 # "right up back" to "right down forward" for cam projection
    P = K @ w2c # (3, 4) projection matrix
    grid = create_meshgrid(h, w, False, device) # (1, h, w, 2)
    xyzs = results_t['xyzs_fine'] # equals results_tp1['xyzs_fine']

    # static buffers
    static_rgb = rearrange(results_t['static_rgbs_fine'],
                           '(h w) n2 c -> h w n2 c', w=w, h=h, c=3)
    static_a = rearrange(results_t['static_alphas_fine'], '(h w) n2 -> h w n2 1', w=w, h=h)

    # compute forward buffers
    xyzs_w = ray_utils.ndc2world(rearrange(xyzs, 'n1 n2 c -> (n1 n2) c'), K)
    xyzs_fw_w = ray_utils.ndc2world(
                    rearrange(xyzs+results_t['transient_flows_fw'],
                    'n1 n2 c -> (n1 n2) c'), K) # fw points with full flow
    xyzs_fw_w = xyzs_w + dt*(xyzs_fw_w-xyzs_w) # scale the flow with dt
    uvds_fw = P[:3, :3] @ rearrange(xyzs_fw_w, 'n c -> c n') + P[:3, 3:]
    uvs_fw = uvds_fw[:2] / uvds_fw[2]
    uvs_fw = rearrange(uvs_fw, 'c (n1 n2) -> c n1 n2', n1=N_rays, n2=N_samples)
    uvs_fw = rearrange(uvs_fw, 'c (h w) n2 -> n2 h w c', w=w, h=h)
    of_fw = rearrange(uvs_fw-grid, 'n2 h w c -> n2 c h w', c=2)

    transient_rgb_t = rearrange(results_t['transient_rgbs_fine'],
                                '(h w) n2 c -> n2 c h w', w=w, h=h, c=3)
    transient_a_t = rearrange(results_t['transient_alphas_fine'],
                              '(h w) n2 -> n2 1 h w', w=w, h=h)
    transient_rgba_t = torch.cat([transient_rgb_t, transient_a_t], 1)

    # compute backward buffers
    xyzs_bw_w = ray_utils.ndc2world(
                    rearrange(xyzs+results_tp1['transient_flows_bw'],
                    'n1 n2 c -> (n1 n2) c'), K) # bw points with full flow
    xyzs_bw_w = xyzs_w + (1-dt)*(xyzs_bw_w-xyzs_w) # scale the flow with 1-dt
    uvds_bw = P[:3, :3] @ rearrange(xyzs_bw_w, 'n c -> c n') + P[:3, 3:]
    uvs_bw = uvds_bw[:2] / uvds_bw[2]
    uvs_bw = rearrange(uvs_bw, 'c (n1 n2) -> c n1 n2', n1=N_rays, n2=N_samples)
    uvs_bw = rearrange(uvs_bw, 'c (h w) n2 -> n2 h w c', w=w, h=h)
    of_bw = rearrange(uvs_bw-grid, 'n2 h w c -> n2 c h w', c=2)

    transient_rgb_tp1 = rearrange(results_tp1['transient_rgbs_fine'],
                                  '(h w) n2 c -> n2 c h w', w=w, h=h, c=3)
    transient_a_tp1 = rearrange(results_tp1['transient_alphas_fine'],
                                '(h w) n2 -> n2 1 h w', w=w, h=h)
    transient_rgba_tp1 = torch.cat([transient_rgb_tp1, transient_a_tp1], 1)
    
    for s in range(N_samples): # compute MPI planes (front to back composition)
        transient_rgba_fw = FunctionSoftsplat(tenInput=transient_rgba_t[s:s+1].cuda(), 
                                              tenFlow=of_fw[s:s+1].cuda(), 
                                              tenMetric=None, 
                                              strType='average').cpu()
        transient_rgba_fw = rearrange(transient_rgba_fw, '1 c h w -> h w c')
        transient_rgba_bw = FunctionSoftsplat(tenInput=transient_rgba_tp1[s:s+1].cuda(), 
                                              tenFlow=of_bw[s:s+1].cuda(), 
                                              tenMetric=None, 
                                              strType='average').cpu()
        transient_rgba_bw = rearrange(transient_rgba_bw, '1 c h w -> h w c')
        composed_rgb = transient_rgba_fw[..., :3]*transient_rgba_fw[..., 3:]*(1-dt) + \
                       transient_rgba_bw[..., :3]*transient_rgba_bw[..., 3:]*dt + \
                       static_rgb[:, :, s]*static_a[:, :, s]
        composed_a = 1 - (1-(transient_rgba_fw[..., 3:]*(1-dt)+
                             transient_rgba_bw[..., 3:]*dt)) * \
                         (1-static_a[:, :, s])
        ret_rgba[..., :3] += (1-ret_rgba[..., 3:])*composed_rgb
        ret_rgba[..., 3:] += (1-ret_rgba[..., 3:])*composed_a

    return ret_rgba[..., :3]