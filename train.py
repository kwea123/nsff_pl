import os
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import PosEmbedding, NeRF
from models.rendering import render_rays

# optimizer, scheduler, visualization
from utils import *
from torchvision.utils import make_grid

# losses
from losses import loss_dict

# metrics
from metrics import *
import third_party.lpips.lpips.lpips as lpips_model

# pytorch-lightning
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TestTubeLogger
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning import seed_everything

seed_everything(42, workers=True)


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)

        # losses and metrics
        self.loss = \
            loss_dict['nerfw'](lambda_geo=self.hparams.lambda_geo_init,
                               thickness=self.hparams.thickness,
                               topk=self.hparams.topk)

        # models
        self.embedding_xyz = PosEmbedding(hparams.S_emb_xyz, hparams.N_emb_xyz)
        self.embedding_dir = PosEmbedding(hparams.S_emb_dir, hparams.N_emb_dir)
        self.embeddings = {'xyz': self.embedding_xyz,
                           'dir': self.embedding_dir}

        if hparams.encode_a:
            self.embedding_a = torch.nn.Embedding(hparams.N_vocab, hparams.N_a)
            self.embeddings['a'] = self.embedding_a
            load_ckpt(self.embedding_a, hparams.weight_path, 'embedding_a')
        if hparams.encode_t:
            self.embedding_t = torch.nn.Embedding(hparams.N_vocab, hparams.N_tau)
            self.embeddings['t'] = self.embedding_t
            load_ckpt(self.embedding_t, hparams.weight_path, 'embedding_t')

        self.output_transient = hparams.encode_t
        self.output_transient_flow = ['fw', 'bw', 'disocc'] if hparams.encode_t else []

        # fine model always exists
        self.nerf_fine = NeRF(typ='fine',
                              in_channels_xyz=6*hparams.N_emb_xyz+3,
                              use_viewdir=hparams.use_viewdir,
                              in_channels_dir=6*hparams.N_emb_dir+3,
                              encode_appearance=hparams.encode_a,
                              in_channels_a=hparams.N_a,
                              encode_transient=hparams.encode_t,
                              in_channels_t=hparams.N_tau,
                              output_flow=len(self.output_transient_flow)>0,
                              flow_scale=hparams.flow_scale)
        self.models = {'fine': self.nerf_fine}
        load_ckpt(self.nerf_fine, hparams.weight_path,
                  'nerf_fine', hparams.prefixes_to_ignore)

        if hparams.N_importance > 0: # coarse to fine
            self.nerf_coarse = NeRF(typ='coarse',
                                    in_channels_xyz=6*hparams.N_emb_xyz+3,
                                    use_viewdir=hparams.use_viewdir,
                                    in_channels_dir=6*hparams.N_emb_dir+3,
                                    encode_transient=hparams.encode_t,
                                    in_channels_t=hparams.N_tau)
            self.models['coarse'] = self.nerf_coarse
            load_ckpt(self.nerf_coarse, hparams.weight_path,
                    'nerf_coarse', hparams.prefixes_to_ignore)

        self.models_to_train = [self.models]
        if hparams.encode_a: self.models_to_train += [self.embedding_a]
        if hparams.encode_t: self.models_to_train += [self.embedding_t]

    def get_progress_bar_dict(self):
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def forward(self, rays, ts, test_time=False, **kwargs):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        kwargs_ = {}
        for k, v in kwargs.items(): kwargs_[k] = v
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            None if ts is None else ts[i:i+self.hparams.chunk],
                            self.train_dataset.N_frames-1,
                            self.hparams.N_samples,
                            self.hparams.perturb if not test_time else 0,
                            self.hparams.noise_std if not test_time else 0,
                            self.hparams.N_importance,
                            self.hparams.chunk//4 if test_time else self.hparams.chunk,
                            **kwargs_)

            for k, v in rendered_ray_chunks.items():
                if test_time: v = v.cpu()
                results[k] += [v]
        for k, v in results.items(): results[k] = torch.cat(v, 0)
        return results

    def setup(self, stage):
        dataset = dataset_dict[self.hparams.dataset_name]
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh),
                  'start_end': tuple(self.hparams.start_end),
                  'cache_dir': self.hparams.cache_dir,
                  'prioritized_replay': self.hparams.prioritized_replay}
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)

        if self.output_transient_flow:
            self.loss.register_buffer('Ks', self.train_dataset.Ks)
            self.loss.register_buffer('Ps', self.train_dataset.Ps)
            self.loss.max_t = self.train_dataset.N_frames-1

    def configure_optimizers(self):
        kwargs = {}
        self.optimizer = get_optimizer(self.hparams, self.models_to_train, **kwargs)
        if self.hparams.lr_scheduler == 'const': return self.optimizer

        scheduler = get_scheduler(self.hparams, self.optimizer)
        return [self.optimizer], [scheduler]

    def train_dataloader(self):
        self.train_dataset.batch_size = self.hparams.batch_size

        return DataLoader(self.train_dataset,
                          shuffle=True,
                          num_workers=4,
                          batch_size=None,
                          pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=None,
                          pin_memory=True)

    def on_epoch_start(self):
        # for evaluation
        if not hasattr(self, 'lpips_m'):
            self.lpips_m = lpips_model.LPIPS(net='alex', spatial=True)

    def on_train_epoch_start(self):
        self.loss.lambda_geo_d = self.hparams.lambda_geo_init * 0.1**(self.current_epoch//10)
        self.loss.lambda_geo_f = self.hparams.lambda_geo_init * 0.1**(self.current_epoch//10)

    def training_step(self, batch, batch_nb):
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch.get('ts', None)
        kwargs = {'epoch': self.current_epoch,
                  'output_transient': self.output_transient,
                  'output_transient_flow': self.output_transient_flow}
        results = self(rays, ts, **kwargs)
        # if self.train_dataset.prioritized_replay:
        #     new_priorities = ((results['rgb_fine']-rgbs)**2).mean(-1)+1e-8
        #     self.train_dataset.replay_buffers[ts[0].item()] \
        #         .update_priorities(batch['batch_idxs'], new_priorities.detach().cpu().numpy())
        #     kwargs['weights'] = batch['weights']

        loss_d = self.loss(results, batch, **kwargs)
        loss = sum(l for l in loss_d.values())

        with torch.no_grad():
            psnr_ = psnr(results['rgb_fine'], rgbs)

        self.log('lr', get_learning_rate(self.optimizer))
        self.log('train/loss', loss)
        for k, v in loss_d.items(): self.log(f'train/{k}', v, prog_bar=True)
        self.log('train/psnr', psnr_, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_nb):
        rays, rgbs, ts = batch['rays'], batch['rgbs'], batch.get('ts', None)
        batch['rgbs'] = rgbs = rgbs.cpu() # (H*W, 3)
        if 'mask' in batch: mask = batch['mask'].cpu() # (H*W)
        if 'disp' in batch: disp = batch['disp'].cpu() # (H*W)
        kwargs = {'output_transient': self.output_transient,
                  'output_transient_flow': []}
        results = self(rays, ts, test_time=True, **kwargs)

        # compute error metrics
        W, H = self.hparams.img_wh
        img = torch.clip(results['rgb_fine'].view(H, W, 3).permute(2, 0, 1).cpu(), 0, 1)
        img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu()

        rmse_map = ((img_gt-img)**2).mean(0)**0.5
        rmse_map_b = blend_images(img, visualize_depth(-rmse_map), 0.5)

        ssim_map = ssim(img_gt.permute(1, 2, 0), img.permute(1, 2, 0), reduction='none').mean(-1)
        ssim_map_b = blend_images(img, visualize_depth(-ssim_map), 0.5)
        if self.hparams.prioritized_replay: # update weights
            # high ssim = low priority
            self.train_dataset.weights[ts[0].item()] = 1-ssim_map.numpy().flatten()

        lpips_map = lpips(self.lpips_m, img_gt.permute(1, 2, 0), img.permute(1, 2, 0), reduction='none')
        lpips_map_b = blend_images(img, visualize_depth(-lpips_map), 0.5)
    
        if self.hparams.prioritized_replay:
            batch_nb_to_visualize = self.train_dataset.N_frames//2 # visualize the middle image
        else:
            batch_nb_to_visualize = 0
        if batch_nb == batch_nb_to_visualize: 
            depth = visualize_depth(results['depth_fine'].view(H, W))
            img_list = [img_gt, img, depth]
            if self.output_transient:
                img_list += [visualize_mask(results['transient_alpha_fine'].view(H, W))]
                img_list += [torch.clip(results['_static_rgb_fine'].view(H, W, 3).permute(2, 0, 1).cpu(), 0, 1)]
                img_list += [visualize_depth(results['_static_depth_fine'].view(H, W))]
            if 'mask' in batch: img_list += [visualize_mask(1-mask.view(H, W))]
            if 'disp' in batch: img_list += [visualize_depth(-disp.view(H, W))]
            img_grid = make_grid(img_list, nrow=3) # 3 images per row
            self.logger.experiment.add_image('reconstruction/decomposition', img_grid, self.global_step)
            self.logger.experiment.add_image('error_map/rmse', rmse_map_b, self.global_step)
            self.logger.experiment.add_image('error_map/ssim', ssim_map_b, self.global_step)
            self.logger.experiment.add_image('error_map/lpips', lpips_map_b, self.global_step)

        log = {'val_psnr': psnr(results['rgb_fine'], rgbs),
               'val_ssim': ssim_map.mean(),
               'val_lpips': lpips_map.mean()}
        if self.output_transient and (mask==0).any():
            log['val_psnr_mask'] = psnr(results['rgb_fine'], rgbs, mask==0)
            log['val_ssim_mask'] = ssim_map[mask.view(H, W)==0].mean()
            log['val_lpips_mask'] = lpips_map[mask.view(H, W)==0].mean()

        return log

    def validation_epoch_end(self, outputs):
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()
        mean_ssim = torch.stack([x['val_ssim'] for x in outputs]).mean()
        mean_lpips = torch.stack([x['val_lpips'] for x in outputs]).mean()
        self.log('val/psnr', mean_psnr, prog_bar=True)
        self.log('val/ssim', mean_ssim)
        self.log('val/lpips', mean_lpips)
        if self.output_transient and all(['val_psnr_mask' in x for x in outputs]):
            mean_psnr_mask = torch.stack([x['val_psnr_mask'] for x in outputs]).mean()
            mean_ssim_mask = torch.stack([x['val_ssim_mask'] for x in outputs]).mean()
            mean_lpips_mask = torch.stack([x['val_lpips_mask'] for x in outputs]).mean()
            self.log('val/psnr_mask', mean_psnr_mask, prog_bar=True)
            self.log('val/ssim_mask', mean_ssim_mask)
            self.log('val/lpips_mask', mean_lpips_mask)


def main(hparams):
    system = NeRFSystem(hparams)
    ckpt_cb = ModelCheckpoint(dirpath=f'ckpts/{hparams.exp_name}', filename='{epoch:d}',
                              save_top_k=-1)

    logger = TestTubeLogger(save_dir="logs",
                            name=hparams.exp_name,
                            debug=False,
                            create_git_tag=False,
                            log_graph=False)

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      callbacks=[ckpt_cb],
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      weights_summary=None,
                      progress_bar_refresh_rate=hparams.refresh_every,
                      gpus=hparams.num_gpus,
                      num_nodes=hparams.num_nodes,
                      accelerator='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=0,
                      reload_dataloaders_every_epoch=True,
                      benchmark=True,
                      profiler="simple" if hparams.num_gpus==1 else None,
                      plugins=[DDPPlugin(find_unused_parameters=False)])

    trainer.fit(system)


def backup_files(args, files):
    """Save files for debugging."""
    backup_dir = os.path.join('files_backup', args.exp_name)
    os.makedirs(backup_dir, exist_ok=True)
    for f in files:
        os.system(f'cp {f} {backup_dir}')


if __name__ == '__main__':
    hparams = get_opts()
    if hparams.debug:
        backup_files(hparams, 
                     ['models/nerf.py', 'models/rendering.py', 'losses.py', 'train.py'])
    main(hparams)