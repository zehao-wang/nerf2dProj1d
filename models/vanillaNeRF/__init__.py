""" The entry of the pipeline of the model """

__all__ = ["VanillaNeRF"]

from .pe import get_embedder
from .model import NeRF
from .renderer import run_network, render
import numpy as np
import torch
import time
import torch.nn.functional as F
from tqdm import tqdm

class VanillaNeRF(object):
    def __init__(self, config, cam_config) -> None:
        self.config = config
        start = 0 # global step
        # Create embedder
        embed_fn, input_ch = get_embedder(input_dims=2, multires=config.multires, i=config.i_embed)
        embeddirs_fn, input_ch_views = get_embedder(input_dims=2, multires=config.multires_views, i=config.i_embed)

        # Create Model
        model = NeRF(D=config.netdepth, W=config.netwidth,
                    input_ch=input_ch, input_ch_views=input_ch_views).to(config.device)
        grad_vars = list(model.parameters())

        network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                        embed_fn=embed_fn,
                                                        embeddirs_fn=embeddirs_fn,
                                                        netchunk=config.netchunk)

        # Create optimizer
        self.optimizer = torch.optim.Adam(params=grad_vars, lr=config.lrate, betas=(0.9, 0.999))

        ####### construct returns ################
        self.render_kwargs_train = {
            "resolution": cam_config['resolution'],
            "fov": np.radians(cam_config["fov"]),
            "near": cam_config['near'],
            "far": cam_config['far'],
            'network_query_fn' : network_query_fn,
            'perturb' : config.perturb,
            'N_samples' : config.N_samples,
            'network_fn' : model,
            'white_bkgd' : config.white_bkgd,
            'raw_noise_std' : config.raw_noise_std,
        }

        self.render_kwargs_test = {k : self.render_kwargs_train[k] for k in self.render_kwargs_train}
        self.render_kwargs_test['perturb'] = False
        self.render_kwargs_test['raw_noise_std'] = 0.

    def train(self, rays):
        rgb, disp, acc, extras = render(rays=rays, chunk=self.config.chunk, **self.render_kwargs_train)
        return rgb
        
    def eval(self, render_poses):
        rgbs = []
        t = time.time()
        for i, cam_pose in enumerate(tqdm(render_poses)):
            print(i, time.time() - t)
            t = time.time()
            rgb, disp, acc, extras = render(cam_pos=cam_pose[:2], cam_dir=cam_pose[2], chunk=self.config.chunk, **self.render_kwargs_test)
            rgbs.append(rgb.cpu().numpy())
            if i==0:
                print(rgb.shape)

        rgbs = np.stack(rgbs, 0)
        return rgbs
    
    def inference_points(self, points, viewdir, resolution):
        """ Draw 2D space """
        viewdir = torch.tensor([viewdir], device=self.config.device).float()
        raw_test = self.render_kwargs_test['network_query_fn'](points, viewdir, self.render_kwargs_test['network_fn'])

        rgb_grid = torch.sigmoid(raw_test[...,:3]).squeeze(0)
        alpha = 1.-torch.exp(-F.relu(raw_test[...,3:]))
        rgb_grid = torch.cat([rgb_grid, alpha.squeeze(0)], dim=1)

        rgb_grid = rgb_grid.reshape([resolution[0], resolution[1], 4])
        rgb_grid = np.clip(rgb_grid.cpu().numpy(),0, 1)

        return rgb_grid 
        
