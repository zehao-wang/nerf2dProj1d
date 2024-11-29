from utils.models import NeRF
from utils.pe import get_embedder
from utils.renderer import render
import numpy as np
import time
import json
from dataset.create_dset import get_dset
import os
import torch
import matplotlib.pyplot as plt
from utils.math_utils import mse2psnr, img2mse, img2mse_torch, mse2psnr_torch
from tqdm import tqdm, trange
import torch.nn.functional as F
from arg_parser import get_args
args = get_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def batchify(fn, chunk):
    """Constructs a version of 'fn' that applies to smaller batches.
    """
    if chunk is None:
        return fn
    def ret(inputs):
        return torch.cat([fn(inputs[i:i+chunk]) for i in range(0, inputs.shape[0], chunk)], 0)
    return ret

def run_network(inputs, viewdirs, fn, embed_fn, embeddirs_fn, netchunk=1024*64):
    """Prepares inputs and applies network 'fn'.
    """
    inputs_flat = torch.reshape(inputs, [-1, inputs.shape[-1]])
    embedded = embed_fn(inputs_flat)

    if viewdirs is not None:
        input_dirs = viewdirs[:,None].expand(inputs.shape)
        input_dirs_flat = torch.reshape(input_dirs, [-1, input_dirs.shape[-1]])
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat([embedded, embedded_dirs], -1)

    outputs_flat = batchify(fn, netchunk)(embedded)
    outputs = torch.reshape(outputs_flat, list(inputs.shape[:-1]) + [outputs_flat.shape[-1]])
    return outputs

def create_nerf(args, cam_config):
    start = 0 # global step
    # Create embedder
    embed_fn, input_ch = get_embedder(input_dims=2, multires=args.multires, i=args.i_embed)
    embeddirs_fn, input_ch_views = get_embedder(input_dims=2, multires=args.multires_views, i=args.i_embed)

    # Create Model
    model = NeRF(D=args.netdepth, W=args.netwidth,
                 input_ch=input_ch, input_ch_views=input_ch_views).to(device)
    grad_vars = list(model.parameters())

    network_query_fn = lambda inputs, viewdirs, network_fn : run_network(inputs, viewdirs, network_fn,
                                                    embed_fn=embed_fn,
                                                    embeddirs_fn=embeddirs_fn,
                                                    netchunk=args.netchunk)

    # Create optimizer
    optimizer = torch.optim.Adam(params=grad_vars, lr=args.lrate, betas=(0.9, 0.999))

    ####### construct returns ################
    render_kwargs_train = {
        "resolution": cam_config['resolution'],
        "fov": np.radians(cam_config["fov"]),
        "near": cam_config['near'],
        "far": cam_config['far'],
        'network_query_fn' : network_query_fn,
        'perturb' : args.perturb,
        'N_samples' : args.N_samples,
        'network_fn' : model,
        'white_bkgd' : args.white_bkgd,
        'raw_noise_std' : args.raw_noise_std,
    }

    render_kwargs_test = {k : render_kwargs_train[k] for k in render_kwargs_train}
    render_kwargs_test['perturb'] = False
    render_kwargs_test['raw_noise_std'] = 0.
    return render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer

def render_path(render_poses, chunk, render_kwargs):
    rgbs = []
    disps = []
    t = time.time()
    for i, cam_pose in enumerate(tqdm(render_poses)):
        print(i, time.time() - t)
        t = time.time()
        rgb, disp, acc, _ = render(cam_pos=cam_pose[:2], cam_dir=cam_pose[2], chunk=chunk, **render_kwargs)
        rgbs.append(rgb.cpu().numpy())
        disps.append(disp.cpu().numpy())
        if i==0:
            print(rgb.shape, disp.shape)

    rgbs = np.stack(rgbs, 0)
    disps = np.stack(disps, 0)

    return rgbs, disps

def eval(poses, ref_data, i_test, render_kwargs, data_dict=None):
    """
    :param poses: [N, 2 + 1] position + orientation
    :param ref_data: [N, resolution, 3]
    :param i_test: index list, length <= N
    """
    print('\033[1;32m [INFO]\033[0m Rendering: ')
    print(render_kwargs)
    print('=' * 20)

    testsavedir = os.path.join(args.basedir, args.expname)
    os.makedirs(testsavedir, exist_ok=True)
    
    results = {
        "PSNR": {}
    }
    with torch.no_grad():
        # Render test views
        render_poses = poses[i_test, :]

        gt_data = ref_data[i_test]
        rgbs, _ = render_path(render_poses, args.chunk, render_kwargs=render_kwargs)

        if testsavedir is not None:
            for i, (rgb, gt_rgb, id) in enumerate(zip(rgbs, gt_data, i_test)):
                
                img_loss = img2mse(rgb[:,  :],  gt_rgb[:,  :])
                psnr = mse2psnr(img_loss)
                results["PSNR"][id] = psnr[0]

                filename = os.path.join(testsavedir, '{:03d}.png'.format(i))
                diff = np.abs(rgb[:, np.newaxis, :] - gt_rgb[:, np.newaxis, :])
                heatmap = diff.sum(axis=2) 
                normalized_heatmap = (heatmap - 0) / (np.max(heatmap) - 0)
                cmap = plt.cm.viridis
                diff_image = cmap(normalized_heatmap)[:, :, :3]

                out_img = np.concatenate([rgb[:, np.newaxis, :], gt_rgb[:, np.newaxis, :], diff_image], axis=1)
                out_img = np.clip(out_img, 0, 1)
                plt.matshow(out_img, aspect="auto")
                # plt.gca().set_xticks(np.arange(-0.5, 1.5, 1), minor=True)
                plt.gca().set_yticks(np.arange(-0.5, rgb.shape[0]+0.5, 1), minor=True)
                plt.gca().grid(which="minor", color="w", linestyle='-', linewidth=0.5)
                plt.tick_params(which="minor", size=0)  # Hide minor ticks
                plt.gca().xaxis.set_visible(False)
                plt.gca().set_aspect('equal')
                plt.savefig(filename)
                plt.close()

        #  Render 2D shape
        #  -- BOUND is set to [-2, 2]
        x_range = -torch.linspace(-2, 2, steps=100)  # 100 points from 2 to -2
        y_range = torch.linspace(-2, 2, steps=100)  # 100 points from -2 to 2

        # # Create a meshgrid
        xx, yy = torch.meshgrid(x_range, y_range, indexing="ij")
        grid_points = torch.stack((yy, xx), dim=-1)
        pts_list = grid_points.reshape([-1,2]).to(device).float()

        for i, viewdir in enumerate([[1,1], [1,0], [0,1], [-1,-1]]):
            viewdir = torch.tensor([viewdir], device=device).float()
            raw_test = render_kwargs['network_query_fn'](
                pts_list[None, ...], viewdir, render_kwargs['network_fn']
            )

            rgb_grid = torch.sigmoid(raw_test[...,:3]).squeeze(0)
            alpha = 1.-torch.exp(-F.relu(raw_test[...,3:]))
            rgb_grid = torch.cat([rgb_grid, alpha.squeeze(0)], dim=1)

            rgb_grid = rgb_grid.reshape([grid_points.shape[0], grid_points.shape[1], 4])
            rgb_grid = np.clip(rgb_grid.cpu().numpy(),0, 1)

            plt.imshow(rgb_grid)
            plt.savefig(os.path.join(testsavedir, f'rendered2d-{i}.png'))
            plt.close()
        
        print('Done rendering', testsavedir)
    return results

def train(data_dict, render_kwargs_train, start=0):
    N_iters = 1000 + 1
    print('Begin')
    print('TRAIN views are', data_dict['i_train'])
    print('TEST views are', data_dict['i_test'])

    print('\033[1;32m [INFO]\033[0m Training: ')
    print(render_kwargs_train)
    print('=' * 20)

    # ====== batching rays =======
    rays_rgb = []
    for i, (rays_o, rays_d, gt_proj) in enumerate(zip(data_dict['rays'][0], data_dict['rays'][1], data_dict['ref_data'])):
        if i not in data_dict['i_train']:
            continue
        rays_rgb.append(np.concatenate([rays_o, rays_d, gt_proj], axis=1))
    rays_rgb = np.concatenate(rays_rgb, axis=0) # [NxM, 2+2+3] rayo, rayd, rgb
    np.random.shuffle(rays_rgb)     
    i_batch = 0
    rays_rgb = torch.from_numpy(rays_rgb).float().to(device)

    # ====== start training ======
    global_step = start

    start = start + 1
    # for i in trange(start, N_iters):
    for i in range(start, N_iters):
        time0 = time.time()

        # Random over all images
        batch = rays_rgb[i_batch:i_batch+args.N_rand] # [B, 2+2+3]
        batch_rays, target_s = batch[:, :4], batch[:, -3:]
        batch_rays = [batch_rays[:, :2], batch_rays[:, 2:]]

        i_batch += args.N_rand
        if i_batch >= rays_rgb.shape[0]:
            # print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0

        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(rays=batch_rays, chunk=args.chunk, **render_kwargs_train)

        optimizer.zero_grad()
        img_loss = img2mse_torch(rgb, target_s)
        trans = extras['raw'][...,-1]
        loss = img_loss
        psnr = mse2psnr_torch(img_loss)
        if global_step % 200 == 0:
            print("Loss: ", loss.item(), "PSNR: ", psnr.item())

        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)

        loss.backward()
        optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        ################################

        dt = time.time()-time0
        # print(f"Step: {global_step}, Loss: {loss}, Time: {dt}")
        #####           end            #####

        global_step += 1


if __name__ == '__main__':
    # main()
    config = json.load(open(args.config))
    dset = get_dset(config)
    data_dict = dset.gen()
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer = create_nerf(args, config['cameras_config'])
    
    train(data_dict, render_kwargs_train)
    # results = eval(data_dict['poses'], data_dict['ref_data'], data_dict['i_test'], render_kwargs_test, data_dict=data_dict)
    results = eval(data_dict['poses'], data_dict['ref_data'], data_dict['i_test'], render_kwargs_test)
    print(results)
