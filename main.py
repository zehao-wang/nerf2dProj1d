import numpy as np
import time
import json
from dataset.create_dset import get_dset
import os
import torch
import matplotlib.pyplot as plt
from utils.math_utils import mse2psnr, img2mse, img2mse_torch, mse2psnr_torch
from models import init_model
from arg_parser import get_args
args = get_args()


def eval(poses, ref_data, i_test, model):
    """
    :param poses: [N, 2 + 1] position + orientation
    :param ref_data: [N, resolution, 3]
    :param i_test: index list, length <= N
    """
    print('\033[1;32m [INFO]\033[0m Rendering: ')

    testsavedir = os.path.join(args.basedir, args.expname)
    os.makedirs(testsavedir, exist_ok=True)
    
    results = {
        "PSNR": {}
    }
    with torch.no_grad():
        # Render test views
        render_poses = poses[i_test, :]

        gt_data = ref_data[i_test]
        rgbs = model.eval(render_poses)

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
        pts_list = grid_points.reshape([-1,2]).to(model.config.device).float()

        rgb_grids = []
        for i, viewdir in enumerate([[1,1], [1,0], [0,1], [-1,-1]]):
            rgb_grid = model.inference_points(pts_list[None, ...], viewdir, resolution=(grid_points.shape[0], grid_points.shape[1])) 
            rgb_grids.append(rgb_grid)

        for i in range(len(rgb_grids)):
            plt.imshow(rgb_grids[i])
            plt.savefig(os.path.join(testsavedir, f'rendered2d-{i}.png'))
            plt.close()
        
        print('Done rendering', testsavedir)
    return results

def train(args, data_dict, model, start=0):
    N_iters = args.training_iters
    print('Begin')
    print('TRAIN views are', data_dict['i_train'])
    print('TEST views are', data_dict['i_test'])

    print('\033[1;32m [INFO]\033[0m Training: ')

    # ====== batching rays =======
    rays_rgb = []
    for i, (rays_o, rays_d, gt_proj) in enumerate(zip(data_dict['rays'][0], data_dict['rays'][1], data_dict['ref_data'])):
        if i not in data_dict['i_train']:
            continue
        rays_rgb.append(np.concatenate([rays_o, rays_d, gt_proj], axis=1))
    rays_rgb = np.concatenate(rays_rgb, axis=0) # [NxM, 2+2+3] rayo, rayd, rgb
    np.random.shuffle(rays_rgb)     
    i_batch = 0
    rays_rgb = torch.from_numpy(rays_rgb).float().to(model.config.device)

    # ====== start training ======
    global_step = start

    start = start + 1
    # for i in trange(start, N_iters):
    for i in range(start, N_iters):
        time0 = time.time()

        # Random over all images
        batch = rays_rgb[i_batch:i_batch+model.config.N_rand] # [B, 2+2+3]
        batch_rays, target_s = batch[:, :4], batch[:, -3:]
        batch_rays = [batch_rays[:, :2], batch_rays[:, 2:]]

        i_batch += model.config.N_rand
        if i_batch >= rays_rgb.shape[0]:
            # print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0

        #####  Core optimization loop  #####
        rgb = model.train(rays=batch_rays)

        model.optimizer.zero_grad()
        img_loss = img2mse_torch(rgb, target_s)
        loss = img_loss
        psnr = mse2psnr_torch(img_loss)
        if global_step % args.print_every == 0:
            print("Loss: ", loss.item(), "PSNR: ", psnr.item())

        loss.backward()
        model.optimizer.step()

        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = model.config.lrate_decay * 1000
        new_lrate = model.config.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in model.optimizer.param_groups:
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

    model = init_model(args.model_name, cam_config=config["cameras_config"])

    train(args, data_dict, model)
    results = eval(data_dict['poses'], data_dict['ref_data'], data_dict['i_test'], model)
    print(results)
