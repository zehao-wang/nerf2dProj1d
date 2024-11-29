from dataset.base import BaseDataset
import os 
import numpy as np
import torch
from utils.camera import DistributeCameras
from utils.math_utils import line2d_intersection
import random
import matplotlib.pyplot as plt
from joblib import dump, load

def is_debug_mode():
    debug_value = os.getenv('DEBUG', '0') 
    return debug_value.lower() in ['1', 'true', 'yes', 'on']

class TriangleDset(BaseDataset):
    def __init__(self, params, full_config):
        super().__init__(params, full_config) 
    
    def construct_shape(self):
        self.vertices = np.array(self.params["vertices"], dtype=float) # [3, 2]
        self.colors = np.array(self.params["colors"], dtype=float) # [n_colors, 3]
        self.line_segments = [
            [self.vertices[0], self.vertices[1]],
            [self.vertices[1], self.vertices[2]],
            [self.vertices[2], self.vertices[0]],
        ]
        if is_debug_mode():
            import matplotlib.pyplot as plt
            for i, line_seg in enumerate(self.line_segments):
                plt.plot([line_seg[0][0], line_seg[1][0]], [line_seg[0][1], line_seg[1][1]],
                         color=self.colors[i])
                
            plt.gca().set_aspect('equal')
            plt.tight_layout()
            plt.savefig('./dataset/cache/triangle.png')

    def construct_poses(self):
        cam_config = self.full_config["cameras_config"]

        cam_dist = DistributeCameras(cam_config)
        cams = cam_dist.gen_cams(
            self.full_config["num_cams"], fov=np.radians(cam_config["fov"]), 
            resolution=cam_config["resolution"]
        )

        cam_poses = []
        cam_dirs = []
        rays_o_list = []
        rays_d_list = []
        for cam in cams:
            cam_poses.append(cam.position)
            # cam_dirs.append(np.array([np.cos(cam.theta), np.sin(cam.theta)], dtype=float))
            cam_dirs.append(cam.theta)
            rays_o, rays_d = cam.get_rays()
            rays_o_list.append(rays_o)
            rays_d_list.append(rays_d)

        self.cams = cams
        self.poses = np.hstack([np.stack(cam_poses), np.array(cam_dirs)[..., None]]) # position: index 0,1; orientation: index 2
        self.rays = [rays_o_list, rays_d_list]

    def gen(self, cache_dir='./dataset/cache/dataset_cache.joblib'):
        if os.path.exists(cache_dir):
            try:
                data_dict = load(cache_dir)
                print(f'\033[1;32m [INFO]\033[0m Successfully load cache from {cache_dir}')
                return data_dict
            except:
                print('\033[1;31m [WARNING]\033[0m joblib file broken, regenerate')
        
        self.construct_shape()
        self.construct_poses()

        # symbols: 
        #   N -> camera number
        #   M -> resolution
        indices = [i for i in range(len(self.cams))]
        random.shuffle(indices)
        num_test = int(len(indices) * self.full_config["test_ratio"])
        data_dict = {
            "poses": self.poses, # [N, 3] 2d position + theta
            "ref_data": [], # [N, M, 3]
            "i_train": indices[num_test:],
            "i_test": indices[:num_test],
            "rays": self.rays    # 2 x [N, M, 2], rays_o list and rays_d list
        }

        # Create Projected Ground Truth 1D img
        for i, (rays_o, rays_d) in enumerate(zip(self.rays[0], self.rays[1])): # N camera
            distances_list = []
            for line_start, line_end in self.line_segments:
                distances, _, _ = line2d_intersection(rays_o, rays_d, line_start, line_end)
                distances_list.append(distances)

            dist_matrix = np.stack(distances_list).T # [resolution, num_segments]
            
            indices = np.argmin(dist_matrix, axis=1)
            value_array = dist_matrix[np.arange(len(indices)), indices]
            bk_mask = np.isinf(value_array) # mask for background color

            # use white for background
            gt_proj = np.ones((self.full_config["cameras_config"]["resolution"], 3)) # [resolution, 3]
            gt_proj[~bk_mask] = self.colors[indices[~bk_mask]]

            data_dict["ref_data"].append(gt_proj)

            if is_debug_mode():
                os.makedirs('./dataset/cache/triangle', exist_ok=True)
                filename1 = os.path.join('./dataset/cache/triangle', '{:03d}-proj.png'.format(i))
                filename2 = os.path.join('./dataset/cache/triangle', '{:03d}-cam.png'.format(i))

                plt.matshow(gt_proj[:, np.newaxis, :], aspect="auto")
                # Add grid lines
                # plt.gca().set_xticks(np.arange(-0.5, 1.5, 1), minor=True)
                plt.gca().set_yticks(np.arange(-0.5, gt_proj.shape[0]+0.5, 1), minor=True)
                plt.gca().grid(which="minor", color="w", linestyle='-', linewidth=0.5)
                plt.tick_params(which="minor", size=0)  # Hide minor ticks
                plt.gca().xaxis.set_visible(False)
                plt.gca().set_aspect('equal')
                plt.savefig(filename1)
                plt.close()

                n_samples = 100
                for i, line_seg in enumerate(self.line_segments):
                    plt.plot([line_seg[0][0], line_seg[1][0]], [line_seg[0][1], line_seg[1][1]],
                            color=self.colors[i])

                for ray_o, ray_d in zip(rays_o, rays_d):
                    pts = self.cams[i].sample_points_along_ray(ray_o, ray_d, n_samples, 0., 4.)
                    pts = np.stack(pts)
                    plt.scatter(pts[:,0], pts[:,1], s=0.1, c='gray')
                plt.xlim(-2,2)
                plt.ylim(-2,2) 
                plt.gca().set_aspect('equal')
                plt.tight_layout()
                plt.savefig(filename2)
                plt.close()

        data_dict["ref_data"] = np.stack(data_dict["ref_data"])
        dump(data_dict, cache_dir)
        return data_dict

