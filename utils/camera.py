"""
This file maintains camera class and its distribution

"""

import numpy as np  
from utils.camera_dist import shape_classes    

class DistributeCameras:
    def __init__(self, config):
        """
        Initialize Cameras
        :param config: distribution shape and other informations

        example:
        config = {"shape": "circle", "params":{"radius": 3, "center": [1,1], "random_init": False}}
        """

        self.dist = shape_classes[config["shape"]](**config["params"])

    def gen_cams(self, num_cams, fov=np.radians(60), resolution=100):
        positions, thetas = self.dist.sample_poses(num_cams)
        cams = []
        for position, theta in zip(positions, thetas):
            cams.append(PinholeCamera2D(position, theta, fov, resolution))
        return cams

class PinholeCamera2D:
    def __init__(self, position, theta, fov, resolution):
        """
        Initialize the 2D camera.

        :param position: Camera position as a 2D coordinate (x, y).
        :param theta: Orientation angle in radians (0 = facing right, π/2 = facing up).
        :param fov: Field of view in radians.
        :param resolution: Number of samples along the 1D projection (e.g., pixels).
        """
        self.position = np.array(position, dtype=float)
        self.theta = theta
        self.fov = fov
        self.resolution = resolution

        # Precompute ray angles within the FOV
        self.angles = np.linspace(-fov / 2, fov / 2, resolution)

    def generate_ray(self, index):
        """
        Generate a ray for a specific index along the 1D projection.

        :param index: Index of the sample (0 <= index < resolution).
        :return: Ray as a tuple (origin, direction), where:
                 - origin is the camera position (x, y).
                 - direction is a normalized 2D vector (dx, dy).
        """
        if index < 0 or index >= self.resolution:
            raise ValueError("Index out of range.")

        # Compute the angle of the ray relative to the camera's orientation
        angle = self.theta + self.angles[index]

        # Compute the ray direction
        direction = np.array([np.cos(angle), np.sin(angle)], dtype=float)

        # The ray originates from the camera's position
        origin = self.position

        return origin, direction
    
    def get_rays(self):
        rays_o = []
        rays_d = []
        for i in range(self.resolution):
            ray_o, ray_d = self.generate_ray(i)
            rays_o.append(ray_o)
            rays_d.append(ray_d)
        rays_o = np.stack(rays_o)
        rays_d = np.stack(rays_d)
        return rays_o, rays_d
        
    def generate_all_rays(self):
        """
        Generate rays for all samples in the 1D projection.

        :return: A list of (origin, direction) tuples for each sample.
        """
        rays = [self.generate_ray(i) for i in range(self.resolution)]
        return rays
    
    def sample_points_along_ray(self, origin: int, direction: np.array , n_samples: int, near: float, far: float):
        """
        NOTE: this function is used for verifying the directions, the code is copied to sampler
        Samples N points along a ray from near to far distances.

        :param n_samples: Number of points to sample along the ray.
        :param near: The nearest distance along the ray.
        :param far: The farthest distance along the ray.
        :return: A list of sampled points along the ray in world coordinates.
        """
        # origin, direction = self.generate_ray(pixel)
        distances = np.linspace(near, far, n_samples)  # Sample N distances between near and far
        points = [origin + t * direction for t in distances]  # Compute points along the ray
        return points


def testcase1():
    # Test single camera ray casting
    # Initialize the 2D camera
    position = [1, 4]  # Camera at the origin
    orientation = np.radians(120)
    fov = np.radians(30)  # Example focal length
    resolution = 20  # 20 pixels in the 1D image line

    camera = PinholeCamera2D(position, orientation, fov, resolution)

    # Generate a ray for the center pixel
    origin, direction = camera.generate_ray(resolution // 2)
    print("Center Pixel Ray Origin:", origin)
    print("Center Pixel Ray Direction:", direction)

    # Generate all rays
    all_rays = camera.generate_all_rays()
    print(f"Generated {len(all_rays)} rays.")
    
    import matplotlib.pyplot as plt
    n_samples = 100
    for ray_o, ray_d in all_rays:
        pts = camera.sample_points_along_ray(ray_o, ray_d, n_samples, 0.5, 10)
        pts = np.stack(pts)
        plt.scatter(pts[:,0], pts[:,1])
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig('./cache/test_ray_casting.png', dpi=300)
    plt.close()

def testcase2():
    # Test camera distribution
    config = {
        "shape": "circle",
        "params":{
            "radius": 5,
            "center": [3, 9],
            "random_init": False
        }
    }
    cam_dist = DistributeCameras(config)
    cams = cam_dist.gen_cams(10)

    import matplotlib.pyplot as plt
    for cam in cams:
        pos_vec = cam.position
        dir_vec = np.array([np.cos(cam.theta), np.sin(cam.theta)], dtype=float)
        plt.plot([pos_vec[0], pos_vec[0]+dir_vec[0]], [pos_vec[1], pos_vec[1]+dir_vec[1]], c='blue')
        plt.scatter([pos_vec[0]], [pos_vec[1]], c='green',marker='o')
    plt.gca().set_aspect('equal')
    plt.tight_layout()
    plt.savefig('./cache/test_cam_dist.png', dpi=300)
    plt.close()

# Example Usage: ray casting given camera 2d coord, orientation, resolution, field of view.
if __name__ == "__main__":
    testcase1()
    testcase2()






