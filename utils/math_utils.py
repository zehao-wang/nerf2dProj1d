import numpy as np
import torch

img2mse = lambda x, y : np.mean((x - y) ** 2)
mse2psnr = lambda x : -10. * np.log(x) / np.log(np.array([10.]))

img2mse_torch = lambda x, y : torch.mean((x - y) ** 2)
mse2psnr_torch = lambda x : -10. * torch.log(x) / torch.log(torch.Tensor([10.]).cuda())

def line2d_intersection(rays_o, rays_d, seg_start, seg_end):
    """
    Compute intersections between N rays and a single line segment in 2D.

    Parameters:
    - rays_o: (N, 2) array of ray origins.
    - rays_d: (N, 2) array of ray directions.
    - seg_start: (2,) array representing the start point of the segment.
    - seg_end: (2,) array representing the end point of the segment.

    Returns:
    - distances: (N,) array of Euclidean distances from ray origins to intersection points.
                 If no intersection, the value is np.inf.
    - intersection_points: (N, 2) array of intersection points.
                           Non-intersecting rays have np.nan entries.
    - intersects: (N,) boolean array indicating which rays intersect the segment.
    """
    # Segment direction vector
    seg_dir = seg_end - seg_start  # Shape: (2,)

    # Vectors from ray origins to segment start
    rays_to_seg = seg_start - rays_o  # Shape: (N, 2)

    # Compute the denominator (cross product of ray directions and segment direction)
    denom = np.cross(rays_d, seg_dir)  # Shape: (N,)

    # Initialize arrays for parameters t and u
    t = np.full(rays_o.shape[0], np.inf)
    u = np.full(rays_o.shape[0], np.inf)

    # Mask for rays that are not parallel to the segment
    non_parallel = denom != 0

    # Compute t and u where denom is not zero (non-parallel rays)
    denom_non_zero = denom[non_parallel]
    rays_d_non_parallel = rays_d[non_parallel]
    rays_to_seg_non_parallel = rays_to_seg[non_parallel]

    t_temp = np.cross(rays_to_seg_non_parallel, seg_dir) / denom_non_zero
    u_temp = np.cross(rays_to_seg_non_parallel, rays_d_non_parallel) / denom_non_zero

    # Update t and u values where computation is valid
    t[non_parallel] = t_temp
    u[non_parallel] = u_temp

    # Intersection occurs when t >= 0 and 0 <= u <= 1
    intersects = (t >= 0) & (u >= 0) & (u <= 1)

    # Initialize intersection points with NaNs
    intersection_points = np.full_like(rays_o, np.nan)

    # Compute intersection points for intersecting rays
    intersecting_rays_o = rays_o[intersects]
    intersecting_rays_d = rays_d[intersects]
    t_intersects = t[intersects][:, np.newaxis]

    intersection_points[intersects] = intersecting_rays_o + intersecting_rays_d * t_intersects

    # Compute Euclidean distances from ray origins to intersection points
    distances = np.full(rays_o.shape[0], np.inf)
    distances[intersects] = np.linalg.norm(intersection_points[intersects] - intersecting_rays_o, axis=1)

    return distances, intersection_points, intersects

if __name__ == '__main__':
    # Define rays
    rays_o = np.array([[0, 0], [1, 1], [-1, -1], [3, 3]])
    rays_d = np.array([[1, 0], [0, -1], [0, 1], [-1, -1]])


    # Define line segment
    line_start = np.array([2, 0])
    line_end = np.array([2, 3])

    # Compute intersections
    distances, _, _ = line2d_intersection(rays_o, rays_d, line_start, line_end)
    print("Distances to intersection:", distances)

