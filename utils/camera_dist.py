import numpy as np


class CircleDist:
    def __init__(self, radius, center, random_init) -> None:
        self.radius = radius
        self.center = center
        self.random_init = random_init

    def sample_poses(self, num_cams):
        angles = np.linspace(0, 2 * np.pi, num_cams, endpoint=False)
        if self.random_init:
            angles += np.random.random()

        x = self.radius * np.cos(angles)
        y = self.radius * np.sin(angles)
        points = np.column_stack((x, y))
        
        thetas = []
        for point in points:
            vec_ref = np.array([1,0])
            dot_product = np.dot(-point, vec_ref)
            magnitude_v1 = np.linalg.norm(point)
            magnitude_v2 = np.linalg.norm(vec_ref)
            angle_rad = np.arccos(np.clip(dot_product / (magnitude_v1 * magnitude_v2), -1.0, 1.0))

            # Use the cross product to determine the direction (clockwise or counterclockwise)
            cross_product = np.cross(vec_ref, -point)
            if cross_product < 0:
                # Adjust angle for clockwise direction
                angle_rad = 2 * np.pi - angle_rad
            thetas.append(angle_rad)
        
        points = points + np.array(self.center)
        return points, thetas

shape_classes = {
    "circle": CircleDist,
    # "square": Square,
    # "triangle": Triangle,
}