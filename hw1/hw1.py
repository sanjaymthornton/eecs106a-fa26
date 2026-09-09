"""HW1: transform the four vehicle corners into the world frame."""

import numpy as np


def get_corners(xy, theta, corner1, corner2, corner3, corner4):
    """Return the four world-frame corner positions in the same order.

    Args:
        xy: Vehicle position as a NumPy array of shape (2, 1).
        theta: Vehicle heading in radians, measured counterclockwise.
        corner1, corner2, corner3, corner4: Vehicle-frame corner positions,
            each with shape (2, 1).

    Returns:
        A tuple of four NumPy arrays, each with shape (2, 1).
    """
    print(xy[0])
    print(theta)
    print(corner1)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    corner1 = xy + (rotation @ corner1)
    corner2 = xy + (rotation @ corner2)
    corner3 = xy + (rotation @ corner3)
    corner4 = xy + (rotation @ corner4)
    return (corner1, corner2, corner3, corner4)
    raise NotImplementedError("Implement get_corners before running the checks.")
