"""Student functions for HW6: trajectories and integration in SE(3)."""

import numpy as np

from hw2.kin_func_skeleton import inverse_SE3, renormalize_SE3, se3_to_twist, twist_to_SE3


L1 = 0.57
L2 = 0.25
L3 = 0.20
L4 = 0.10
K = 0.5

def trajectory(t):
    """Return the chair-swing pose defined in the handout.

    Args:
        t: Scalar time in seconds, t >= 0.

    Returns:
        NumPy array with shape (4, 4), representing g_ab(t) in the ride
        frame.
    """
    phi = K * np.sin(t)
    g = np.eye(4)
    R = np.zeros((3, 3))
    R[:, 0] = [-np.cos(t), -np.sin(t), 0]
    R[:, 1] = [np.cos(phi) * np.sin(t), -np.cos(phi) * np.cos(t), -np.sin(phi)]
    R[:, 2] = [np.sin(phi) * np.sin(t), -np.sin(phi) * np.cos(t), np.cos(phi)]
    p_x = (-L2 * np.sin(t)) - (L3 * np.sin(t) * np.sin(phi))
    p_y = (L2 * np.cos(t)) + (L3 * np.cos(t) * np.sin(phi))
    p_z = (L1 + L4 - L4 * np.cos(t)) - (L3 * np.cos(phi))
    p = np.array([p_x, p_y, p_z])
    g[:3, :3] = R
    g[:3, 3] = p
    return g

g = trajectory(0.7)

R = g[:3, :3]
p = g[:3, 3]

print("R.T @ R:\n", R.T @ R)
print("det(R):", np.linalg.det(R))
print("p:", p)

def body_velocity(t):
    """Calculate the instantaneous body velocity of the chair swing.

    Args:
        t: Scalar time in seconds, t >= 0.

    Returns:
        NumPy array with shape (4, 4), representing the body velocity
        matrix in se(3). Analytically differentiate the full trajectory
        matrix, including its rotation, and compute the body velocity.

    inverse_SE3 from kin_func_skeleton inverts the pose.
    """
    d_g = np.eye(4)
    phi = K * np.sin(t)
    d_phi = K * np.cos(t)
    d_R = np.array([
        [np.sin(t), np.cos(t) * np.cos(phi) - d_phi * np.sin(t) * np.sin(phi), np.cos(t) * np.sin(phi) + d_phi * np.sin(t) * np.cos(phi)],
        [-np.cos(t), np.sin(t) * np.cos(phi) + d_phi * np.cos(t) * np.sin(phi), np.sin(t) * np.sin(phi) - d_phi * np.cos(t) * np.cos(phi)],
        [0, -d_phi * np.cos(phi), -d_phi * np.sin(phi)]
        ])
    d_p_x = (-L2 * np.cos(t)) + (-L3 * np.cos(t) * np.sin(phi) - L3 * np.sin(t) * np.cos(phi) * d_phi)
    d_p_y = (-L2 * np.sin(t)) + (-L3 * np.sin(t) * np.sin(phi) + L3 * np.cos(t) * np.cos(phi) * d_phi)
    d_p_z = (L4 * np.sin(t)) + (L3 * np.sin(phi) * d_phi)
    d_p = np.array([d_p_x, d_p_y, d_p_z])
    d_g[:3, :3] = d_R
    d_g[:3, 3] = d_p
    d_g[3, 3] = 0
    g = trajectory(t)
    return inverse_SE3(g) @ d_g


def euler_step(pose, velocity_hat, dt):
    """Advance an SE(3) pose with body velocity held constant for one step.

    Args:
        pose: NumPy array with shape (4, 4), the current pose.
        velocity_hat: NumPy array with shape (4, 4), a body velocity in se(3).
        dt: Nonnegative scalar timestep in seconds.

    Returns:
        NumPy array with shape (4, 4), the next SE(3) pose. Apply the
        exponential rigid-motion increment on the right of the current
        pose, as in Problem 4.

    se3_to_twist converts the velocity matrix to a vector. twist_to_SE3
    computes the rigid-motion increment over dt. Apply renormalize_SE3
    to the updated pose to correct floating-point drift in its rotation.
    """
    # TODO
    return ...
