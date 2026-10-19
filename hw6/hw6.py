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
    # TODO
    return ...


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
    # TODO
    return ...


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
