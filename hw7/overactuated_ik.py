"""Student functions for HW7: Jacobian IK and Panda null-space IK."""

import numpy as np

from jacobian_func_skeleton import spatial_jacobian
from panda_model import (
    PANDA_G_ST0,
    PANDA_THETA_MAX,
    PANDA_THETA_MIN,
    PANDA_TWISTS,
)


def jacobian_ik_step(
    theta, velocity, step_size=0.2, twists=PANDA_TWISTS, g_st0=PANDA_G_ST0,
):
    """PROVIDED: apply one spatial-Jacobian inverse-kinematics update.

    Args:
        theta: NumPy array with shape ``(n,)`` containing current joint
            angles in radians. The UR7e has six joints; the Panda has seven.
        velocity: NumPy array with shape ``(6,)`` containing the desired
            spatial end-effector velocity.
        step_size: Scalar multiplier applied to the joint update.
        twists: NumPy array with shape ``(6, n)`` of space-frame joint
            twists.
        g_st0: NumPy array with shape ``(4, 4)`` for the zero-configuration
            tool pose.

    This calls your ``spatial_jacobian`` from P1. g_st0 is retained for the
    common robot interface; the spatial Jacobian does not depend on it.

    Returns:
        NumPy array with shape ``(n,)`` containing the updated joint angles.
    """

    jacobian = spatial_jacobian(twists, theta)
    return theta + step_size * np.linalg.pinv(jacobian) @ velocity


def null_space_ik_step(
    theta,
    velocity,
    secondary_gradient,
    step_size=0.2,
    secondary_gain=0.05,
):
    """Apply one IK update with a projected secondary objective.

    Args:
        theta: NumPy array with shape ``(7,)`` containing the current Panda
            joint angles in radians.
        velocity: NumPy array with shape ``(6,)`` containing the desired
            spatial end-effector velocity.
        secondary_gradient: NumPy array with shape ``(7,)`` containing a
            joint-space direction that increases the secondary objective.
        step_size: Scalar multiplier applied to the complete joint update.
        secondary_gain: Scalar multiplier applied to the null-space term.

    Returns:
        NumPy array with shape ``(7,)`` containing the updated joint angles.
    """

    # TODO
    return ...


def joint_limit_gradient(theta):
    """Compute the Panda joint-limit avoidance direction.

    Args:
        theta: NumPy array with shape ``(7,)`` containing the current Panda
            joint angles in radians.

    Returns:
        NumPy array with shape ``(7,)`` containing the gradient of our
        joint-limit avoidance direction.
    """

    # TODO
    return ...
