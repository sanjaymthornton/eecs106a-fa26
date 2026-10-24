"""Provided Franka-Emika Panda model and helpers for HW7.

The kinematic constants match ``roboticstoolbox.models.URDF.Panda`` with
its default gripper tool tip as the end-effector frame.

Sources:
https://github.com/petercorke/robotics-toolbox-python/blob/main/roboticstoolbox/models/URDF/Panda.py
https://github.com/frankarobotics/franka_description
"""

import numpy as np

from hw2.kin_func_skeleton import (
    R3_to_so3,
    SE3_to_se3,
    forward_kinematics,
    inverse_SE3,
    se3_to_twist,
    twist_to_SE3,
)


# Space-frame joint twists at the zero configuration. Twists use the course
# ordering [linear velocity, angular velocity].
PANDA_TWISTS = np.array(
    [
        [0.0, -0.333, 0.0, 0.649, 0.0, 1.033, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.088],
        [0.0, 0.0, 0.0, -0.0825, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
        [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0],
    ]
)

# End-effector pose at the zero joint configuration.
PANDA_G_ST0 = np.array(
    [
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.088],
        [np.sqrt(0.5), -np.sqrt(0.5), 0.0, 0.0],
        [0.0, 0.0, -1.0, 0.8226],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

PANDA_THETA_MIN = np.array(
    [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
)
PANDA_THETA_MAX = np.array(
    [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]
)

PANDA_READY = np.array([0.0, -0.3, 0.0, -2.2, 0.0, 2.0, np.pi / 4])

POSITION_TOLERANCE = 1e-5
ORIENTATION_TOLERANCE = 1e-4


def load_panda():
    """Load the provided Robotics Toolbox Panda model.

    Returns:
        A Robotics Toolbox Panda model whose seven joint values correspond to
        the columns of ``PANDA_TWISTS``.
    """
    try:
        import roboticstoolbox as rtb
    except ImportError as error:
        raise ImportError(
            "Install the HW7 requirements before loading the Panda model."
        ) from error

    return rtb.models.URDF.Panda()


def panda_fk(theta):
    """Compute the Panda end-effector pose with product of exponentials.

    Args:
        theta: NumPy array with shape ``(7,)`` containing the Panda joint
            angles in radians.

    Returns:
        NumPy array with shape ``(4, 4)`` containing the pose of
            the gripper tool tip in the robot base frame.
    """
    return forward_kinematics(PANDA_TWISTS, theta) @ PANDA_G_ST0


def pose_error(actual_end_effector_pose, desired_end_effector_pose):
    """Measure the position and orientation difference between two poses.

    Args:
        actual_end_effector_pose: NumPy array with shape ``(4, 4)`` for the
            achieved pose.
        desired_end_effector_pose: NumPy array with shape ``(4, 4)`` for the
            desired pose.

    Returns:
        ``(position_error, orientation_error)``. Position error is the
        Euclidean distance between positions in meters. Orientation error is
        the smallest rotation angle between orientations in radians.
    """
    position_error = np.linalg.norm(
        actual_end_effector_pose[:3, 3]
        - desired_end_effector_pose[:3, 3]
    )
    relative_rotation = (
        actual_end_effector_pose[:3, :3].T
        @ desired_end_effector_pose[:3, :3]
    )
    cosine = (np.trace(relative_rotation) - 1.0) / 2.0
    orientation_error = np.arccos(np.clip(cosine, -1.0, 1.0))
    return position_error, orientation_error


def body_velocity_command(
    actual_end_effector_pose,
    desired_end_effector_pose,
    gain=1.0,
):
    """Compute a body-frame velocity command toward a desired pose.

    Args:
        actual_end_effector_pose: NumPy array with shape ``(4, 4)`` for the
            current Panda end-effector pose.
        desired_end_effector_pose: NumPy array with shape ``(4, 4)`` for the
            desired Panda end-effector pose.
        gain: Scalar multiplier applied to the velocity command.

    Returns:
        NumPy array with shape ``(6,)`` containing body-frame linear and
        angular velocity.
    """
    relative_pose = (
        inverse_SE3(actual_end_effector_pose)
        @ desired_end_effector_pose
    )
    velocity = se3_to_twist(SE3_to_se3(relative_pose))
    return gain * np.asarray(np.real_if_close(velocity), dtype=float)


def spatial_velocity_command(actual_end_effector_pose, desired_end_effector_pose, gain=1.0):
    """Return a spatial velocity command toward the desired tool pose.

    Both poses are NumPy arrays with shape (4, 4) in the same reference
    frame. Return a NumPy array with shape (6,), using the P1 spatial error
    functions and scalar gain.
    """
    from jacobian_func_skeleton import spatial_configuration_error, spatial_error_twist

    error = spatial_configuration_error(actual_end_effector_pose, desired_end_effector_pose)
    return gain * np.asarray(np.real_if_close(spatial_error_twist(error)), dtype=float)


def manipulability(theta):
    """Calculate the Panda's Yoshikawa manipulability measure.

    Args:
        theta: NumPy array with shape ``(7,)`` containing the Panda joint
            angles in radians.

    Returns:
        A scalar equal to the product of the six singular values of the spatial
        Jacobian.
    """
    from jacobian_func_skeleton import spatial_jacobian

    jacobian = spatial_jacobian(PANDA_TWISTS, theta)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    return np.prod(singular_values)


def manipulability_gradient(theta):
    """Calculate Panda manipulability's analytic gradient.

    Args:
        theta: NumPy array with shape ``(7,)`` containing the Panda joint
            angles in radians.

    Returns:
        NumPy array with shape ``(7,)`` containing the manipulability
        gradient with respect to the seven joint angles. For a full-row-rank
        spatial Jacobian J, component i is m * trace(pinv(J) @ dJ/dtheta_i).
        At numerical rank deficiency, return zero: the singular-value product
        need not have a unique gradient there. No joint perturbations are used.

    Jacobian derivatives use the twist Lie bracket.
    For column j > i, dJ[:, j]/dtheta_i = ad(J[:, i]) @ J[:, j];
    earlier columns and column i have zero derivative.

    https://arxiv.org/abs/2010.08696
    """
    from jacobian_func_skeleton import spatial_jacobian

    jacobian = spatial_jacobian(PANDA_TWISTS, theta)
    singular_values = np.linalg.svd(jacobian, compute_uv=False)
    gradient = np.zeros(7)
    if np.linalg.matrix_rank(jacobian) < 6:
        return gradient
    measure = np.prod(singular_values)
    pseudoinverse = np.linalg.pinv(jacobian)
    for index in range(7):
        bracket = np.zeros((6, 6))
        bracket[:3, :3] = R3_to_so3(jacobian[3:, index])
        bracket[:3, 3:] = R3_to_so3(jacobian[:3, index])
        bracket[3:, 3:] = bracket[:3, :3]
        derivative = np.zeros_like(jacobian)
        for column in range(index + 1, 7):
            derivative[:, column] = bracket @ jacobian[:, column]
        gradient[index] = measure * np.trace(pseudoinverse @ derivative)
    return gradient
