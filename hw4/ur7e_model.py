"""Provided UR7e model and helpers for HW4.

Sources:
https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
https://github.com/UniversalRobots/Universal_Robots_ROS2_Description/blob/rolling/config/ur7e/default_kinematics.yaml

Students do not need to modify this file.
"""

import numpy as np
from eaik.IK_DH import DhRobot


"""
Denavit-Hartenberg (DH) parameters are another way to describe a robot arm.
Instead of describing each joint with a twist, DH parameters describe the
position and orientation of one link frame relative to the next. They are
compact and efficient for forward kinematics, so many robot solvers use them
internally. However, they require assigning frames according to a specific
convention, which can be awkward for some robot geometries, and do not build the
same intuition for velocities and Jacobians as twists. For those reasons, we
use twists in class and provide the DH parameters needed by EAIK here.

These standard UR7e DH parameters are published by Universal Robots at the
first source linked above. Students do not need to derive or modify them.
"""
UR7E_DH_D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
UR7E_DH_A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
UR7E_DH_ALPHA = np.array(
    [np.pi / 2, 0.0, 0.0, np.pi / 2, -np.pi / 2, 0.0]
)

robot = DhRobot(UR7E_DH_ALPHA, UR7E_DH_A, UR7E_DH_D)

# Space-frame twists and zero configuration for the same UR7e model. These
# constants let students reuse their HW2 product-of-exponentials code.
UR7E_TWISTS = np.array(
    [
        [0.0, 0.1625, 0.1625, 0.1625, 0.1333, 0.0628],
        [0.0, 0.0, 0.0, 0.0, -0.8172, 0.0],
        [0.0, 0.0, 0.4250, 0.8172, 0.0, 0.8172],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, -1.0, -1.0, -1.0, 0.0, -1.0],
        [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
    ]
)

UR7E_ZERO_POSE = np.array(
    [
        [1.0, 0.0, 0.0, -0.8172],
        [0.0, 0.0, -1.0, -0.2329],
        [0.0, 1.0, 0.0, 0.0628],
        [0.0, 0.0, 0.0, 1.0],
    ]
)

POSITION_TOLERANCE = 1e-5
ORIENTATION_TOLERANCE = 1e-4


def pose_error(actual_end_effector_pose, desired_end_effector_pose):
    """Measure the position and orientation difference between two poses.

    Args:
        actual_end_effector_pose: NumPy array with shape ``(4, 4)`` for the
            achieved pose.
        desired_end_effector_pose: NumPy array with shape ``(4, 4)`` for the
            desired pose.

    Returns:
        ``(position_error, orientation_error)``. Position error is the
        Euclidean distance between the positions in meters. Orientation error
        is the smallest rotation angle between the orientations in radians.

    This is provided so everyone verifies poses in the same way. ``np.clip``
    only protects ``arccos`` from tiny roundoff errors.
    """
    position_error = np.linalg.norm(
        actual_end_effector_pose[:3, 3] - desired_end_effector_pose[:3, 3]
    )
    relative_rotation = (
        actual_end_effector_pose[:3, :3].T
        @ desired_end_effector_pose[:3, :3]
    )
    cosine = (np.trace(relative_rotation) - 1.0) / 2.0
    orientation_error = np.arccos(np.clip(cosine, -1.0, 1.0))
    return position_error, orientation_error
