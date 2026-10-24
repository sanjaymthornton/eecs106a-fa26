"""Provided UR7e kinematics for HW7, matching the HW4 tool0 frame.

The visualizer reuses the official description and meshes used in HW4.
https://www.universal-robots.com/articles/ur/application-installation/dh-parameters-for-calculations-of-kinematics-and-dynamics/
"""

import numpy as np

from hw2.kin_func_skeleton import forward_kinematics, twist_to_SE3


UR7E_TWISTS = np.array([
    [0.0, 0.1625, 0.1625, 0.1625, 0.1333, 0.0628],
    [0.0, 0.0, 0.0, 0.0, -0.8172, 0.0],
    [0.0, 0.0, 0.4250, 0.8172, 0.0, 0.8172],
    [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    [0.0, -1.0, -1.0, -1.0, 0.0, -1.0],
    [1.0, 0.0, 0.0, 0.0, -1.0, 0.0],
])
UR7E_ZERO_POSE = np.array([
    [1.0, 0.0, 0.0, -0.8172],
    [0.0, 0.0, -1.0, -0.2329],
    [0.0, 1.0, 0.0, 0.0628],
    [0.0, 0.0, 0.0, 1.0],
])
UR7E_READY = np.array([0.0, -0.8, 1.4, -0.6, np.pi / 2, 0.0])
UR7E_EXTENDED = np.array([0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.0])


def ur7e_fk(theta):
    """Return the base-frame tool0 pose (4, 4) for joint angles theta (6,).

    Uses the student's HW2 product of exponentials with angles in radians.
    """
    return forward_kinematics(UR7E_TWISTS, theta) @ UR7E_ZERO_POSE


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


def load_ur7e():
    """Return the Robotics Toolbox UR7e mesh model and its tool0 link.

    The first load may download the official description, as in HW4.
    """
    from roboticstoolbox.models.URDF.URDFRobot import URDFRobot
    from spatialmath import SE3

    robot = URDFRobot("ur7e")
    # ROS base_link differs from the course DH base by a half-turn about z.
    robot.base = SE3.Rz(np.pi)
    tool = next(link for link in robot.ee_links if link.name == "tool0")
    return robot, tool
