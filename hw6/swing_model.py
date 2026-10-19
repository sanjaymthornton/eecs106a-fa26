"""Provided chair-swing geometry and display joint coordinates."""

from pathlib import Path

import numpy as np


def load_swing():
    """Return the desired ride and free-moving chair URDF models.

    Both models use local primitive geometry, without mesh downloads.
    The ride dimensions match L1=0.57, L2=0.25, L3=0.20, and L4=0.10.
    The chair's base can be set to any integrated SE(3) pose; no IK is used.
    """
    from roboticstoolbox.models.URDF.URDFRobot import URDFRobot

    assets = Path(__file__).resolve().parent / "assets"
    return URDFRobot(str(assets / "swing.urdf")), URDFRobot(str(assets / "chair.urdf"))


def swing_joint_values(pose):
    """Return the desired ride's display coordinates from a chair pose.

    pose is a NumPy array (4, 4) in the ride frame. Return a NumPy array
    (3,) containing fulcrum lift in meters, spin in radians, and sway in
    radians. These coordinates display the desired assembly only; never
    apply them to the integrated chair, which must retain its full pose.
    """
    rotation = pose[:3, :3]
    attachment = pose[:3, 3] + 0.20 * rotation[:, 2]
    lift = attachment[2] - 0.57
    spin = np.arctan2(rotation[1, 0], rotation[0, 0])
    sway = np.arctan2(rotation[2, 1], rotation[2, 2])
    return np.array([lift, spin, sway])
