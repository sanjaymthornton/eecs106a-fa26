#!/usr/bin/env python3
"""Lightweight HW2 checks; no private autograder answers."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_submission(hw2_path: Path, kin_path: Path):
    """Load the kinematics module first so hw2.py can import it."""
    sys.modules.pop("hw2_submission", None)
    sys.modules.pop("kin_func_skeleton", None)
    kin = load_module(kin_path, "kin_func_skeleton")
    hw2 = load_module(hw2_path, "hw2_submission")
    return hw2, kin


def interface_checks(hw2, kin) -> list[tuple[str, bool]]:
    checks = []

    try:
        actual = hw2.box_pose_in_world(2.0, 0.5, 1.2)
        expected = np.eye(4)
        expected[:3, 3] = [1.0, 1.2, 0.0]
        passed = np.allclose(actual, expected)
    except BaseException:
        passed = False
    checks.append(("box_pose_in_world passes a simple translation example", passed))

    try:
        actual = hw2.camera_pose_in_world(0.0, 0.4, 1.5)
        expected = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, -1, 0, 1.5],
            [0, 0, 0, 1],
        ])
        passed = np.allclose(actual, expected)
    except BaseException:
        passed = False
    checks.append(("camera_pose_in_world has the expected pose at t = 0", passed))

    try:
        box_pose = hw2.box_pose_in_world(1.0, 0.4, 1.0)
        camera_pose = hw2.camera_pose_in_world(1.0, 0.2, 1.5)
        actual = hw2.box_pose_in_camera(1.0, 0.4, 0.2, 1.0, 1.5)
        passed = np.allclose(camera_pose @ actual, box_pose)
    except BaseException:
        passed = False
    checks.append(("box_pose_in_camera uses the correct frame composition", passed))

    try:
        actual = hw2.box_twist_in_world(0.5)
        passed = np.allclose(actual, [0.5, 0, 0, 0, 0, 0])
    except BaseException:
        passed = False
    checks.append(("box_twist_in_world uses [v, omega] ordering", passed))

    try:
        actual = hw2.camera_twist_in_world(0.4)
        passed = np.allclose(actual, [0, 0, 0, 0, 0, -0.4])
    except BaseException:
        passed = False
    checks.append(("camera_twist_in_world uses clockwise angular velocity", passed))

    for name, arguments, shape in (
        ("R3_to_so3", (np.array([1., 2, 3]),), (3, 3)),
        ("so3_to_R3", (np.zeros((3, 3)),), (3,)),
        ("axis_angle_to_SO3", (np.array([0., 0, 1]), 0.5), (3, 3)),
        ("so3_to_SO3", (np.zeros((3, 3)),), (3, 3)),
        ("twist_to_se3", (np.array([1., 0, 0, 0, 0, 0]),), (4, 4)),
        ("se3_to_twist", (np.zeros((4, 4)),), (6,)),
        ("twist_to_SE3", (np.array([1., 0, 0, 0, 0, 0]), 0.5), (4, 4)),
        ("se3_to_SE3", (np.zeros((4, 4)), 0.5), (4, 4)),
        ("forward_kinematics", (np.array([[1., 0, 0, 0, 0, 0]]).T, np.array([0.5])), (4, 4)),
    ):
        try:
            result = np.asarray(getattr(kin, name)(*arguments))
            passed = result.shape == shape and np.all(np.isfinite(result))
        except BaseException:
            passed = False
        checks.append((f"{name} returns a finite array with shape {shape}", bool(passed)))
    return checks


def main() -> int:
    directory = Path(__file__).resolve().parent
    try:
        hw2, kin = load_submission(directory / "hw2.py", directory / "kin_func_skeleton.py")
        checks = interface_checks(hw2, kin)
    except BaseException as error:
        print(f"HW2 self-check could not import the submission: {type(error).__name__}: {error}")
        return 1

    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    return 0 if all(passed for _, passed in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
