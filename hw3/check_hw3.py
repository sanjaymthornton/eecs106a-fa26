#!/usr/bin/env python3
"""Lightweight HW3 interface checks suitable for a future student release."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
from scipy.spatial.transform import Rotation as Rot


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load {path.name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_submission(hw3_path: Path, kin_path: Path):
    sys.modules.pop("hw3_submission", None)
    sys.modules.pop("hw2.kin_func_skeleton", None)

    # hw3.py imports the one shared kinematics file from hw2/. Register that
    # module name explicitly so this loader also works in private test folders.
    hw2_package = types.ModuleType("hw2")
    hw2_package.__path__ = [str(kin_path.parent)]
    sys.modules["hw2"] = hw2_package
    kin = load_module(kin_path, "hw2.kin_func_skeleton")
    hw3 = load_module(hw3_path, "hw3_submission")
    return hw3, kin


def load_rotation_submission(rotation_path: Path):
    sys.modules.pop("rotation_learning_submission", None)
    return load_module(rotation_path, "rotation_learning_submission")


def interface_checks(hw3) -> list[tuple[str, bool]]:
    checks = []
    for problem, degrees_of_freedom in ((1, 3), (2, 3), (3, 6)):
        function = getattr(hw3, f"fk_{problem}", None)
        try:
            pose, twists = function(np.zeros(degrees_of_freedom))
            pose = np.asarray(pose)
            twists = np.asarray(twists)
            valid = (
                pose.shape == (4, 4)
                and twists.shape == (6, degrees_of_freedom)
                and np.all(np.isfinite(pose))
                and np.all(np.isfinite(twists))
            )
        except BaseException:
            valid = False
        checks.append((f"fk_{problem} returns finite arrays with the documented shapes", valid))
    return checks


def rotation_checks(rotation, kin) -> list[tuple[str, bool]]:
    checks = []

    try:
        values = np.array([0.31, 1.17, 2.42])
        angles = np.stack([
            0.4 * np.sin(values), 0.3 * np.sin(2 * values), values
        ], axis=1)
        actual = rotation.rotations_from_s(values).as_matrix()
        expected = Rot.from_euler("XYZ", angles).as_matrix()
        extrinsic = Rot.from_euler("xyz", angles).as_matrix()
        valid = (
            actual.shape == (3, 3, 3)
            and np.allclose(actual, expected)
            and not np.allclose(actual, extrinsic)
        )
    except BaseException:
        valid = False
    checks.append(("rotations_from_s uses intrinsic uppercase XYZ angles", valid))

    try:
        data = rotation.make_dataset(seed=0)
        rng = np.random.default_rng(0)
        s0 = rng.uniform(0, 2 * np.pi, 500)
        expected_s = np.concatenate([s0, s0 + 2 * np.pi])
        expected_indices = rng.permutation(1000)
        expected_angles = np.stack([
            0.4 * np.sin(expected_s),
            0.3 * np.sin(2 * expected_s),
            expected_s,
        ], axis=1)
        expected_rotations = Rot.from_euler("XYZ", expected_angles)
        expected_sweep = np.linspace(0, 4 * np.pi, 4000, endpoint=False)
        expected_sweep_angles = np.stack([
            0.4 * np.sin(expected_sweep),
            0.3 * np.sin(2 * expected_sweep),
            expected_sweep,
        ], axis=1)
        expected_sweep_rotations = Rot.from_euler("XYZ", expected_sweep_angles)
        expected_X = np.stack([np.cos(expected_s), np.sin(expected_s)], axis=1)
        expected_X_sweep = np.stack([
            np.cos(expected_sweep), np.sin(expected_sweep)
        ], axis=1)
        valid = (
            set(("s", "X", "rot", "R", "train", "eval", "s_sweep",
                 "X_sweep", "rot_sweep", "R_sweep")).issubset(data)
            and np.allclose(data["s"], expected_s)
            and np.allclose(data["X"], expected_X)
            and np.allclose(data["X"][:500], data["X"][500:])
            and np.allclose(data["R"], expected_rotations.as_matrix())
            and np.allclose(data["R"][:500], data["R"][500:])
            and np.allclose(
                data["rot"].as_quat()[:500],
                -data["rot"].as_quat()[500:],
            )
            and np.allclose(data["rot"].as_matrix(), expected_rotations.as_matrix())
            and np.array_equal(data["train"], expected_indices[:900])
            and np.array_equal(data["eval"], expected_indices[900:])
            and np.allclose(data["s_sweep"], expected_sweep)
            and np.allclose(data["X_sweep"], expected_X_sweep)
            and np.allclose(
                data["R_sweep"], expected_sweep_rotations.as_matrix()
            )
            and np.allclose(data["rot_sweep"].as_matrix(), data["R_sweep"])
        )
    except BaseException:
        valid = False
    checks.append(("make_dataset constructs the documented paired data and sweep", valid))

    known = Rot.from_euler(
        "XYZ", [[0.2, -0.1, 0.4], [0.8, 0.3, -0.5], [1.7, -0.2, 0.6], [3.1, 0.1, -0.3]]
    ).as_matrix()
    raw_matrices = known.reshape(-1, 9) + np.linspace(-0.01, 0.01, 36).reshape(4, 9)
    try:
        decoded = np.asarray(rotation.decode_matrix(raw_matrices))
        expected = kin.renormalize_SO3(raw_matrices.reshape(-1, 3, 3))
        valid = (
            decoded.shape == known.shape
            and np.allclose(decoded, expected, atol=1e-10)
            and getattr(rotation, "renormalize_SO3", None)
            is getattr(kin, "renormalize_SO3", None)
        )
    except BaseException:
        valid = False
    checks.append(("decode_matrix projects flattened predictions onto SO(3)", valid))

    try:
        angles = np.array([[0.31, -0.47, 0.82], [-0.2, 0.6, 1.1]])
        decoded = np.asarray(rotation.decode_euler(angles))
        expected = Rot.from_euler("XYZ", angles).as_matrix()
        extrinsic = Rot.from_euler("xyz", angles).as_matrix()
        valid = decoded.shape == (2, 3, 3) and np.allclose(decoded, expected) and not np.allclose(decoded, extrinsic)
    except BaseException:
        valid = False
    checks.append(("decode_euler uses intrinsic uppercase XYZ angles", valid))

    axes = np.array([[1., 0, 0], [0, 1., 0], [0, 0, 1.], [-1., 2, -3.]])
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    branch = np.concatenate([
        Rot.from_rotvec(np.pi * axes).as_matrix(),
        Rot.from_rotvec((np.pi - 1e-5) * axes[-1:]).as_matrix(),
        Rot.from_rotvec((np.pi + 1e-5) * axes[-1:]).as_matrix(),
    ])
    exp_rotations = np.concatenate([np.eye(3)[None], known, branch])
    try:
        vectors = np.asarray(rotation.encode_exponential(exp_rotations))
        expected = np.array([
            kin.so3_to_R3(kin.SO3_to_so3(matrix)) for matrix in exp_rotations
        ])
        valid = (
            vectors.shape == (len(exp_rotations), 3)
            and np.allclose(vectors, expected, atol=1e-6)
            and all(
                getattr(rotation, name, None) is getattr(kin, name, None)
                for name in ("so3_to_R3", "SO3_to_so3")
            )
        )
    except BaseException:
        valid = False
    checks.append(("encode_exponential handles identity and exact/near-pi rotations", valid))

    try:
        expected_vectors = np.array([
            kin.so3_to_R3(kin.SO3_to_so3(matrix)) for matrix in exp_rotations
        ])
        decoded = np.asarray(rotation.decode_exponential(expected_vectors))
        valid = (
            decoded.shape == exp_rotations.shape
            and np.allclose(decoded, exp_rotations, atol=1e-7)
            and all(
                getattr(rotation, name, None) is getattr(kin, name, None)
                for name in ("R3_to_so3", "so3_to_SO3")
            )
        )
    except BaseException:
        valid = False
    checks.append(("decode_exponential reconstructs identity and exact/near-pi rotations", valid))

    raw_quaternions = np.array([[1., 2, 3, 4], [-2., 0.5, 1., -3.], [0., 0, 0, 0]])
    try:
        normalized = np.asarray(rotation.normalize_quaternions(raw_quaternions))
        norms = np.linalg.norm(raw_quaternions, axis=1, keepdims=True)
        expected = raw_quaternions / np.maximum(norms, 1e-12)
        valid = normalized.shape == (3, 4) and np.allclose(normalized, expected)
    except BaseException:
        valid = False
    checks.append(("normalize_quaternions normalizes rows without dividing by zero", valid))

    try:
        saved = raw_quaternions.copy()
        canonical = np.asarray(rotation.canonicalize_quaternions(raw_quaternions))
        expected = saved.copy()
        expected[expected[:, 3] < 0] *= -1
        valid = canonical.shape == (3, 4) and np.allclose(canonical, expected) and np.array_equal(raw_quaternions, saved)
    except BaseException:
        valid = False
    checks.append(("canonicalize_quaternions applies w >= 0 without modifying input", valid))

    try:
        decoded = np.asarray(rotation.decode_quaternions(raw_quaternions[:2]))
        normalized = raw_quaternions[:2] / np.linalg.norm(raw_quaternions[:2], axis=1, keepdims=True)
        expected = Rot.from_quat(normalized).as_matrix()
        valid = decoded.shape == (2, 3, 3) and np.allclose(decoded, expected)
    except BaseException:
        valid = False
    checks.append(("decode_quaternions normalizes and decodes predictions", valid))

    try:
        encoded = np.asarray(rotation.encode_6d(known))
        expected = known[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
        valid = encoded.shape == (4, 6) and np.allclose(encoded, expected)
    except BaseException:
        valid = False
    checks.append(("encode_6d stores the first two rotation columns", valid))

    try:
        encoded = known[:, :, :2].transpose(0, 2, 1).reshape(-1, 6)
        decoded = np.asarray(rotation.decode_6d(encoded))
        valid = (
            decoded.shape == known.shape
            and np.allclose(decoded, known, atol=1e-10)
            and np.allclose(decoded.transpose(0, 2, 1) @ decoded, np.repeat(np.eye(3)[None], len(decoded), axis=0))
            and np.allclose(np.linalg.det(decoded), 1)
        )
    except BaseException:
        valid = False
    checks.append(("decode_6d reconstructs valid rotations with Gram--Schmidt", valid))

    return checks


def main() -> int:
    directory = Path(__file__).resolve().parent
    kin_path = directory.parent / "hw2" / "kin_func_skeleton.py"
    if not kin_path.is_file():
        # Canonical staff source layout: sources/hw2/student/hw2/...
        kin_path = (
            directory.parents[2]
            / "hw2"
            / "student"
            / "hw2"
            / "kin_func_skeleton.py"
        )
    try:
        hw3, kin = load_submission(directory / "hw3.py", kin_path)
        rotation = load_rotation_submission(directory / "rotation_learning.py")
        checks = interface_checks(hw3) + rotation_checks(rotation, kin)
    except BaseException as error:
        print(f"HW3 self-check could not import the submission: {type(error).__name__}: {error}")
        return 1
    for name, passed in checks:
        print(f"[{'PASS' if passed else 'FAIL'}] {name}")
    return 0 if all(passed for _, passed in checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
