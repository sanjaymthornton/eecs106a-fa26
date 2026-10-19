"""Every scored HW6 case; run python hw6/check_hw6.py.

"""

import json
from pathlib import Path
import sys

import numpy as np


HW6_DIR = Path(__file__).resolve().parent
for parent in (HW6_DIR.parent, HW6_DIR.parent / "hw2" / "student"):
    if parent.is_dir() and str(parent) not in sys.path:
        sys.path.append(str(parent))

FUNCTION_NAMES = ("trajectory", "body_velocity", "euler_step")


def cases(function_name):
    """Yield all public deterministic inputs for one scored function."""
    if function_name in ("trajectory", "body_velocity"):
        times = ((0.0, 1e-8, 0.7, np.pi / 2, 4 * np.pi) if function_name == "trajectory"
                 else (0.0, 1e-8, 0.2, 0.7, np.pi / 2, np.pi, 3 * np.pi / 2,
                       2 * np.pi, 8.3, 4 * np.pi))
        for time in times:
            yield f"t = {time:g} s", {"t": time}
        return
    pose = np.eye(4)
    moved = np.eye(4)
    moved[:3, 3] = [0.4, -0.2, 0.7]
    rotated = moved.copy()
    rotated[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    translation = np.zeros((4, 4))
    translation[:3, 3] = [0.3, -0.2, 0.1]
    screw = np.array([[0.0, -0.4, -0.2, 0.3], [0.4, 0.0, -0.1, -0.2],
                      [0.2, 0.1, 0.0, 0.1], [0.0, 0.0, 0.0, 0.0]])
    if function_name == "euler_step":
        pi_rotation = np.zeros((4, 4))
        pi_rotation[0, 1], pi_rotation[1, 0] = -np.pi, np.pi
        for name, actual, velocity, dt in (
            ("stationary pose", moved, np.zeros((4, 4)), 0.1),
            ("zero timestep", rotated, screw, 0.0),
            ("body translation from a rotated pose", rotated, translation, 0.3),
            ("general screw motion", moved, screw, 0.2),
            ("exact pi rotation", rotated, pi_rotation, 1.0),
            ("near pi rotation", rotated, pi_rotation, 1.0 - 1e-6),
            ("tiny timestep", rotated, screw, 1e-15),
            ("negative velocity and larger timestep", moved, -screw, 2.0),
        ):
            yield name, {"pose": actual.copy(), "velocity_hat": velocity.copy(), "dt": dt}


def output_matches(function_name, actual, expected):
    """Check shape, finite values, numerical agreement, and matrix structure."""
    actual = np.asarray(actual)
    if actual.shape != (4, 4) or not np.isfinite(actual).all():
        return False
    if not np.allclose(actual, expected, atol=1e-7, rtol=1e-7):
        return False
    if function_name in ("trajectory", "euler_step"):
        rotation = actual[:3, :3]
        return (np.allclose(actual[3], [0, 0, 0, 1], atol=1e-7)
                and np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-7)
                and np.isclose(np.linalg.det(rotation), 1, atol=1e-7))
    if not np.allclose(actual[3], 0, atol=1e-7):
        return False
    if function_name == "body_velocity":
        return np.allclose(actual[:3, :3], -actual[:3, :3].T, atol=1e-7)
    return True


def run_checks():
    """Run the public cases with the student's own HW2 and HW6 functions."""
    import hw6

    expected = json.loads(HW6_DIR.joinpath("expected_results.json").read_text())
    passed_count = 0
    count = 0
    for function_name in FUNCTION_NAMES:
        for index, (name, arguments) in enumerate(cases(function_name)):
            try:
                actual = getattr(hw6, function_name)(**arguments)
                passed = output_matches(function_name, actual, expected[function_name][index])
            except Exception as error:
                passed = False
                print(f"{type(error).__name__}: {error}")
            count += 1
            passed_count += int(passed)
            print(f"[{'PASS' if passed else 'FAIL'}] {function_name}: {name}")
    print(f"{passed_count}/{count} HW6 checks passed.")
    return passed_count == count


if __name__ == "__main__":
    raise SystemExit(0 if run_checks() else 1)
