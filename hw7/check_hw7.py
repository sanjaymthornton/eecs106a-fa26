"""Public checks for the thirteen student functions in HW7.

Run from the semester repository root with:

    python hw7/check_hw7.py

The functions below define every deterministic input used by the HW7
autograder. The local checker and autograder use the same cases.
"""

from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


HW7_DIR = Path(__file__).resolve().parent
REPOSITORY_ROOT = HW7_DIR.parent
KIN_PARENT = REPOSITORY_ROOT / "hw2" / "student"
for path in (REPOSITORY_ROOT, KIN_PARENT):
    if path.is_dir() and str(path) not in sys.path:
        sys.path.insert(0, str(path))


JACOBIAN_FUNCTION_NAMES = (
    "adjoint",
    "adjoint_inv",
    "spatial_jacobian",
    "body_jacobian",
    "spatial_velocity",
    "body_velocity",
    "spatial_configuration_error",
    "spatial_error_twist",
    "jacobian_inverse_kinematics",
)

EULER_FUNCTION_NAMES = ("euler_jacobian", "manipulability")

IK_FUNCTION_NAMES = (
    "null_space_ik_step",
    "joint_limit_gradient",
)

FUNCTION_NAMES = JACOBIAN_FUNCTION_NAMES + EULER_FUNCTION_NAMES + IK_FUNCTION_NAMES

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
ATOL = 1e-7
RTOL = 1e-7


def make_transform(angle, translation):
    """Return a transform with a z-axis rotation for public test cases."""
    cosine = np.cos(angle)
    sine = np.sin(angle)
    result = np.eye(4)
    result[:3, :3] = np.array(
        [
            [cosine, -sine, 0.0],
            [sine, cosine, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    result[:3, 3] = translation
    return result


def make_general_transform(rotation_vector, translation):
    """Return a transform from a rotation vector and translation."""
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    result[:3, 3] = translation
    return result


def adjoint_cases():
    """Yield every public test case for ``adjoint``."""
    yield "identity transformation", {"g": np.eye(4)}
    yield "pure translation", {
        "g": make_transform(0.0, np.array([0.4, -0.7, 1.2]))
    }
    yield "rotation and translation", {
        "g": make_transform(0.63, np.array([-0.3, 0.8, 0.5]))
    }
    yield "arbitrary-axis rotation", {
        "g": make_general_transform(
            np.array([0.35, -0.42, 0.18]),
            np.zeros(3),
        )
    }
    yield "arbitrary rigid transformation", {
        "g": make_general_transform(
            np.array([-0.27, 0.31, 0.46]),
            np.array([0.9, -0.6, 0.25]),
        )
    }


def adjoint_inv_cases():
    """Yield every public test case for ``adjoint_inv``."""
    yield "identity transformation", {"g": np.eye(4)}
    yield "pure translation", {
        "g": make_transform(0.0, np.array([-0.2, 0.5, 0.9]))
    }
    yield "rotation and translation", {
        "g": make_transform(-0.48, np.array([0.6, -0.1, 0.4]))
    }
    yield "arbitrary-axis rotation", {
        "g": make_general_transform(
            np.array([-0.2, 0.38, 0.29]),
            np.zeros(3),
        )
    }
    yield "arbitrary rigid transformation", {
        "g": make_general_transform(
            np.array([0.41, -0.16, -0.33]),
            np.array([-0.75, 0.2, 0.55]),
        )
    }


def robot_cases():
    """Yield robots used by the public Jacobian and velocity checks."""
    single_twist = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).T
    yield (
        "single revolute joint",
        single_twist,
        np.array([0.7]),
        make_transform(0.0, np.array([0.5, 0.0, 0.2])),
    )

    two_twists = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
        ]
    ).T
    yield (
        "two revolute joints",
        two_twists,
        np.array([0.35, -0.6]),
        make_transform(0.2, np.array([0.8, -0.3, 0.4])),
    )

    mixed_twists = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    ).T
    yield (
        "revolute and prismatic joints",
        mixed_twists,
        np.array([0.4, -0.25, 0.3]),
        make_transform(-0.3, np.array([0.7, 0.2, 1.0])),
    )
    yield (
        "zero joint values",
        mixed_twists,
        np.zeros(3),
        make_transform(0.5, np.array([-0.4, 0.6, 0.9])),
    )

    integer_twists = np.array(
        [
            [0, 0, 0, 0, 0, 1],
            [0, 0, -1, 1, 0, 0],
            [-1, 0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
        ],
        dtype=int,
    ).T
    yield (
        "integer screw axes and four joints",
        integer_twists,
        np.array([0.22, -0.51, 0.37, -0.15]),
        make_general_transform(
            np.array([0.2, -0.3, 0.1]),
            np.array([0.6, -0.5, 1.2]),
        ),
    )
    yield (
        "non-unit arbitrary-axis and helical twists",
        np.array([
            [0.2, -0.1, 0.4, 0.3, 0.8, -0.5],
            [-0.3, 0.4, 0.1, -0.7, 0.2, 0.6],
            [0.4, -0.2, 0.3, 0.0, 0.0, 0.0],
        ]).T,
        np.array([0.31, -0.42, 0.2]),
        make_general_transform(np.array([0.3, 0.1, -0.2]), np.array([0.4, -0.3, 0.7])),
    )
    yield "seven-joint Panda", PANDA_TWISTS.copy(), PANDA_READY.copy(), PANDA_G_ST0.copy()
    yield "singular seven-joint Panda", PANDA_TWISTS.copy(), np.zeros(7), PANDA_G_ST0.copy()
    near_singular = np.zeros(7)
    near_singular[3] = -1e-4
    yield "near-singular seven-joint Panda", PANDA_TWISTS.copy(), near_singular, PANDA_G_ST0.copy()


def spatial_jacobian_cases():
    """Yield every public test case for ``spatial_jacobian``."""
    for name, twists, theta, _ in robot_cases():
        yield name, {"twists": twists, "theta": theta}


def body_jacobian_cases():
    """Yield every public test case for ``body_jacobian``."""
    for name, twists, theta, g_st0 in robot_cases():
        yield name, {"twists": twists, "theta": theta, "g_st0": g_st0}


def spatial_velocity_cases():
    """Yield every public test case for ``spatial_velocity``."""
    rates = (
        np.array([0.45]),
        np.array([-0.2, 0.7]),
        np.array([0.3, -0.5, 0.8]),
        np.array([-0.4, 0.6, 0.2]),
        np.zeros(4),
        np.array([0.2, -0.3, 0.4]),
        np.array([0.1, -0.2, 0.3, -0.4, 0.2, 0.1, -0.1]),
        np.array([-0.1, 0.3, -0.2, 0.1, 0.2, -0.1, 0.4]),
        np.array([0.2, -0.1, 0.1, 0.3, -0.2, 0.4, -0.3]),
    )
    for (name, twists, theta, _), theta_dot in zip(robot_cases(), rates):
        yield name, {
            "twists": twists,
            "theta": theta,
            "theta_dot": theta_dot,
        }


def body_velocity_cases():
    """Yield every public test case for ``body_velocity``."""
    rates = (
        np.array([0.45]),
        np.array([-0.2, 0.7]),
        np.array([0.3, -0.5, 0.8]),
        np.array([-0.4, 0.6, 0.2]),
        np.zeros(4),
        np.array([0.2, -0.3, 0.4]),
        np.array([0.1, -0.2, 0.3, -0.4, 0.2, 0.1, -0.1]),
        np.array([-0.1, 0.3, -0.2, 0.1, 0.2, -0.1, 0.4]),
        np.array([0.2, -0.1, 0.1, 0.3, -0.2, 0.4, -0.3]),
    )
    for (name, twists, theta, g_st0), theta_dot in zip(robot_cases(), rates):
        yield name, {
            "twists": twists,
            "theta": theta,
            "theta_dot": theta_dot,
            "g_st0": g_st0,
        }


def jacobian_ik_step_cases():
    """Yield every public test case for ``jacobian_ik_step``."""
    yield "zero velocity", {
        "theta": PANDA_READY.copy(),
        "velocity": np.zeros(6),
    }
    yield "linear body velocity", {
        "theta": PANDA_READY.copy(),
        "velocity": np.array([0.08, -0.04, 0.03, 0.0, 0.0, 0.0]),
    }
    yield "mixed body velocity and smaller step", {
        "theta": np.array([0.2, -0.6, 0.3, -1.8, -0.2, 1.4, 0.5]),
        "velocity": np.array([0.04, 0.02, -0.03, 0.1, -0.08, 0.05]),
        "step_size": 0.1,
    }
    yield "singular zero configuration", {
        "theta": np.zeros(7),
        "velocity": np.array([0.02, -0.01, 0.03, 0.04, 0.0, -0.02]),
        "step_size": 0.3,
    }
    yield "negative joint values and angular velocity", {
        "theta": np.array([-0.4, -0.8, -0.2, -2.4, -0.3, 1.1, -0.5]),
        "velocity": np.array([0.0, 0.0, 0.0, -0.08, 0.06, 0.1]),
    }


    yield "zero step size", {
        "theta": PANDA_READY.copy(), "velocity": np.ones(6), "step_size": 0.0,
    }
    near_singular = np.zeros(7)
    near_singular[3] = -1e-4
    yield "near-singular configuration", {
        "theta": near_singular,
        "velocity": np.array([0.01, -0.02, 0.03, -0.01, 0.02, 0.01]),
        "step_size": 0.1,
    }


    regular = np.array([0.0, -0.8, 1.4, -0.6, np.pi / 2, 0.0])
    extended = np.array([0.0, 0.0, 0.0, 0.0, np.pi / 2, 0.0])
    near = extended.copy()
    near[1:4] = [-1e-4, 2e-4, -1e-4]
    for name, theta, velocity, step in (
        ("UR7e regular posture", regular, [0.04, -0.03, 0.02, 0.01, 0.03, -0.02], 0.2),
        ("UR7e zero velocity", regular, np.zeros(6), 0.2),
        ("UR7e zero step", regular, np.ones(6), 0.0),
        ("UR7e exact elbow singularity", extended, [0.03, 0, 0, 0, 0, 0], 0.1),
        ("UR7e near elbow singularity", near, [0.03, 0, 0, 0, 0, 0], 0.1),
    ):
        yield name, {"theta": theta.copy(), "velocity": np.array(velocity),
                     "step_size": step, "twists": UR7E_TWISTS.copy(),
                     "g_st0": UR7E_ZERO_POSE.copy()}


def null_space_ik_step_cases():
    """Yield every public test case for ``null_space_ik_step``."""
    yield "zero primary and secondary motion", {
        "theta": PANDA_READY.copy(),
        "velocity": np.zeros(6),
        "secondary_gradient": np.zeros(7),
    }
    yield "primary motion only", {
        "theta": PANDA_READY.copy(),
        "velocity": np.array([0.06, -0.02, 0.04, 0.03, 0.01, -0.05]),
        "secondary_gradient": np.zeros(7),
    }
    yield "projected secondary motion only", {
        "theta": PANDA_READY.copy(),
        "velocity": np.zeros(6),
        "secondary_gradient": np.array(
            [0.4, -0.3, 0.2, 0.5, -0.1, 0.25, -0.35]
        ),
    }
    yield "combined primary and secondary motion", {
        "theta": np.array([0.25, -0.7, 0.4, -2.0, 0.2, 1.6, -0.3]),
        "velocity": np.array([0.03, 0.05, -0.02, -0.04, 0.07, 0.02]),
        "secondary_gradient": np.array(
            [-0.2, 0.1, 0.3, -0.15, 0.25, -0.4, 0.2]
        ),
    }
    yield "custom gains at a singular configuration", {
        "theta": np.zeros(7),
        "velocity": np.array([0.01, -0.03, 0.02, 0.05, -0.02, 0.04]),
        "secondary_gradient": np.array(
            [0.3, 0.2, -0.1, 0.4, -0.25, 0.15, -0.35]
        ),
        "step_size": 0.12,
        "secondary_gain": 0.08,
    }


    yield "zero step size", {
        "theta": PANDA_READY.copy(), "velocity": np.ones(6),
        "secondary_gradient": np.arange(7, dtype=float), "step_size": 0.0,
    }
    yield "zero secondary gain", {
        "theta": PANDA_READY.copy(), "velocity": np.ones(6) * 0.02,
        "secondary_gradient": np.arange(7, dtype=float), "secondary_gain": 0.0,
    }
    near_singular = np.zeros(7)
    near_singular[3] = -1e-4
    yield "near-singular configuration", {
        "theta": near_singular, "velocity": np.ones(6) * 0.01,
        "secondary_gradient": np.array([0.2, -0.1, 0.4, -0.2, 0.1, 0.3, -0.4]),
    }


def joint_limit_gradient_cases():
    """Yield every public test case for ``joint_limit_gradient``."""
    midpoint = (PANDA_THETA_MIN + PANDA_THETA_MAX) / 2
    half_range = (PANDA_THETA_MAX - PANDA_THETA_MIN) / 2
    yield "joint-range midpoint", {"theta": midpoint}
    yield "above every midpoint", {"theta": midpoint + 0.4 * half_range}
    yield "below every midpoint", {"theta": midpoint - 0.35 * half_range}
    yield "mixed normalized offsets", {
        "theta": midpoint
        + half_range * np.array([0.2, -0.4, 0.6, -0.3, 0.1, 0.5, -0.2])
    }
    yield "Panda ready configuration", {"theta": PANDA_READY.copy()}
    yield "exact lower joint limits", {"theta": PANDA_THETA_MIN.copy()}
    yield "exact upper joint limits", {"theta": PANDA_THETA_MAX.copy()}


def spatial_configuration_error_cases():
    """Yield every public test case for ``spatial_configuration_error``."""
    current = make_general_transform(
        np.array([0.4, -0.3, 0.2]), np.array([0.6, -0.5, 0.8]))
    desired = make_general_transform(
        np.array([-0.2, 0.5, 0.3]), np.array([-0.4, 0.7, 0.2]))
    yield "identity poses", {"pose": np.eye(4), "desired_pose": np.eye(4)}
    yield "equal nonidentity poses", {"pose": current.copy(), "desired_pose": current.copy()}
    yield "translation error", {
        "pose": make_transform(0.0, np.array([0.3, -0.2, 0.4])),
        "desired_pose": make_transform(0.0, np.array([-0.5, 0.7, 0.1])),
    }
    yield "rotation error", {
        "pose": np.eye(4),
        "desired_pose": make_general_transform(np.array([-0.3, 0.4, 0.2]), np.zeros(3)),
    }
    yield "rotated current pose and translated target", {
        "pose": current.copy(),
        "desired_pose": make_transform(0.0, np.array([0.2, 0.4, -0.3])),
    }
    yield "noncommuting poses", {"pose": current.copy(), "desired_pose": desired.copy()}
    yield "reversed noncommuting poses", {"pose": desired.copy(), "desired_pose": current.copy()}
    yield "tiny displacement", {
        "pose": np.eye(4),
        "desired_pose": make_general_transform(
            np.array([2e-9, -3e-9, 1e-9]), np.array([-4e-9, 2e-9, 3e-9])),
    }
    yield "identity target", {"pose": current.copy(), "desired_pose": np.eye(4)}
    same_rotation = current.copy()
    same_rotation[:3, 3] = [-0.8, 0.2, -0.4]
    yield "equal orientations and different positions", {
        "pose": current.copy(), "desired_pose": same_rotation}
    same_position = desired.copy()
    same_position[:3, 3] = current[:3, 3]
    yield "equal positions and different orientations", {
        "pose": current.copy(), "desired_pose": same_position}
    yield "noncommuting pure rotations", {
        "pose": make_general_transform(np.array([0.6, 0.0, 0.0]), np.zeros(3)),
        "desired_pose": make_general_transform(np.array([0.0, -0.8, 0.0]), np.zeros(3)),
    }
    integer_pose = np.array([[0, -1, 0, 2], [1, 0, 0, -3],
                             [0, 0, 1, 1], [0, 0, 0, 1]])
    yield "integer-valued rigid transformation", {
        "pose": integer_pose, "desired_pose": np.eye(4, dtype=int)}
    yield "exact pi coordinate-axis rotation", {
        "pose": np.eye(4),
        "desired_pose": make_transform(np.pi, np.array([0.2, -0.3, 0.4])),
    }
    axis = np.array([1.0, -2.0, 3.0]) / np.sqrt(14)
    yield "exact pi mixed-axis rotation", {
        "pose": current.copy(),
        "desired_pose": make_general_transform(np.pi * axis, np.array([-0.2, 0.6, 0.1])),
    }
    yield "rotation beyond pi", {
        "pose": np.eye(4),
        "desired_pose": make_general_transform(
            (np.pi + 1e-3) * axis, np.array([0.1, -0.2, 0.3])),
    }
    yield "large translations with rotation", {
        "pose": make_general_transform(np.array([0.3, -0.5, 0.2]), np.array([1e3, -2e3, 3e3])),
        "desired_pose": make_general_transform(
            np.array([-0.4, 0.2, 0.6]), np.array([-4e3, 5e3, -6e3])),
    }
    yield "small resolvable rigid displacement", {
        "pose": np.eye(4),
        "desired_pose": make_general_transform(
            np.array([2e-4, -3e-4, 1e-4]), np.array([-4e-4, 2e-4, 3e-4])),
    }


def spatial_error_twist_cases():
    """Yield every public test case for ``spatial_error_twist``."""
    yield "identity error", {"error": np.eye(4)}
    yield "pure translation", {
        "error": make_transform(0.0, np.array([0.4, -0.6, 0.2]))}
    yield "tiny translation", {
        "error": make_transform(0.0, np.array([2e-9, -1e-9, 3e-9]))}
    yield "x-axis rotation", {
        "error": make_general_transform(np.array([0.7, 0.0, 0.0]), np.zeros(3))}
    yield "negative rotation", {"error": make_transform(-1.2, np.zeros(3))}
    yield "mixed-axis rotation", {
        "error": make_general_transform(np.array([0.5, -0.6, 0.3]), np.zeros(3))}
    yield "coupled rotation and translation", {
        "error": make_general_transform(
            np.array([-0.4, 0.7, 0.2]), np.array([0.6, -0.3, 0.8]))}
    yield "tiny rotation with translation", {
        "error": make_general_transform(
            np.array([1e-8, -2e-8, 3e-8]), np.array([0.2, -0.1, 0.4]))}
    axis = np.array([1.0, -2.0, 3.0]) / np.sqrt(14)
    for sign in (1, -1):
        yield f"near pi rotation, sign {sign}", {
            "error": make_general_transform(
                sign * (np.pi - 1e-3) * axis, np.array([0.3, -0.5, 0.7]))}
    yield "y-axis rotation", {
        "error": make_general_transform(np.array([0.0, 0.9, 0.0]), np.zeros(3))}
    yield "positive z-axis rotation", {"error": make_transform(0.8, np.zeros(3))}
    integer_error = np.eye(4, dtype=int)
    integer_error[:3, 3] = [2, -3, 4]
    yield "integer-valued translation", {"error": integer_error}
    coupled = make_general_transform(
        np.array([-0.4, 0.7, 0.2]), np.array([0.6, -0.3, 0.8]))
    yield "inverse coupled error", {"error": np.linalg.inv(coupled)}
    yield "large translation with rotation", {
        "error": make_general_transform(
            np.array([0.3, -0.5, 0.8]), np.array([1e3, -2e3, 3e3]))}
    yield "small resolvable pure rotation", {
        "error": make_general_transform(np.array([2e-4, -3e-4, 1e-4]), np.zeros(3))}
    yield "small resolvable rigid error", {
        "error": make_general_transform(
            np.array([-2e-5, 3e-5, 4e-5]), np.array([5e-5, -3e-5, 2e-5]))}
    yield "full-turn identity rotation", {"error": make_transform(2 * np.pi, np.zeros(3))}
    yield "principal branch beyond pi", {
        "error": make_general_transform(
            (np.pi + 1e-3) * axis, np.array([-0.2, 0.4, -0.6]))}
    for sign in (1, -1):
        yield f"closer to pi rotation, sign {sign}", {
            "error": make_general_transform(
                sign * (np.pi - 1e-5) * axis, np.array([0.3, -0.5, 0.7]))}



IK_CONVERGENCE_CASES = ("reachable translation", "reachable UR7e", "reachable Panda")


def jacobian_inverse_kinematics_cases():
    """Yield all public solver cases, including bounded and reachable tasks."""
    prism = np.vstack((np.eye(3), np.zeros((3, 3))))
    home = make_general_transform(np.array([0.2, -0.3, 0.4]), np.array([0.4, -0.2, 0.6]))
    target = home.copy()
    target[:3, 3] += [0.3, -0.2, 0.1]
    ready = np.array([0.0, -0.8, 1.4, -0.6, np.pi / 2, 0.0])
    ur_target = np.array([[0.029982003239722273,0.0008997300323979427,-0.9995500337489874,-0.6788774675562862],[-0.999100269967602,-0.029982003239722294,-0.029995500202495397,-0.1831601597876279],[-0.029995500202495647,0.9995500337489874,2.398651000552687e-16,0.15066303712816267],[0,0,0,1]])
    panda_target = np.array([[0.9739062404961993,0.10559724151512845,0.20088767335740523,0.49133556229286685],[0.1120324416303497,-0.9934845996392585,-0.02090651338613782,0.03523821836982802],[0.1973711395946687,0.042866920393447305,-0.9793911682219114,0.44713163407883627],[0,0,0,1]])
    revolute = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 1.0]]).T
    yield "already solved rotation", {
        "twists": revolute, "theta": np.array([0.4]), "g_st0": np.eye(4),
        "desired_pose": make_transform(0.4, np.zeros(3))}
    yield "already solved nonidentity home", {
        "twists": prism.copy(), "theta": np.zeros(3), "g_st0": home.copy(),
        "desired_pose": home.copy()}
    yield "zero iteration budget", {
        "twists": UR7E_TWISTS.copy(), "theta": ready.copy(), "g_st0": UR7E_ZERO_POSE.copy(),
        "desired_pose": ur_target.copy(), "max_iterations": 0}
    yield "zero step size", {
        "twists": UR7E_TWISTS.copy(), "theta": ready.copy(), "g_st0": UR7E_ZERO_POSE.copy(),
        "desired_pose": ur_target.copy(), "max_iterations": 3, "step_size": 0.0}
    yield "single joint update", {
        "twists": revolute, "theta": np.array([0.2]), "g_st0": np.eye(4),
        "desired_pose": make_transform(-0.5, np.zeros(3)), "max_iterations": 1}
    for sign in (1, -1):
        moved = home.copy()
        moved[:3, 3] += sign * np.array([0.3, -0.2, 0.1])
        yield f"one translation update, sign {sign}", {
            "twists": prism.copy(), "theta": np.array([-0.1, 0.05, 0.1]),
            "g_st0": home.copy(), "desired_pose": moved, "max_iterations": 1}
    yield "one UR7e update", {
        "twists": UR7E_TWISTS.copy(), "theta": ready.copy(), "g_st0": UR7E_ZERO_POSE.copy(),
        "desired_pose": ur_target.copy(), "max_iterations": 1}
    yield "one redundant update", {
        "twists": PANDA_TWISTS.copy(), "theta": PANDA_READY.copy(), "g_st0": PANDA_G_ST0.copy(),
        "desired_pose": panda_target.copy(), "max_iterations": 1}
    singular_target = UR7E_ZERO_POSE.copy()
    singular_target[:3, 3] += [0.02, -0.01, 0.03]
    yield "singular pseudoinverse update", {
        "twists": UR7E_TWISTS.copy(), "theta": np.zeros(6), "g_st0": UR7E_ZERO_POSE.copy(),
        "desired_pose": singular_target, "max_iterations": 1}
    yield "unreachable direction and finite budget", {
        "twists": np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0]]).T,
        "theta": np.zeros(1), "g_st0": np.eye(4),
        "desired_pose": make_transform(0.0, np.array([0.3, 0.0, 0.0])), "max_iterations": 3}
    yield "reachable translation", {
        "twists": prism.copy(), "theta": np.array([-0.1, 0.05, 0.1]),
        "g_st0": home.copy(), "desired_pose": target,
        "step_size": 0.4, "max_iterations": 100, "tolerance": 1e-7}
    yield "reachable UR7e", {
        "twists": UR7E_TWISTS.copy(), "theta": ready.copy(), "g_st0": UR7E_ZERO_POSE.copy(),
        "desired_pose": ur_target.copy(), "step_size": 0.3, "max_iterations": 150}
    yield "reachable Panda", {
        "twists": PANDA_TWISTS.copy(), "theta": PANDA_READY.copy(), "g_st0": PANDA_G_ST0.copy(),
        "desired_pose": panda_target.copy(), "max_iterations": 200}
    yield "already solved singular chain", {
        "twists": np.zeros((6, 2)), "theta": np.zeros(2), "g_st0": home.copy(),
        "desired_pose": home.copy()}
    name, twists, theta, initial = list(robot_cases())[5]
    yield name, {"twists": twists, "theta": theta, "g_st0": initial,
                 "desired_pose": home.copy(), "max_iterations": 3, "step_size": 0.1}


def euler_jacobian_cases():
    """Yield ZYX angular-Jacobian cases, including gimbal lock."""
    angles = (
        ("zero angles", [0, 0, 0]),
        ("three distinct angles", [0.3, -0.5, 0.7]),
        ("negative and positive angles", [-0.8, 0.4, -1.1]),
        ("positive gimbal lock", [0, np.pi / 2, 0]),
        ("negative gimbal lock", [1, -np.pi / 2, -0.4]),
        ("second positive gimbal lock", [0.4, 3 * np.pi / 2, 1.2]),
        ("below gimbal lock", [0.3, np.pi / 2 - 1e-5, -0.4]),
        ("above gimbal lock", [0.3, np.pi / 2 + 1e-5, -0.4]),
        ("tiny second angle", [0.3, 1e-8, -0.4]),
        ("second angle pi", [0.3, np.pi, -0.4]),
        ("different third angle", [0.3, -0.5, 2.0]),
        ("full-turn offsets", [0.3 + 2 * np.pi, -0.5 + 2 * np.pi, 0.7 + 2 * np.pi]),
        ("integer angles", [1, 2, 3]),
        ("third angle only", [0, 0, 1.2]),
    )
    for name, theta in angles:
        yield name, {"theta": np.array(theta)}


def manipulability_cases():
    """Yield full-rank, singular, scaled, and sign-changing 3x3 matrices."""
    matrices = (
        ("identity", np.eye(3)), ("zero", np.zeros((3, 3))),
        ("rank two", np.diag([1.0, 0.0, 2.0])),
        ("negative determinant", np.diag([-2.0, 3.0, 0.5])),
        ("axis permutation", np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])),
        ("rank one", np.outer([1.0, 2.0, -1.0], [0.5, -0.3, 0.7])),
        ("near singular", np.diag([1.0, 2.0, 1e-5])),
        ("uniform scaling", 3 * np.eye(3)),
        ("nonsymmetric", np.array([[1.0, 2.0, 3.0], [0.0, 2.0, 4.0], [0.0, 0.0, -0.5]])),
        ("integer entries", np.diag([2, 3, 4])),
        ("small scaling", 1e-3 * np.eye(3)),
        ("dependent rows", np.array([[1.0, 2.0, 3.0], [2.0, 4.0, 6.0], [0.0, 1.0, 1.0]])),
    )
    for name, jacobian in matrices:
        yield name, {"jacobian": jacobian}


CASE_FUNCTIONS = {
    "adjoint": adjoint_cases,
    "adjoint_inv": adjoint_inv_cases,
    "spatial_jacobian": spatial_jacobian_cases,
    "body_jacobian": body_jacobian_cases,
    "spatial_velocity": spatial_velocity_cases,
    "body_velocity": body_velocity_cases,
    "spatial_configuration_error": spatial_configuration_error_cases,
    "spatial_error_twist": spatial_error_twist_cases,
    "jacobian_inverse_kinematics": jacobian_inverse_kinematics_cases,
    "euler_jacobian": euler_jacobian_cases,
    "manipulability": manipulability_cases,
    "null_space_ik_step": null_space_ik_step_cases,
    "joint_limit_gradient": joint_limit_gradient_cases,
}

EXPECTED_RESULTS = {'adjoint': [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
             [[1.0, -0.0, 0.0, 0.0, -1.2, -0.7],
              [0.0, 1.0, 0.0, 1.2, 0.0, -0.4],
              [0.0, 0.0, 1.0, 0.7, 0.4, 0.0],
              [0.0, 0.0, 0.0, 1.0, -0.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
             [[0.8080275083121519,
               -0.5891447579422695,
               0.0,
               -0.29457237897113475,
               -0.40401375415607593,
               0.8],
              [0.5891447579422695,
               0.8080275083121519,
               0.0,
               0.40401375415607593,
               -0.29457237897113475,
               0.3],
              [0.0, 0.0, 1.0, -0.8231654340324024, 0.2289075538601701, 0.0],
              [0.0, 0.0, 0.0, 0.8080275083121519, -0.5891447579422695, 0.0],
              [0.0, 0.0, 0.0, 0.5891447579422695, 0.8080275083121519, 0.0],
              [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
             [[0.8984506673088334, -0.2417174114824961, -0.366550257670778, 0.0, 0.0, 0.0],
              [0.0987312821184971, 0.9246647910255665, -0.367759647281867, 0.0, 0.0, 0.0],
              [0.42783002739820614, 0.29422392360895316, 0.8546307684799344, 0.0, 0.0, 0.0],
              [0.0, 0.0, 0.0, 0.8984506673088334, -0.2417174114824961, -0.366550257670778],
              [0.0, 0.0, 0.0, 0.0987312821184971, 0.9246647910255665, -0.367759647281867],
              [0.0, 0.0, 0.0, 0.42783002739820614, 0.29422392360895316, 0.8546307684799344]],
             [[0.8509681221571013,
               -0.4719103270540472,
               0.23055129210689557,
               0.11280914137657917,
               -0.1050738761618016,
               -0.6314535439525328],
              [0.3908315608827724,
               0.862204844828389,
               0.3222630859598868,
               0.5285175779351837,
               0.0477384208044317,
               -0.7686938356671179],
              [-0.3508617193287871,
               -0.18412889174215943,
               0.918146287437602,
               0.8623292780887558,
               0.4928381641131218,
               0.42836755262803544],
              [0.0, 0.0, 0.0, 0.8509681221571013, -0.4719103270540472, 0.23055129210689557],
              [0.0, 0.0, 0.0, 0.3908315608827724, 0.862204844828389, 0.3222630859598868],
              [0.0, 0.0, 0.0, -0.3508617193287871, -0.18412889174215943, 0.918146287437602]]],
 'adjoint_inv': [[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                 [[1.0, 0.0, 0.0, 0.0, 0.9, -0.5],
                  [0.0, 1.0, 0.0, -0.9, 0.0, -0.2],
                  [0.0, 0.0, 1.0, 0.5, 0.2, 0.0],
                  [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                 [[0.8869949227792843,
                   -0.4617791755414829,
                   0.0,
                   0.18471167021659318,
                   0.3547979691117137,
                   -0.18836801304696135],
                  [0.46177917554148296,
                   0.8869949227792842,
                   0.0,
                   -0.35479796911171374,
                   0.18471167021659318,
                   0.5783748712217188],
                  [0.0, 0.0, 1.0, -0.1, -0.6, 0.0],
                  [0.0, 0.0, 0.0, 0.8869949227792843, -0.4617791755414829, 0.0],
                  [0.0, 0.0, 0.0, 0.46177917554148296, 0.8869949227792842, 0.0],
                  [0.0, 0.0, 0.0, 0.0, 0.0, 1.0]],
                 [[0.8882835738447966, 0.24003828993835358, -0.3915787427848795, 0.0, 0.0, 0.0],
                  [-0.31435293665866393, 0.9393260022500626, -0.13729126960950516, 0.0, 0.0, 0.0],
                  [0.3348649334456953, 0.2450475073539553, 0.9098445996366761, 0.0, 0.0, 0.0],
                  [0.0, 0.0, 0.0, 0.8882835738447966, 0.24003828993835358, -0.3915787427848795],
                  [0.0, 0.0, 0.0, -0.31435293665866393, 0.9393260022500626, -0.13729126960950516],
                  [0.0, 0.0, 0.0, 0.3348649334456953, 0.2450475073539553, 0.9098445996366761]],
                 [[0.9344288077728644,
                   -0.3455881931260571,
                   0.08609067299104099,
                   0.20729164081753967,
                   0.5785038490183563,
                   0.07230538328997],
                  [0.281625810819736,
                   0.8649574702831482,
                   0.41537390057814555,
                   -0.3926518285401025,
                   0.46642462138446406,
                   -0.7050432648763084],
                  [-0.2180130864978284,
                   -0.36389198311208226,
                   0.9055677328297683,
                   0.38125413727759894,
                   0.5592686020485206,
                   0.31652160463362733],
                  [0.0, 0.0, 0.0, 0.9344288077728644, -0.3455881931260571, 0.08609067299104099],
                  [0.0, 0.0, 0.0, 0.281625810819736, 0.8649574702831482, 0.41537390057814555],
                  [0.0, 0.0, 0.0, -0.2180130864978284, -0.36389198311208226, 0.9055677328297683]]],
 'spatial_jacobian': [[[0.0], [0.0], [0.0], [0.0], [0.0], [1.0]],
                      [[0.0, 0.0],
                       [0.0, 0.0],
                       [0.0, 1.0],
                       [0.0, -0.3428978074554513],
                       [0.0, 0.9393727128473789],
                       [1.0, 0.0]],
                      [[0.0, 0.0, 0.8924274382425491],
                       [0.0, 0.0, 0.3773122691048195],
                       [0.0, 1.0, 0.2474039592545229],
                       [0.0, -0.3894183423086505, 0.0],
                       [0.0, 0.9210609940028851, 0.0],
                       [1.0, 0.0, 0.0]],
                      [[0.0, 0.0, 1.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0],
                       [1.0, 0.0, 0.0]],
                      [[0.0, 0.0, -1.4523083793848723, -0.19045870494943115],
                       [0.0, 0.0, -0.3247643596647827, 0.8517091389287839],
                       [0.0, -1.0, -4.1435110494381405e-18, -0.4881772468829075],
                       [0.0, 0.9758974493306055, -0.19045870494943115, 0.0],
                       [0.0, 0.21822962308086932, 0.8517091389287839, 0.0],
                       [1.0, 0.0, -0.4881772468829075, 0.0]],
                      [[0.2, -0.2562110931934241, 0.28818610150498153],
                       [-0.1, 0.31193574239640637, -0.43500312908255995],
                       [0.4, 0.19701600126563362, 0.13312042888956702],
                       [0.3, -0.49531030204352167, 0.0],
                       [0.8, 0.2291838288386453, 0.0],
                       [-0.5, 0.7695079449157194, 0.0]],
                      [[0.0, -0.333, 0.0, 0.659266747613252, 0.0, 0.6131933111718786, 0.0],
                       [0.0,
                        0.0,
                        -0.09840822881822608,
                        0.0,
                        0.7016541349804644,
                        0.0,
                        0.5228228378002562],
                       [0.0, 0.0, 0.0, 0.01456912495212083, 0.0, -0.37548149798608454, 0.0],
                       [0.0,
                        0.0,
                        -0.29552020666133966,
                        0.0,
                        0.946300087687413,
                        0.0,
                        0.09983341664682789],
                       [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
                       [1.0,
                        0.0,
                        0.955336489125606,
                        0.0,
                        -0.323289566863503,
                        0.0,
                        -0.9950041652780242]],
                      [[0.0, -0.333, 0.0, 0.649, 0.0, 1.033, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.088],
                       [0.0, 0.0, 0.0, -0.0825, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
                       [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, -1.0]],
                      [[0.0, -0.333, 0.0, 0.649, 0.0, 1.0330082480799863, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 6.490041239183301e-05, 0.0, 0.08793509958760816],
                       [0.0, 0.0, 0.0, -0.0825, 0.0, -3.8400412435999644e-05, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 9.999999983333336e-05, 0.0, -9.999999983333336e-05],
                       [0.0, 1.0, 0.0, -1.0, 0.0, -1.0, 0.0],
                       [1.0, 0.0, 1.0, 0.0, 0.999999995, 0.0, -0.999999995]]],
 'body_jacobian': [[[0.0], [0.5], [0.0], [0.0], [0.0], [1.0]],
                   [[0.3636698545073377, 0.3920266311364964],
                    [0.5477442585759713, -0.07946773231802441],
                    [-0.1693927420185105, 0.1999999999999999],
                    [0.5533872166040864, 0.19866933079506124],
                    [-0.11217714232785979, 0.9800665778412417],
                    [0.8253356149096784, 3.320618680663827e-17]],
                   [[-0.4075348157390566, 0.9553364891256063, 0.9553364891256061],
                    [0.6617158194754819, 0.2955202066613396, 0.2955202066613396],
                    [0.04948079185090463, -2.2466232380769025e-16, -9.020588961312932e-17],
                    [0.23635402982999043, -0.2955202066613396, 0.0],
                    [0.07311286916773023, 0.9553364891256062, 0.0],
                    [0.9689124217106447, 1.8413419812565367e-17, 0.0]],
                   [[-0.7183197525759049, 0.7898243057013355, 0.8775825618903726],
                    [-0.06337770159362728, -0.43148298474378266, -0.479425538604203],
                    [0.0, 1.4, 0.0],
                    [0.0, 0.47942553860420295, 0.0],
                    [0.0, 0.8775825618903728, 0.0],
                    [1.0, 0.0, 0.0]],
                   [[-0.014518225866799453,
                     0.035208974081264055,
                     0.008356495539636435,
                     0.06803131640494],
                    [0.6055928941930588,
                     -1.2992484444816679,
                     -0.1337909610001649,
                     0.9752903089530458],
                    [0.18226917250423919,
                     -1.4046838465464246,
                     -0.6180858740797663,
                     -0.21019170595074294],
                    [-0.08672119877333906, 0.9957974482903429, 0.06803131640494, 0.0],
                    [-0.2890255216007936, -0.053431428415994825, 0.9752903089530458, 0.0],
                    [0.953385379343896, 0.07438094134836017, -0.21019170595074294, 0.0]],
                   [[0.43750619602229585, -0.19645760339586124, 0.38802376644480874],
                    [-0.3431056735571674, 1.22191751598905, -0.018124790105764472],
                    [0.09922543061037696, -0.13572764709926638, 0.37297325461433084],
                    [0.033080004532710344, -0.7952119765565896, 0.0],
                    [0.4249942225215568, 0.20991574282910788, 0.0],
                    [-0.8934683117623216, 0.4621399065796693, 0.0]],
                   [[4.683576360777408e-17,
                     0.030907910867441934,
                     4.713820846609099e-17,
                     0.29518334752098335,
                     2.0159109857174607e-17,
                     0.21039999999999953,
                     7.011087588369713e-18],
                    [-0.48404681539304467,
                     5.666629661355687e-18,
                     -0.4859597928910379,
                     3.053677265782641e-17,
                     -0.15469525698797273,
                     2.1123930205567168e-17,
                     1.357481464804738e-15],
                    [-6.927334861769064e-19,
                     0.48957831005686375,
                     9.672474836453496e-19,
                     -0.47150232591620067,
                     -3.3769496929639446e-18,
                     -0.0879999999999998,
                     2.1851419414444446e-18],
                    [0.09983341664682818,
                     1.1116038687513694e-16,
                     -0.19866933079506163,
                     -1.1116038687513694e-16,
                     0.9092974268256814,
                     -1.1116038687513694e-16,
                     -7.537614950993347e-18],
                    [1.1083735778830504e-17,
                     -1.0,
                     -2.2056726533554908e-17,
                     1.0,
                     1.0095229394942799e-16,
                     1.0,
                     -8.368433669351635e-34],
                    [-0.9950041652780275,
                     4.179507442020495e-18,
                     -0.9800665778412433,
                     -4.179507442020495e-18,
                     0.4161468365471424,
                     -4.179507442020495e-18,
                     1.0]],
                   [[0.06222539674441617,
                     0.34619948006893364,
                     0.06222539674441617,
                     -0.12275373721398464,
                     0.06222539674441617,
                     0.14877526676164954,
                     0.0],
                    [-0.06222539674441617,
                     0.34619948006893364,
                     -0.06222539674441617,
                     -0.12275373721398464,
                     -0.06222539674441617,
                     0.14877526676164954,
                     0.0],
                    [0.0,
                     0.08799999999999998,
                     0.0,
                     -0.005499999999999977,
                     0.0,
                     -0.08799999999999998,
                     0.0],
                    [0.0,
                     0.7071067811865475,
                     0.0,
                     -0.7071067811865475,
                     0.0,
                     -0.7071067811865475,
                     0.0],
                    [0.0,
                     -0.7071067811865475,
                     0.0,
                     0.7071067811865475,
                     0.0,
                     0.7071067811865475,
                     0.0],
                    [-1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]],
                   [[0.062237672098671676,
                     0.34620531258264,
                     0.062237672098671676,
                     -0.12275373721398464,
                     0.06222539674441618,
                     0.14877526676164954,
                     -1.2006962981936428e-18],
                    [-0.062237672098671676,
                     0.34620531258264,
                     -0.062237672098671676,
                     -0.12275373721398464,
                     -0.06222539674441618,
                     0.14877526676164954,
                     1.2006962981936428e-18],
                    [3.987311818139641e-22,
                     0.08796839958755265,
                     3.987311818139641e-22,
                     -0.005499999999999977,
                     6.652223817627119e-22,
                     -0.08799999999999998,
                     -1.9129180290491145e-21],
                    [-7.071067800080363e-05,
                     0.7071067811865475,
                     -7.071067800080363e-05,
                     -0.7071067811865475,
                     -7.472330495012297e-21,
                     -0.7071067811865475,
                     7.472330495012297e-21],
                    [-7.071067800080363e-05,
                     -0.7071067811865475,
                     -7.071067800080363e-05,
                     0.7071067811865475,
                     -7.472330495012297e-21,
                     0.7071067811865475,
                     7.472330495012297e-21],
                    [-0.999999995,
                     -1.4178359628254577e-20,
                     -0.999999995,
                     1.4178359628254577e-20,
                     -1.0,
                     1.4178359628254577e-20,
                     1.0]]],
 'spatial_velocity': [[0.0, 0.0, 0.0, 0.0, 0.0, 0.45],
                      [0.0, 0.0, 0.7, -0.24002846521881588, 0.6575608989931652, -0.2],
                      [0.7139419505940393,
                       0.3018498152838556,
                       -0.3020768325963817,
                       0.19470917115432526,
                       -0.46053049700144255,
                       0.3],
                      [0.2, 0.0, 0.6, 0.0, 0.6, -0.4],
                      [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                      [0.23213776856001986,
                       -0.2875819743519459,
                       0.07414337117613674,
                       0.2085930906130565,
                       0.09124485134840644,
                       -0.3308523834747158],
                      [-0.13578736792811294,
                       0.05852607457059943,
                       -0.04337579977945679,
                       0.09062061387439793,
                       0.1,
                       0.4214434498927836],
                      [-0.1383, 0.0352, -0.00825, 0.0, 0.3, -0.5],
                      [0.6412032992319945,
                       -0.026393509958760813,
                       -0.024765360164974402,
                       9.999999983333332e-06,
                       -0.8,
                       0.3999999995]],
 'body_velocity': [[0.0, 0.225, 0.0, 0.0, 0.0, 0.45],
                   [0.20168467089407988,
                    -0.16517626433781135,
                    0.17387854840370204,
                    0.028391088235725573,
                    0.7084820329544412,
                    -0.16506712298193568],
                   [0.16434050201596473,
                    0.2871708078410464,
                    0.014844237555271428,
                    0.21866631227966693,
                    -0.45573438381248405,
                    0.2906737265131934],
                   [0.9367389968292378,
                    -0.32942381792965925,
                    0.84,
                    0.28765532316252174,
                    0.5265495371342236,
                    -0.4],
                   [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                   [0.30164802680114106,
                    -0.4424463055504543,
                    0.20975268209758763,
                    0.24517959387351895,
                    0.022024121655579,
                    -0.31733563432636513],
                   [-0.10321492118188176,
                    -0.2251316708042105,
                    0.08188526835510755,
                    0.13224202779130062,
                    -0.09999999999999999,
                    -0.41029102257074723],
                   [0.07048440394867506,
                    0.08292948329755828,
                    0.03464999999999999,
                    0.21213203435596423,
                    -0.21213203435596423,
                    0.5],
                   [-0.0057103234370813,
                    -0.018162767998517834,
                    -0.04564683995875525,
                    -0.5657066381526382,
                    0.5656642117458378,
                    -0.3999999985]],
 'jacobian_ik_step': [[0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.7853981633974483],
                      [0.0058954741048670095,
                       -0.24750948635768202,
                       0.010094002275509469,
                       -2.160680677160238,
                       0.001558129730360636,
                       2.013171190802556,
                       0.8005085681967404],
                      [0.19417979824292597,
                       -0.5756900403175895,
                       0.3038504099970071,
                       -1.7670787532128118,
                       -0.1970043055903502,
                       1.3809581254526817,
                       0.5044980629382134],
                      [0.024105912994996065,
                       0.06045994483531492,
                       0.02410591299499607,
                       0.10003994229172143,
                       0.024105912994995718,
                       -0.04806527883064503,
                       0.06631773898498786],
                      [-0.40694881131213595,
                       -0.8124432504265792,
                       -0.19828814216447316,
                       -2.4053862596308186,
                       -0.29587801064503594,
                       1.1124309072116296,
                       -0.4831918898222416],
                      [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.7853981633974483],
                      [-3.5355339118269753,
                       0.019545443183196842,
                       -3.5355339118226556,
                       0.038533864658160145,
                       7.0965686632669005,
                       -0.016967101131403015,
                       0.026500874972610167],
                      [0.011943265687254767,
                        -0.7807666017130422,
                        1.3810934015559049,
                        -0.5983267998428626,
                        1.5767395924821512,
                        -0.003999999999999998],
                       [0.0, -0.8, 1.4, -0.6, 1.5707963267948966, 0.0],
                       [0.0, -0.8, 1.4, -0.6, 1.5707963267948966, 0.0],
                       [0.003575927447937172,
                        1.3634713720724357e-05,
                        -1.645964787728433e-05,
                        -4.423143286255494e-05,
                        1.574364586202898,
                        0.0],
                       [0.0036710719713636494,
                        5.757004642332688,
                        -11.995480555110328,
                        6.238475912777637,
                        1.574467398766261,
                        5.115620996361075e-19]],
 'null_space_ik_step': [[0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.7853981633974483],
                        [0.004798346275244855,
                         -0.2594449227344588,
                         0.001420111808265291,
                         -2.1788526982903003,
                         0.006381957309555274,
                         2.0214077755558413,
                         0.7789085107024915],
                        [0.0008486065542102341,
                         -0.3,
                         -0.0007625699799619024,
                         -2.2,
                         -0.00025978140081597205,
                         2.0,
                         0.7856032683110873],
                        [0.24038890807485358,
                         -0.6724181020740437,
                         0.4058669856687662,
                         -1.9691310693894624,
                         0.203416328189936,
                         1.6107638793387196,
                         -0.2949132673684428],
                        [0.01589648693066456,
                         0.011843807401767928,
                         0.012056486930664535,
                         0.022755232331477637,
                         0.01061648693066441,
                         -0.0168511218916767,
                         0.0433694607919935],
                        [0.0, -0.3, 0.0, -2.2, 0.0, 2.0, 0.7853981633974483],
                        [-0.003029593317197137,
                         -0.2875168683819748,
                         -0.006282714213791768,
                         -2.1982765296199838,
                         0.003358936177943332,
                         2.014759661238009,
                         0.7788284165440583],
                        [-14.143135647307865,
                         0.017342751216754687,
                         -14.141135647290623,
                         0.0241490671283279,
                         28.289851001087584,
                         -0.006906315911570628,
                         0.007579847910456666]],
 'joint_limit_gradient': [[-0.0, -0.0, -0.0, -0.0, -0.0, -0.0, -0.0],
                          [-0.13805957270562247,
                           -0.2269117313365101,
                           -0.13805957270562247,
                           -0.2664890073284477,
                           -0.13805957270562247,
                           -0.2122015915119364,
                           -0.13805957270562247],
                          [0.12080212611741965,
                           0.19854776491944634,
                           0.12080212611741965,
                           0.2331778814123916,
                           0.12080212611741965,
                           0.18567639257294433,
                           0.12080212611741965],
                          [-0.06902978635281123,
                           0.2269117313365101,
                           -0.2070893590584337,
                           0.1998667554963358,
                           -0.03451489317640562,
                           -0.26525198938992045,
                           0.06902978635281123],
                          [-0.0,
                           0.09654175090899851,
                           -0.0,
                           0.27927195771329,
                           -0.0,
                           -0.037290067473914564,
                           -0.0935627436254723],
                          [0.34514893176405614,
                           0.5672793283412753,
                           0.34514893176405614,
                           0.6662225183211192,
                           0.34514893176405614,
                           0.5305039787798409,
                           0.34514893176405614],
                          [-0.34514893176405614,
                           -0.5672793283412753,
                           -0.34514893176405614,
                           -0.6662225183211191,
                           -0.34514893176405614,
                           -0.5305039787798409,
                           -0.34514893176405614]]}


EXPECTED_RESULTS.update({
  "spatial_configuration_error": [
    [
      [
        1,
        0,
        0,
        0
      ],
      [
        0,
        1,
        0,
        0
      ],
      [
        0,
        0,
        1,
        0
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        0.9999999999999999,
        0,
        0,
        1.1102230246251565e-16
      ],
      [
        0,
        0.9999999999999999,
        -5.551115123125783e-17,
        8.326672684688674e-17
      ],
      [
        -5.551115123125783e-17,
        -5.551115123125783e-17,
        1,
        -1.1102230246251565e-16
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        1,
        0,
        0,
        -0.8
      ],
      [
        0,
        1,
        0,
        0.8999999999999999
      ],
      [
        0,
        0,
        1,
        -0.30000000000000004
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        0.9023934261437778,
        -0.24903648038416887,
        0.35166309998400436,
        0
      ],
      [
        0.13190859175670216,
        0.9365557269934556,
        0.32475143364814213,
        0
      ],
      [
        -0.4102270442977377,
        -0.24666617456316434,
        0.8779917826797222,
        0
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        0.9365557269934554,
        0.1319085917567021,
        0.32475143364814213,
        -0.5557802872362358
      ],
      [
        -0.2490364803841688,
        0.9023934261437776,
        0.3516630999840043,
        0.7192881213151867
      ],
      [
        -0.24666617456316434,
        -0.41022704429773765,
        0.8779917826797222,
        -1.0595072435547481
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        0.7559499686768936,
        -0.3678645361084151,
        0.541493608393621,
        -1.4706971359752405
      ],
      [
        -0.07940073791204869,
        0.7695441785092207,
        0.6336381302775194,
        0.625502027779824
      ],
      [
        -0.649796250894406,
        -0.5219937168168876,
        0.5525281820208497,
        -0.1131416534884801
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        0.7559499686768933,
        -0.07940073791204871,
        -0.649796250894406,
        1.0879197541880727
      ],
      [
        -0.3678645361084151,
        0.7695441785092207,
        -0.5219937168168878,
        -1.081427996036443
      ],
      [
        0.541493608393621,
        0.6336381302775194,
        0.5525281820208497,
        0.46254511575901497
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        1,
        -1.000000003e-9,
        -2.999999999e-9,
        -4e-9
      ],
      [
        9.99999997e-10,
        1,
        -2.0000000015e-9,
        2e-9
      ],
      [
        3.000000001e-9,
        1.9999999985000003e-9,
        1,
        3e-9
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        0.9365557269934554,
        0.1319085917567021,
        0.32475143364814213,
        -0.7557802872362358
      ],
      [
        -0.2490364803841688,
        0.9023934261437776,
        0.3516630999840043,
        0.3192881213151867
      ],
      [
        -0.24666617456316434,
        -0.41022704429773765,
        0.8779917826797222,
        -0.7595072435547481
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        0.9999999999999999,
        0,
        0,
        -1.4
      ],
      [
        0,
        0.9999999999999999,
        -5.551115123125783e-17,
        0.7000000000000001
      ],
      [
        -5.551115123125783e-17,
        -5.551115123125783e-17,
        1,
        -1.2000000000000002
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        0.7559499686768936,
        -0.3678645361084151,
        0.541493608393621,
        -0.47069713597524054
      ],
      [
        -0.07940073791204869,
        0.7695441785092207,
        0.6336381302775194,
        -0.574497972220176
      ],
      [
        -0.649796250894406,
        -0.5219937168168876,
        0.5525281820208497,
        0.4868583465115199
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        0.6967067093471655,
        0.4050497174705004,
        -0.5920595303917608,
        0
      ],
      [
        0,
        0.8253356149096783,
        0.5646424733950354,
        0
      ],
      [
        0.7173560908995228,
        -0.3933901995966995,
        0.5750168603707414,
        0
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        0,
        1,
        0,
        3
      ],
      [
        -1,
        0,
        0,
        2
      ],
      [
        0,
        0,
        1,
        -1
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        -1,
        -1.2246467991473532e-16,
        0,
        0.2
      ],
      [
        1.2246467991473532e-16,
        -1,
        0,
        -0.3
      ],
      [
        0,
        0,
        1,
        0.4
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        -0.8373228464116982,
        -0.5467027908172827,
        -0.002551350545384759,
        0.03108339287468531
      ],
      [
        0.05057071922065424,
        -0.0728050280226159,
        -0.9960632661895183,
        1.3301056674079144
      ],
      [
        0.5443648163084355,
        -0.8341555528840048,
        0.08860846662854915,
        -0.714583439529903
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        -0.8571423928571814,
        -0.2849124306936134,
        0.4291058438233182,
        0.1
      ],
      [
        -0.286515997877827,
        -0.4285710714286012,
        -0.8568753816597916,
        -0.2
      ],
      [
        0.4280367990338424,
        -0.8574099040545295,
        0.28571446428569935,
        0.3
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        0.8177762869660169,
        -0.4761838654511153,
        0.3232504768134279,
        -6739.895448308531
      ],
      [
        0.05528073717919049,
        0.6240428108560245,
        0.7794322358716226,
        3854.5081769179906
      ],
      [
        -0.5728751910957566,
        -0.6195316751409453,
        0.5366512078846504,
        -8276.141782840084
      ],
      [
        0,
        0,
        0,
        1
      ]
    ],
    [
      [
        0.9999999500000007,
        -0.0001000299976663167,
        -0.0002999899930001167,
        -0.0004
      ],
      [
        0.00009996999766701669,
        0.9999999750000004,
        -0.00020001499533315838,
        0.0002
      ],
      [
        0.0003000099929998834,
        0.00019998499533350838,
        0.999999935000001,
        0.0003
      ],
      [
        0,
        0,
        0,
        1
      ]
    ]
  ],
  "spatial_error_twist": [
    [
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      0.4,
      -0.6,
      0.2,
      0,
      0,
      0
    ],
    [
      2e-9,
      -1e-9,
      3e-9,
      0,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0.7,
      0,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      -1.2000000000000002
    ],
    [
      0,
      0,
      0,
      0.49999999999999983,
      -0.5999999999999995,
      0.2999999999999999
    ],
    [
      0.2648763092396157,
      -0.5196627692515388,
      0.8985723108596166,
      -0.40000000000000024,
      0.6999999999999998,
      0.19999999999999973
    ],
    [
      0.20000000250000002,
      -0.10000000099999988,
      0.39999999849999995,
      1.0000000020613649e-8,
      -2.0000000006554904e-8,
      2.9999999981470444e-8
    ],
    [
      0.20093407382328948,
      -0.5696613714115168,
      0.6865810611178927,
      0.8393586929397326,
      -1.6787173858779698,
      2.518076078819313
    ],
    [
      0.2848699431171791,
      -0.4017896328236028,
      0.7705169304118723,
      -0.8393586929394891,
      1.6787173858785844,
      -2.518076078818702
    ],
    [
      0,
      0,
      0,
      0,
      0.8999999999999999,
      0
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0.8000000000000005
    ],
    [
      2,
      -3,
      4,
      0,
      0,
      0
    ],
    [
      -0.264876309239616,
      0.519662769251539,
      -0.8985723108596171,
      0.4,
      -0.7000000000000004,
      -0.19999999999999965
    ],
    [
      961.014510201684,
      -1940.680029829344,
      3051.6945400310283,
      0.3000000000000005,
      -0.49999999999999994,
      0.8000000000000002
    ],
    [
      0,
      0,
      0,
      0.00020000000000046296,
      -0.00029999999999988597,
      0.00010000000000017238
    ],
    [
      0.00004999909998975002,
      -0.000030001199995500005,
      0.000020000449991500003,
      -0.00001999999999985763,
      0.000029999999999824187,
      0.00003999999999986871
    ],
    [
      0,
      0,
      0,
      0,
      0,
      0
    ],
    [
      -0.20000000000000057,
      0.4000000000000003,
      -0.6000000000000005,
      -0.8393586929393975,
      1.6787173858784659,
      -2.5180760788189387
    ],
    [
      0.2008764275778005,
      -0.5696767260694777,
      0.6865900400944156,
      0.8396232815985503,
      -1.6792465631288975,
      2.518869844693122
    ],
    [
      0.2848387557305701,
      -0.4017520697582697,
      0.7705523682509635,
      -0.8396232816138132,
      1.6792465631199622,
      -2.518869844691805
    ]
  ]
})


EXPECTED_RESULTS.update({
  "jacobian_inverse_kinematics": [
    [
      [
        0.4
      ],
      [
        [
          0.4
        ]
      ]
    ],
    [
      [
        0.0,
        0.0,
        0.0
      ],
      [
        [
          0.0,
          0.0,
          0.0
        ]
      ]
    ],
    [
      [
        0.0,
        -0.8,
        1.4,
        -0.6,
        1.5707963267948966,
        0.0
      ],
      [
        [
          0.0,
          -0.8,
          1.4,
          -0.6,
          1.5707963267948966,
          0.0
        ]
      ]
    ],
    [
      [
        0.0,
        -0.8,
        1.4,
        -0.6,
        1.5707963267948966,
        0.0
      ],
      [
        [
          0.0,
          -0.8,
          1.4,
          -0.6,
          1.5707963267948966,
          0.0
        ],
        [
          0.0,
          -0.8,
          1.4,
          -0.6,
          1.5707963267948966,
          0.0
        ],
        [
          0.0,
          -0.8,
          1.4,
          -0.6,
          1.5707963267948966,
          0.0
        ],
        [
          0.0,
          -0.8,
          1.4,
          -0.6,
          1.5707963267948966,
          0.0
        ]
      ]
    ],
    [
      [
        0.05999999999999997
      ],
      [
        [
          0.2
        ],
        [
          0.05999999999999997
        ]
      ]
    ],
    [
      [
        -0.01999999999999999,
        6.938893903907228e-18,
        0.1
      ],
      [
        [
          -0.1,
          0.05,
          0.1
        ],
        [
          -0.01999999999999999,
          6.938893903907228e-18,
          0.1
        ]
      ]
    ],
    [
      [
        -0.14,
        0.08000000000000002,
        0.06000000000000001
      ],
      [
        [
          -0.1,
          0.05,
          0.1
        ],
        [
          -0.14,
          0.08000000000000002,
          0.06000000000000001
        ]
      ]
    ],
    [
      [
        0.01529727207227702,
        -0.8130627239629767,
        1.4218271751698164,
        -0.6088544512066377,
        1.5800940488874236,
        -0.0059995499797499555
      ],
      [
        [
          0.0,
          -0.8,
          1.4,
          -0.6,
          1.5707963267948966,
          0.0
        ],
        [
          0.01529727207227702,
          -0.8130627239629767,
          1.4218271751698164,
          -0.6088544512066377,
          1.5800940488874236,
          -0.0059995499797499555
        ]
      ]
    ],
    [
      [
        0.007223772905546735,
        -0.3074869331056586,
        0.008383491559169195,
        -2.1950241884636927,
        -0.004814658131492764,
        2.007741979924546,
        0.7805816006939597
      ],
      [
        [
          0.0,
          -0.3,
          0.0,
          -2.2,
          0.0,
          2.0,
          0.7853981633974483
        ],
        [
          0.007223772905546735,
          -0.3074869331056586,
          0.008383491559169195,
          -2.1950241884636927,
          -0.004814658131492764,
          2.007741979924546,
          0.7805816006939597
        ]
      ]
    ],
    [
      [
        0.002447381302006857,
        -0.013564561066967134,
        0.012965220050804593,
        0.03744752629642825,
        0.0024473813020068555,
        -0.036848185280265705
      ],
      [
        [
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0
        ],
        [
          0.002447381302006857,
          -0.013564561066967134,
          0.012965220050804593,
          0.03744752629642825,
          0.0024473813020068555,
          -0.036848185280265705
        ]
      ]
    ],
    [
      [
        0.0
      ],
      [
        [
          0.0
        ],
        [
          0.0
        ],
        [
          0.0
        ],
        [
          0.0
        ]
      ]
    ],
    [],
    [],
    [],
    [
      [
        0.0,
        0.0
      ],
      [
        [
          0.0,
          0.0
        ]
      ]
    ],
    [
      [
        0.0024450601463884836,
        -0.32502690241218757,
        0.3006080143741459
      ],
      [
        [
          0.31,
          -0.42,
          0.2
        ],
        [
          0.19634354854674246,
          -0.38397856015715726,
          0.23734584991670066
        ],
        [
          0.09417725417467802,
          -0.3523951056945343,
          0.27098116169825587
        ],
        [
          0.0024450601463884836,
          -0.32502690241218757,
          0.3006080143741459
        ]
      ]
    ]
  ],
  "euler_jacobian": [
    [
      [
        0.0,
        0.0,
        1.0
      ],
      [
        0.0,
        1.0,
        0.0
      ],
      [
        1.0,
        0.0,
        0.0
      ]
    ],
    [
      [
        0.0,
        -0.29552020666133966,
        0.8383866435942036
      ],
      [
        0.0,
        0.955336489125606,
        0.2593433800522308
      ],
      [
        1.0,
        0.0,
        0.47942553860420306
      ]
    ],
    [
      [
        0.0,
        0.7173560908995228,
        0.6417093742397794
      ],
      [
        0.0,
        0.6967067093471654,
        -0.6607287141379384
      ],
      [
        1.0,
        0.0,
        -0.3894183423086506
      ]
    ],
    [
      [
        0.0,
        0.0,
        0.0
      ],
      [
        0.0,
        1.0,
        0.0
      ],
      [
        1.0,
        0.0,
        -1.0
      ]
    ],
    [
      [
        0.0,
        -0.8414709848078966,
        0.0
      ],
      [
        0.0,
        0.5403023058681397,
        0.0
      ],
      [
        1.0,
        0.0,
        1.0
      ]
    ],
    [
      [
        0.0,
        -0.3894183423086505,
        -1.6823048309138896e-16
      ],
      [
        0.0,
        0.9210609940028851,
        -7.112670743608426e-17
      ],
      [
        1.0,
        0.0,
        0.9999999999999962
      ]
    ],
    [
      [
        0.0,
        -0.29552020666133966,
        9.553364891212582e-06
      ],
      [
        0.0,
        0.955336489125606,
        2.955202066599947e-06
      ],
      [
        1.0,
        0.0,
        -0.9999999999499999
      ]
    ],
    [
      [
        0.0,
        -0.29552020666133966,
        -9.553364891106518e-06
      ],
      [
        0.0,
        0.955336489125606,
        -2.9552020665671374e-06
      ],
      [
        1.0,
        0.0,
        -0.9999999999500001
      ]
    ],
    [
      [
        0.0,
        -0.29552020666133966,
        0.955336489125606
      ],
      [
        0.0,
        0.955336489125606,
        0.2955202066613396
      ],
      [
        1.0,
        0.0,
        -1e-08
      ]
    ],
    [
      [
        0.0,
        -0.29552020666133966,
        -0.9553364891256058
      ],
      [
        0.0,
        0.955336489125606,
        -0.29552020666133955
      ],
      [
        1.0,
        0.0,
        -1.526951044735943e-16
      ]
    ],
    [
      [
        0.0,
        -0.29552020666133966,
        0.8383866435942036
      ],
      [
        0.0,
        0.955336489125606,
        0.2593433800522308
      ],
      [
        1.0,
        0.0,
        0.47942553860420306
      ]
    ],
    [
      [
        0.0,
        -0.29552020666134066,
        0.8383866435942041
      ],
      [
        0.0,
        0.9553364891256114,
        0.2593433800522304
      ],
      [
        1.0,
        0.0,
        0.4794255386042009
      ]
    ],
    [
      [
        0.0,
        -0.8414709848078966,
        -0.22484509536615283
      ],
      [
        0.0,
        0.5403023058681397,
        -0.35017548837401463
      ],
      [
        1.0,
        0.0,
        -0.9092974268256817
      ]
    ],
    [
      [
        0.0,
        0.0,
        1.0
      ],
      [
        0.0,
        1.0,
        0.0
      ],
      [
        1.0,
        0.0,
        0.0
      ]
    ]
  ],
  "manipulability": [
    1.0,
    0.0,
    0.0,
    3.0,
    1.0,
    6.578507617809082e-50,
    2e-05,
    27.0,
    0.9999999999999996,
    24.0,
    1e-09,
    2.3332081284731083e-15
  ],
  "null_space_ik_step": [
    [
      0.0,
      -0.3,
      0.0,
      -2.2,
      0.0,
      2.0,
      0.7853981633974483
    ],
    [
      -0.009128965357145653,
      -0.26188483266084656,
      -0.015196021776010213,
      -2.144723643894964,
      0.0031497759322440424,
      1.980838811234117,
      0.7706599646462199
    ],
    [
      0.0008486065542102389,
      -0.3,
      -0.0007625699799619024,
      -2.2,
      -0.0002597814008159711,
      2.0,
      0.7856032683110873
    ],
    [
      0.2368358631568998,
      -0.644181774688841,
      0.4304795382379601,
      -1.9823368136236947,
      0.19797371133063416,
      1.625352967216697,
      -0.2879807087210051
    ],
    [
      -0.008996363636363615,
      -0.01778587012987012,
      -0.012836363636363623,
      -0.029090909090909084,
      -0.014276363636363617,
      0.013705038961038968,
      -0.04090909090909085
    ],
    [
      0.0,
      -0.3,
      0.0,
      -2.2,
      0.0,
      2.0,
      0.7853981633974483
    ],
    [
      0.004050403828028346,
      -0.2810267266014614,
      0.003172307651805746,
      -2.1753309274243544,
      0.005064596882325875,
      1.9903042008228928,
      0.7868491058160682
    ],
    [
      -9.999999966669073,
      -0.007494071296229043,
      -9.997999966652007,
      -0.02434929220083126,
      20.007977212298353,
      0.014755220904600748,
      0.007977178977277838
    ]
  ]
})


def inverse_kinematics_output_matches(actual, expected, arguments, helpers, case_name):
    """Check solver outputs without choosing a unique converged posture."""
    theta, history = actual
    theta, history = np.asarray(theta), np.asarray(history)
    initial = arguments["theta"]
    limit = arguments.get("max_iterations", 200)
    if (theta.shape != initial.shape or history.ndim != 2
            or history.shape[1:] != initial.shape or not 1 <= len(history) <= limit + 1
            or not np.isrealobj(theta) or not np.isrealobj(history)
            or not np.isfinite(theta).all() or not np.isfinite(history).all()
            or not np.allclose(history[0], initial, atol=ATOL, rtol=RTOL)
            or not np.allclose(history[-1], theta, atol=ATOL, rtol=RTOL)):
        return False
    if case_name in IK_CONVERGENCE_CASES:
        pose = helpers.forward_kinematics(arguments["twists"], theta) @ arguments["g_st0"]
        error = helpers.spatial_configuration_error(pose, arguments["desired_pose"])
        vector = helpers.spatial_error_twist(error)
        return np.linalg.norm(vector) <= arguments.get("tolerance", 1e-6) * 1.00001
    return (output_matches(theta, np.asarray(expected[0]))
            and output_matches(history, np.asarray(expected[1])))


def output_matches(actual, expected):
    """Return whether one result has the expected shape and values."""
    actual = np.asarray(actual)
    return (
        actual.shape == expected.shape
        and np.all(np.isfinite(actual))
        and np.allclose(actual, expected, rtol=RTOL, atol=ATOL)
    )


def run_checks():
    """Run all public HW7 checks and return whether they passed."""
    try:
        import jacobian_func_skeleton as jacobian_student
    except BaseException as error:
        print(f"Jacobian import error: {type(error).__name__}: {error}")
        jacobian_student = None
    try:
        import euler as euler_student
    except BaseException as error:
        print(f"Euler import error: {type(error).__name__}: {error}")
        euler_student = None
    try:
        import overactuated_ik as ik_student
    except BaseException as error:
        print(f"IK import error: {type(error).__name__}: {error}")
        ik_student = None

    all_passed = True
    for function_name in FUNCTION_NAMES:
        student = (jacobian_student if function_name in JACOBIAN_FUNCTION_NAMES
                   else euler_student if function_name in EULER_FUNCTION_NAMES else ik_student)
        function = getattr(student, function_name, None)
        if student is None:
            all_passed = False
            for case_name, _ in CASE_FUNCTIONS[function_name]():
                print(f"[FAIL] {function_name}: {case_name}")
            continue
        for index, (case_name, arguments) in enumerate(CASE_FUNCTIONS[function_name]()):
            expected = EXPECTED_RESULTS[function_name][index]
            before = {
                name: value.copy() if hasattr(value, "copy") else value
                for name, value in arguments.items()
            }
            try:
                actual = function(**arguments)
                if function_name == "jacobian_inverse_kinematics":
                    passed = inverse_kinematics_output_matches(
                        actual, expected, before, student, case_name)
                else:
                    passed = output_matches(actual, np.array(expected))
            except BaseException:
                passed = False
            all_passed &= passed
            print(f"[{'PASS' if passed else 'FAIL'}] {function_name}: {case_name}")

    return all_passed


if __name__ == "__main__":
    raise SystemExit(0 if run_checks() else 1)
