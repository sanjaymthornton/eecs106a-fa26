"""P1: fixed-goal spatial Jacobian IK on the HW4 UR7e.

Run python hw7/run_ur_ik.py to replay your full iterative IK solver.
Use --reference only in the staff repository. --target accepts one fixed
base-frame tool0 pose in a NumPy file with shape (4, 4).
"""

import argparse
from pathlib import Path
import sys

import numpy as np

HW7_DIR = Path(__file__).resolve().parent
for parent in (HW7_DIR.parent, HW7_DIR.parent / "hw2" / "student"):
    if parent.is_dir() and str(parent) not in sys.path:
        sys.path.append(str(parent))

import ur7e_model
from ur7e_model import UR7E_EXTENDED, UR7E_READY, UR7E_TWISTS, UR7E_ZERO_POSE, pose_error


def load_implementation(reference=False):
    """Load the student's P1 functions, or the private staff reference."""
    if reference:
        if not (HW7_DIR.parents[1] / "autograders/hw7/source/reference.py").is_file():
            raise ValueError("--reference is available only in the staff repository")
        from run_panda_ik import load_implementation as load_reference

        return load_reference(True)
    import jacobian_func_skeleton

    return jacobian_func_skeleton


def solve_target(theta, target, solver, step_size=0.2, iterations=200):
    """Call the student's P1 solver once for a fixed goal pose.

    Args:
        theta: Initial joint angles with shape (6,).
        target: Fixed base-frame tool0 pose with shape (4, 4).
        solver: Student jacobian_inverse_kinematics function.
        step_size: Positive multiplier for each joint update.
        iterations: Maximum number of updates.

    Returns:
        Playback fields with N samples: theta (N, 6), achieved and targets
        (N, 4, 4), and shape (N,) pose errors, smallest singular values,
        largest joint updates, and limited flags. targets repeats the same
        goal for rendering, not a sequence of waypoints. No limiting or
        extra correction is added to the solver. These are kinematic
        histories, not collision-checked robot commands.
    """
    from jacobian_func_skeleton import spatial_jacobian

    theta = np.asarray(theta, dtype=float).copy()
    target = np.asarray(target, dtype=float)
    if theta.shape != (6,) or target.shape != (4, 4) or not np.isfinite(target).all():
        raise ValueError("Use theta (6,) and a finite target (4, 4)")
    final, history = solver(
        UR7E_TWISTS, theta.copy(), UR7E_ZERO_POSE, target.copy(),
        step_size=step_size, max_iterations=iterations,
    )
    history = np.asarray(history, dtype=float)
    if (history.ndim != 2 or history.shape[1:] != (6,)
            or not 1 <= len(history) <= iterations + 1
            or not np.isfinite(history).all()
            or not np.allclose(history[0], theta)
            or not np.allclose(history[-1], final)):
        raise ValueError("Return final joints (6,) and history (N, 6), including the start")
    achieved = np.array([ur7e_model.ur7e_fk(values) for values in history])
    errors = np.array([pose_error(pose, target) for pose in achieved])
    return {
        "theta": history.copy(),
        "targets": np.repeat(target[None], len(history), axis=0),
        "achieved": achieved,
        "position_error": errors[:, 0],
        "orientation_error": errors[:, 1],
        "smallest_singular_value": np.array([
            np.linalg.svd(spatial_jacobian(UR7E_TWISTS, values), compute_uv=False)[-1]
            for values in history
        ]),
        "raw_update": np.r_[0.0, np.max(np.abs(np.diff(history, axis=0)), axis=1)],
        "limited": np.zeros(len(history), dtype=bool),
    }


def run_demo(name, implementation, target=None, iterations=200, step_size=0.2):
    """Return fixed-goal UR7e histories for regular or singular starts.

    target is an optional (4, 4) goal. The student's solver is called once
    per initial posture; every iteration in that history uses the same goal.
    """
    if name == "regular":
        starts = {"Spatial Jacobian IK": UR7E_READY.copy()}
        goal = UR7E_READY + np.array([0.5, -0.3, 0.4, -0.3, 0.5, -0.4])
    elif name == "singularity":
        bent = UR7E_EXTENDED.copy()
        bent[1:4] = [-0.1, 0.2, -0.1]
        starts = {"Exactly extended": UR7E_EXTENDED.copy(), "Slightly bent": bent}
        goal = UR7E_EXTENDED.copy()
        goal[1:4] = [-0.3, 0.6, -0.3]
    else:
        raise ValueError("Unknown UR demo")
    target = ur7e_model.ur7e_fk(goal) if target is None else target
    return {
        label: solve_target(theta, target, implementation.jacobian_inverse_kinematics,
                            step_size=step_size, iterations=iterations)
        for label, theta in starts.items()
    }


def main():
    """Run P1 IK toward a fixed goal and open browser playback."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--demo", choices=("regular", "singularity"), default="regular")
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--target", type=Path)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--playback-dt", type=float, default=0.02)
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--browser")
    parser.add_argument("--save", type=Path)
    args = parser.parse_args()
    if (args.iterations < 1 or not np.isfinite(args.step_size) or args.step_size <= 0
            or not np.isfinite(args.playback_dt) or args.playback_dt <= 0):
        parser.error("Use positive iterations, step size, and playback interval")
    implementation = load_implementation(args.reference)
    if args.reference:
        ur7e_model.forward_kinematics = implementation.forward_kinematics
        ur7e_model.twist_to_SE3 = implementation.twist_to_SE3
    target = np.load(args.target, allow_pickle=False) if args.target else None
    results = run_demo(args.demo, implementation, target, args.iterations, args.step_size)
    for name, result in results.items():
        print(f"{name}: {len(result['theta']) - 1} updates; "
              f"final position error {result['position_error'][-1]:.2e} m")
    if args.save:
        data = {"names": np.array(list(results))}
        for index, result in enumerate(results.values()):
            data.update({f"{index}_{key}": value for key, value in result.items()})
        np.savez(args.save, **data)
    if not args.no_visualize:
        from visualize_ur import visualize_results

        visualize_results(results, dt=args.playback_dt, browser=args.browser)


if __name__ == "__main__":
    main()
