"""P5: fixed-goal Panda inverse-kinematics experiments.

P1 uses run_ur_ik.py for the UR7e. Every Panda demo here defines starting
joint values and one desired tool pose, then repeatedly calls the IK step.
Use --demo regular, singularity, manipulability, joint-limits,
initial-conditions, or ikflow. IKFlow supplies initial joint proposals only.
Use --target my_pose.npy to specify a custom fixed (4, 4) goal pose.
"""

import argparse
import importlib.util
from pathlib import Path
import sys

import numpy as np


HW7_DIR = Path(__file__).resolve().parent
for parent in (HW7_DIR.parent, HW7_DIR.parent / "hw2" / "student"):
    if parent.is_dir() and str(parent) not in sys.path:
        sys.path.append(str(parent))

import panda_model
from panda_model import (
    ORIENTATION_TOLERANCE,
    PANDA_READY,
    PANDA_THETA_MAX,
    PANDA_THETA_MIN,
    POSITION_TOLERANCE,
    spatial_velocity_command,
    manipulability,
    manipulability_gradient,
    panda_fk,
    pose_error,
)


def load_implementation(reference=False):
    """Load student functions, or the private staff functions when requested.

    Args:
        reference: If True, load the references from the staff repository.

    Returns:
        The module containing the provided step and two student IK functions.
        In reference mode, the
        provided model helpers also use the staff Jacobian and HW2 functions.
    """
    if not reference:
        import overactuated_ik

        return overactuated_ik

    staff_root = HW7_DIR.parents[1]
    reference_dir = str(staff_root / "autograders/hw7/source")
    if reference_dir not in sys.path:
        sys.path.insert(0, reference_dir)
    modules = []
    for homework in ("hw2", "hw7"):
        path = staff_root / "autograders" / homework / "source" / "reference.py"
        if not path.is_file():
            raise ValueError("--reference is available only in the staff repository")
        spec = importlib.util.spec_from_file_location(f"{homework}_reference", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)

    kinematics, implementation = modules
    panda_model.forward_kinematics = kinematics.forward_kinematics
    panda_model.se3_to_twist = kinematics.se3_to_twist
    panda_model.R3_to_so3 = kinematics.R3_to_so3
    panda_model.twist_to_SE3 = kinematics.twist_to_SE3
    sys.modules["jacobian_func_skeleton"] = implementation
    return implementation


def initial_conditions():
    """Return a dictionary of four named Panda starting configurations.

    Each value is a NumPy array with shape (7,) of joint angles in radians.
    All four configurations are within the provided Panda joint limits.
    """
    return {
        "Ready": PANDA_READY.copy(),
        "Left elbow": np.array([0.5, -0.6, 0.7, -2.0, -0.4, 1.7, 0.3]),
        "Right elbow": np.array([-0.5, -0.6, -0.7, -2.0, 0.4, 1.7, 1.2]),
        "Near joint limits": np.array([1.6, -1.3, 1.0, -2.8, -1.2, 0.5, 1.8]),
    }


def run_demo(
    name, implementation, target=None, seeds_path=HW7_DIR / "ikflow_seeds.npz",
    target_index=0, seed_count=8, step_size=None, max_iterations=300,
):
    """Run a fixed-target Panda IK comparison.

    Args:
        name: "regular", "singularity", "initial-conditions",
            "manipulability", "joint-limits", or "ikflow".
        implementation: Module containing the three student IK functions.
        target: Optional (4, 4) tool-tip pose overriding the demo target.
            IKFlow uses its archive target.
        seeds_path: Path to a pickle-free IKFlow seed archive.
        target_index: Index of the target in the seed archive.
        seed_count: Maximum number of learned seeds to refine.
        step_size: Positive scalar multiplier for IK updates. If omitted,
            use 0.7 for shoulder comparisons and 0.2 for the other demos.
        max_iterations: Maximum updates for each fixed-target refinement.

    Returns:
        A dictionary of named ``run_ik`` results. Each includes a "demo"
        string identifying the comparison. Failed refinements are retained.
        Comparisons use the same start and targets as standard IK. The fixed
        joint-limit demo reserves up to 80 final updates for primary-task
        correction. Fixed-target standard IK stops at its first feasible pose;
        it is not a minimum-time trajectory optimizer.
        Every history solves for one fixed target. IKFlow proposes starting
        joints only; its seeds are refined toward the archive's fixed pose.
        Shoulder comparisons use uncapped updates and the same pose stopping
        condition. Other demos retain the provided loop's safety limiting.
    """
    if step_size is None:
        step_size = 0.7 if name in ("singularity", "manipulability") else 0.2
    if name in ("singularity", "manipulability"):
        start, default = singularity_demo_setup()
        target = default if target is None else target
        settings = {"step_size": step_size, "max_iterations": max_iterations, "uncapped": True}
        results = {"Standard IK": run_ik(
            start, target, implementation.jacobian_ik_step, **settings,
        )}
        if name == "manipulability":
            results["Manipulability"] = run_ik(
                start, target, implementation.null_space_ik_step,
                secondary_gradient=manipulability_gradient, secondary_gain=6.0, **settings,
            )
        for result in results.values():
            result["demo"] = name
        return results
    if name == "joint-limits":
        start, demo_target, gain = secondary_demo_setup(name)
        target = demo_target if target is None else target
    else:
        target = default_target() if target is None else target
    settings = {"step_size": step_size, "max_iterations": max_iterations}
    results = {}
    if name == "regular":
        results["Standard IK"] = run_ik(
            PANDA_READY, target, implementation.jacobian_ik_step, **settings,
        )
    elif name in ("initial-conditions", "ikflow"):
        if name == "initial-conditions":
            starts = initial_conditions()
        else:
            target, seeds = load_ikflow_seeds(seeds_path, target_index)
            starts = {
                f"IKFlow seed {index + 1}": theta
                for index, theta in enumerate(seeds[:seed_count])
            }
        for label, theta in starts.items():
            result = run_ik(theta, target, implementation.jacobian_ik_step, **settings)
            if name == "ikflow":
                result["refinement_end"] = len(result["theta"]) - 1
                result["refinement_converged"] = result["converged"]
            results[label] = result
    elif name in ("manipulability", "joint-limits"):
        results["Standard IK"] = run_ik(start, target, implementation.jacobian_ik_step, **settings)
        if name == "manipulability":
            label, gradient = "Manipulability", manipulability_gradient
        else:
            label, gradient = "Joint-limit avoidance", implementation.joint_limit_gradient
        results[label] = run_ik(
            start, target, implementation.null_space_ik_step,
            secondary_gradient=gradient, secondary_gain=gain,
            settle_iterations=min(80, max_iterations // 3), **settings,
        )
    else:
        raise ValueError("Unknown Panda demo")
    for result in results.values():
        result["demo"] = name
    return results


def singularity_demo_setup():
    """Return the fixed start (7,) and goal pose (4, 4) for both comparisons.

    The tool starts about 1 m from the goal in a well-conditioned posture.
    Standard IK passes near a rank-five shoulder configuration. The goal
    remains fixed for every update, and the null-space method uses the same
    start, goal, and primary step size. These are kinematic comparisons,
    not collision-checked robot commands.

    The nearby singular family has sin(theta[1]) = 0 and
    cos(theta[2]) = cos(theta[4]) = 0; see Eq. (36a):
    https://arxiv.org/pdf/2211.02516
    """
    start = np.array([-0.7897167591692444, 0.2754510370036218,
                      0.6202741611320922, -1.88540946394611,
                      2.3551787399613393, 2.4128617580538303,
                      1.0160258062636123])
    target = np.array([
        [0.67583492743073, 0.09025320732934045, -0.7315063290440258, -0.0006952074603470762],
        [0.6864740349206827, -0.4383950296574934, 0.5801406703130912, 0.5545022267000204],
        [-0.26832918261794997, -0.8942394290895322, -0.35823915645447013, 0.26032166756789676],
        [0.0, 0.0, 0.0, 1.0],
    ])
    return start, target


def secondary_demo_setup(name):
    """Return a reproducible start, target, and secondary gain for a demo.

    Args:
        name: "manipulability" or "joint-limits".

    Returns:
        ``(start, target, gain)`` with shapes (7,), (4, 4), and a scalar.
        Manipulability uses the start and final target of the shared path.
        Joint centering uses a roughly 5 cm reach from a near-limit posture.
        These are kinematic comparisons, not collision-checked trajectories.

    The projected-gradient controller follows the Robotics Toolbox examples:
    https://github.com/petercorke/robotics-toolbox-python/blob/main/examples/park.py
    https://github.com/petercorke/robotics-toolbox-python/blob/main/examples/baur.py
    """
    if name == "manipulability":
        start, target = singularity_demo_setup()
        return start, target, 6.0
    elif name == "joint-limits":
        start = np.array([2.857300, -1.138688, 0.454045, -1.465993, -1.753296, 0.510303, 1.204372])
        goal = np.array([2.724996, -1.123376, 0.502677, -1.409465, -1.731928, 0.505223, 1.298988])
        gain = 0.5
    else:
        raise ValueError("Unknown secondary-task demo")
    return start, panda_fk(goal), gain


def default_target():
    """Return a reachable target tool-tip pose with shape (4, 4).

    The NumPy array expresses the desired tool-tip frame in the robot base
    frame. It is computed from a fixed offset from the ready configuration.
    """
    theta = PANDA_READY + np.array([0.25, -0.15, 0.2, 0.15, -0.15, 0.1, -0.2])
    return panda_fk(theta)


def joint_limit_cost(theta):
    """Calculate the normalized distance from the joint-range midpoint.

    Args:
        theta: NumPy array with shape (7,) of joint angles in radians.

    Returns:
        A scalar equal to half the sum of squared midpoint offsets, each
        divided by that joint's squared half-range. Smaller values place
        joints closer to the midpoint of their limits.
    """
    midpoint = (PANDA_THETA_MIN + PANDA_THETA_MAX) / 2
    half_range = (PANDA_THETA_MAX - PANDA_THETA_MIN) / 2
    return 0.5 * np.sum(((theta - midpoint) / half_range) ** 2)


def run_ik(
    theta,
    target_pose,
    step_function,
    secondary_gradient=None,
    step_size=0.2,
    secondary_gain=0.05,
    max_iterations=300,
    settle_iterations=0,
    uncapped=False,
):
    """Run the provided IK loop and record every joint configuration.

    Args:
        theta: NumPy array with shape (7,) of initial joint angles in radians.
        target_pose: NumPy array with shape (4, 4) for the desired tool pose.
        step_function: ``jacobian_ik_step`` or ``null_space_ik_step`` from HW7.
        secondary_gradient: Optional function mapping shape (7,) joint values
            to a shape (7,) ascent direction for the secondary objective.
        step_size: Scalar multiplier passed to the student step function.
        secondary_gain: Scalar multiplier passed to ``null_space_ik_step``.
        max_iterations: Maximum number of joint updates.
        settle_iterations: For secondary tasks, reserve this many final
            updates for pose correction with a zero secondary gradient.
            Zero keeps the secondary objective active for the entire run.
        uncapped: If True, do not cap or clip updates. Stop at the pose
            tolerances for either method, or before an invalid update, a
            step over pi radians per joint, or a joint-limit crossing.

    The loop calls ``spatial_velocity_command`` for a signed six-vector toward
    the target and ``pose_error`` for the stopping condition. Each update is
    capped at 0.15 radians per joint and clipped to the joint limits unless
    uncapped is True. Primary
    IK stops at the pose tolerances or a numerically zero update; secondary
    tasks run for all iterations.

    Returns:
        A dictionary with "theta" of shape (N, 7), "target" of shape (4, 4),
        and shape (N,) arrays "position_error" (meters), "orientation_error"
        (radians), "manipulability", and "joint_limit_cost". N includes the
        initial configuration. "converged" is a Boolean for the final pose.
        Uncapped runs also include shape (N, 7) "joint_update", shape (N,)
        "smallest_singular_value", and a "stop_reason" string. A rejected
        proposal is recorded separately as shape (7,) "rejected_update".
    """
    theta = np.asarray(theta, dtype=float).copy()
    history = []
    position_errors = []
    orientation_errors = []
    manipulabilities = []
    limit_costs = []
    singular_values = []
    stop_reason = ""
    rejected_update = None

    for iteration in range(max_iterations + 1):
        pose = panda_fk(theta)
        position_error, orientation_error = pose_error(pose, target_pose)
        history.append(theta.copy())
        position_errors.append(position_error)
        orientation_errors.append(orientation_error)
        manipulabilities.append(manipulability(theta))
        limit_costs.append(joint_limit_cost(theta))
        if uncapped:
            from jacobian_func_skeleton import spatial_jacobian

            jacobian = spatial_jacobian(panda_model.PANDA_TWISTS, theta)
            singular_values.append(np.linalg.svd(jacobian, compute_uv=False)[-1])
        converged = (
            position_error < POSITION_TOLERANCE
            and orientation_error < ORIENTATION_TOLERANCE
        )
        secondary_active = secondary_gradient is not None and iteration < max_iterations - settle_iterations
        if iteration == max_iterations or (converged and (uncapped or not secondary_active)):
            break

        velocity = spatial_velocity_command(pose, target_pose)
        if secondary_gradient is None:
            next_theta = step_function(theta, velocity, step_size=step_size)
        else:
            next_theta = step_function(
                theta,
                velocity,
                secondary_gradient(theta) if secondary_active else np.zeros(7),
                step_size=step_size,
                secondary_gain=secondary_gain,
            )
        next_theta = np.asarray(next_theta, dtype=float)
        if next_theta.shape != (7,) or not np.all(np.isfinite(next_theta)):
            if uncapped:
                stop_reason = "invalid joint update"
                break
            raise ValueError("The IK step must return seven finite joint angles")
        delta = next_theta - theta
        if secondary_gradient is None and np.linalg.norm(delta) < 1e-12:
            if uncapped:
                stop_reason = "zero joint update"
            break
        if uncapped:
            if np.max(np.abs(delta)) > np.pi:
                stop_reason = "joint update exceeds pi radians"
            elif np.any(next_theta < PANDA_THETA_MIN) or np.any(next_theta > PANDA_THETA_MAX):
                stop_reason = "joint limit reached"
            if stop_reason:
                rejected_update = delta.copy()
                break
            theta = next_theta.copy()
        else:
            delta *= min(1.0, 0.15 / max(np.max(np.abs(delta)), 1e-12))
            theta = np.clip(theta + delta, PANDA_THETA_MIN, PANDA_THETA_MAX)

    result = {
        "theta": np.array(history),
        "target": np.asarray(target_pose).copy(),
        "position_error": np.array(position_errors),
        "orientation_error": np.array(orientation_errors),
        "manipulability": np.array(manipulabilities),
        "joint_limit_cost": np.array(limit_costs),
        "converged": bool(converged),
    }
    if uncapped:
        result["joint_update"] = np.vstack([np.zeros(7), np.diff(result["theta"], axis=0)])
        result["smallest_singular_value"] = np.array(singular_values)
        result["stop_reason"] = stop_reason or ("" if converged else "iteration budget reached")
        if rejected_update is not None:
            result["rejected_update"] = rejected_update
    return result


def load_ikflow_seeds(path, target_index=0):
    """Load one target and its learned joint seeds from a provided archive.

    Args:
        path: Path to the staff-generated ``ikflow_seeds.npz`` file.
        target_index: Integer index of the desired target.

    Returns:
        ``(target_pose, seeds)`` with shapes (4, 4) and (N, 7). Targets use
        the course tool-tip frame, and seeds contain joint angles in radians.
    """
    with np.load(path, allow_pickle=False) as data:
        targets = data["targets"]
        seeds = data["seeds"]
        if targets.ndim != 3 or targets.shape[1:] != (4, 4):
            raise ValueError("targets must have shape (T, 4, 4)")
        if seeds.ndim != 3 or seeds.shape[0] != len(targets) or seeds.shape[2] != 7:
            raise ValueError("seeds must have shape (T, N, 7)")
        if str(data["frame"]) != "panda_tool_tip":
            raise ValueError("Seed targets must use the course tool-tip frame")
        if not np.all(np.isfinite(targets)) or not np.all(np.isfinite(seeds)):
            raise ValueError("Seed data must be finite")
        return targets[target_index].copy(), seeds[target_index].copy()


def main():
    """Run a fixed-goal IK experiment using the student functions."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--demo", choices=("regular", "singularity", "initial-conditions",
                                           "manipulability", "joint-limits", "ikflow"))
    parser.add_argument("--experiment", choices=("ik", "secondary-tasks", "initial-conditions", "ikflow"),
                        default="ik")
    parser.add_argument("--target", type=Path, help="Custom fixed (4, 4) target .npy file")
    parser.add_argument("--seeds", type=Path, default=HW7_DIR / "ikflow_seeds.npz")
    parser.add_argument("--target-index", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=8)
    parser.add_argument("--max-iterations", type=int, default=300)
    parser.add_argument("--step-size", type=float,
                        help="Primary step size: default 1 for shoulder demos, 0.2 otherwise")
    parser.add_argument("--playback-dt", type=float, default=0.03)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--no-visualize", action="store_true")
    parser.add_argument("--browser")
    parser.add_argument("--save", type=Path)
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    if (args.max_iterations < 1 or args.seed_count < 1
            or (args.step_size is not None and (not np.isfinite(args.step_size) or args.step_size <= 0))
            or not np.isfinite(args.playback_dt) or args.playback_dt <= 0):
        parser.error("Use positive iterations, seed count, step size, and playback interval")
    target = np.load(args.target, allow_pickle=False) if args.target else None
    if target is not None and (target.shape != (4, 4) or not np.isfinite(target).all()):
        parser.error("target must be a finite (4, 4) pose")
    implementation = load_implementation(args.reference)
    if not args.demo and args.experiment == "secondary-tasks":
        target = default_target() if target is None else target
        settings = {"step_size": 0.2 if args.step_size is None else args.step_size,
                    "max_iterations": args.max_iterations}
        results = {"Standard IK": run_ik(PANDA_READY, target, implementation.jacobian_ik_step, **settings)}
        for name, gradient, gain in (
            ("Manipulability", manipulability_gradient, 0.5),
            ("Joint-limit avoidance", implementation.joint_limit_gradient, 0.05),
        ):
            results[name] = run_ik(PANDA_READY, target, implementation.null_space_ik_step,
                                   secondary_gradient=gradient, secondary_gain=gain, **settings)
    else:
        name = args.demo or ("regular" if args.experiment == "ik" else args.experiment)
        results = run_demo(name, implementation, target, args.seeds, args.target_index,
                           args.seed_count, args.step_size, args.max_iterations)
    for name, result in results.items():
        message = (f"{name}: {len(result['theta']) - 1} updates; "
                   f"final position error {result['position_error'][-1]:.2e} m; "
                   f"converged={result['converged']}")
        if "joint_update" in result:
            message += f"; largest step {np.degrees(np.max(np.abs(result['joint_update']))):.1f} deg"
        if result.get("stop_reason"):
            message += f"; {result['stop_reason']}"
        print(message)
    if args.save:
        save_results(args.save, results)
    if (args.demo or args.visualize) and not args.no_visualize:
        from visualize_panda import visualize_results

        visualize_results(results, dt=args.playback_dt, browser=args.browser)


def save_results(path, results):
    """Save named ``run_ik`` histories to a pickle-free NumPy archive.

    Args:
        path: Output path ending in .npz.
        results: Dictionary mapping names to ``run_ik`` result dictionaries,
            including (N, 7) joint histories and (N,) error/objective arrays.

    Returns:
        None. The archive can be opened by ``visualize_panda.load_results``.
    """
    arrays = {"names": np.array(list(results))}
    for index, result in enumerate(results.values()):
        for key, value in result.items():
            arrays[f"{index}_{key}"] = value
    np.savez_compressed(path, **arrays)


if __name__ == "__main__":
    main()
