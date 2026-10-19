"""Run HW6 trajectory integration and compare timesteps.

python hw6/run_trajectory.py --sweep --visualize
The default sweep compares dt = 0.5, 0.3, 0.2, 0.1, and 0.05 seconds
over 500 seconds, with 4x playback speed.
python hw6/run_trajectory.py --desired-only --visualize
Use --reference only in the staff repository.
"""

import argparse
import importlib.util
from pathlib import Path
import sys
import types

import numpy as np


HW6_DIR = Path(__file__).resolve().parent
for parent in (HW6_DIR.parent, HW6_DIR.parent / "hw2" / "student"):
    if parent.is_dir() and str(parent) not in sys.path:
        sys.path.append(str(parent))


def load_implementation(reference=False):
    """Return student functions, or staff functions with correct HW2 helpers.

    reference is a Boolean. Staff mode requires the staff repository's
    private HW2 reference; it is not available in the distributed package.
    """
    if not reference:
        import hw6

        return hw6
    staff_root = HW6_DIR.parents[1]
    kin_path = staff_root / "autograders/hw2/source/reference.py"
    if not kin_path.is_file():
        raise ValueError("--reference is available only in the staff repository")
    package = types.ModuleType("hw2")
    package.__path__ = [str(kin_path.parent)]
    sys.modules["hw2"] = package
    spec = importlib.util.spec_from_file_location("hw2.kin_func_skeleton", kin_path)
    kin = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(kin)
    kin.inverse_SE3 = np.linalg.inv
    provided_path = staff_root / "sources/hw2/student/hw2/kin_func_skeleton.py"
    spec = importlib.util.spec_from_file_location("hw2_provided", provided_path)
    provided = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(provided)
    kin.renormalize_SE3 = provided.renormalize_SE3
    sys.modules["hw2.kin_func_skeleton"] = kin
    reference_path = staff_root / "autograders/hw6/source/reference.py"
    spec = importlib.util.spec_from_file_location("hw6_reference", reference_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def simulate(trajectory, velocity_function, step_function, dt=0.1, duration=500.0,
             desired_only=False):
    """Integrate a supplied trajectory using derivative-only velocity commands.

    Args:
        trajectory: Function t -> desired pose (4, 4).
        velocity_function: Student body_velocity(t) -> body velocity (4, 4).
        step_function: Student euler_step(pose, velocity_hat, dt).
        dt: Positive timestep in seconds.
        duration: Positive simulation duration in seconds.
        desired_only: If True, display the desired poses without calling
            the velocity or integration functions.

    Returns:
        Dictionary with times (N,), desired and integrated poses (N, 4, 4),
        position_error (N,) in meters, and orientation_error (N,) in radians.
        desired_path (1001, 3) densely samples the first revolution for
        drawing the repeating path independently of the integration timestep.
        The initial integrated pose equals trajectory(0). Each velocity
        command uses the desired trajectory at the step's start,
        not the achieved pose. There is no feedback. The last step is shortened
        when needed so every timestep comparison ends at the same time.
    """
    if not np.isfinite(dt) or not np.isfinite(duration) or dt <= 0 or duration <= 0:
        raise ValueError("Use positive finite dt and duration")
    times = np.append(np.arange(0.0, duration, dt), duration)
    desired = np.array([trajectory(t) for t in times], dtype=float)
    if desired.shape != (len(times), 4, 4) or not np.isfinite(desired).all():
        raise ValueError("The trajectory must return finite (4, 4) poses")
    poses = [desired[0].copy()]
    for index in range(len(times) - 1):
        if desired_only:
            next_pose = desired[index + 1].copy()
        else:
            velocity = velocity_function(times[index])
            next_pose = np.asarray(step_function(
                poses[-1].copy(), velocity, times[index + 1] - times[index]
            ), dtype=float)
        if next_pose.shape != (4, 4) or not np.isfinite(next_pose).all():
            raise ValueError("euler_step must return a finite (4, 4) pose")
        poses.append(next_pose)
    poses = np.array(poses)
    errors = []
    for actual, target in zip(poses, desired):
        position_error = np.linalg.norm(actual[:3, 3] - target[:3, 3])
        relative_rotation = actual[:3, :3].T @ target[:3, :3]
        cosine = (np.trace(relative_rotation) - 1.0) / 2.0
        orientation_error = np.arccos(np.clip(cosine, -1.0, 1.0))
        errors.append([position_error, orientation_error])
    errors = np.array(errors)
    desired_path = np.array([trajectory(t)[:3, 3]
                             for t in np.linspace(0, min(duration, 2 * np.pi), 1001)])
    return {"times": times, "desired": desired, "poses": poses,
            "position_error": errors[:, 0], "orientation_error": errors[:, 1],
            "desired_path": desired_path}


def plot_results(results):
    """Show the provided position and orientation error plots.

    results maps display names to simulate dictionaries. Plot against physical
    time, not sample index, so timestep choices can be compared fairly.
    """
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, sharex=True)
    for name, result in results.items():
        axes[0].plot(result["times"], 1000 * result["position_error"], label=name)
        axes[1].plot(result["times"], np.degrees(result["orientation_error"]), label=name)
    axes[0].set_ylabel("Position error (mm)")
    axes[1].set_ylabel("Orientation error (degrees)")
    axes[1].set_xlabel("Time (seconds)")
    axes[0].set_title("Derivative-only SE(3) integration")
    for axis in axes:
        axis.legend()
        axis.grid(True, alpha=0.3)
    figure.tight_layout()
    plt.show()


def main(visualize=False):
    """Run the chair swing and compare integration timesteps."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", action="store_true")
    parser.add_argument("--desired-only", action="store_true")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--duration", type=float, default=500.0)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--timesteps", type=float, nargs="+", default=[0.5, 0.3, 0.2, 0.1, 0.05])
    parser.add_argument("--visualize", action="store_true", default=visualize)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--browser")
    parser.add_argument("--speed", type=float, default=4.0,
                        help="Playback speed multiplier; does not change integration timesteps.")
    parser.add_argument("--start-time", type=float, default=0.0,
                        help="Begin playback at this time; integration still starts at zero.")
    args = parser.parse_args()
    implementation = load_implementation(args.reference)
    results = {}
    try:
        for dt in args.timesteps if args.sweep else [args.dt]:
            name = f"dt = {dt:g} s"
            results[name] = simulate(implementation.trajectory, implementation.body_velocity,
                                     implementation.euler_step, dt, args.duration, args.desired_only)
            print(f"{name}: mean position error {1000 * results[name]['position_error'].mean():.2f} mm; "
                  f"max {1000 * results[name]['position_error'].max():.2f} mm; "
                  f"max orientation error {np.degrees(results[name]['orientation_error'].max()):.2f} degrees")
    except (ValueError, NotImplementedError) as error:
        parser.error(str(error))
    if args.visualize:
        from visualize import visualize_results

        visualize_results(results, speed=args.speed, browser=args.browser, start_time=args.start_time)
    if not args.no_plots:
        plot_results(results)


if __name__ == "__main__":
    main()
