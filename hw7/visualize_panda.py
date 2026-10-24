"""Play Panda IK histories in Swift.

Run ``python hw7/visualize_panda.py histories.npz`` after saving an experiment
with ``run_panda_ik.py --save histories.npz``. The same visualizer accepts
custom joint histories through ``visualize_results``.
"""

import argparse
from time import monotonic
from html import escape
from pathlib import Path
import sys

import numpy as np


HW7_DIR = Path(__file__).resolve().parent
for parent in (HW7_DIR.parent, HW7_DIR.parent / "hw2" / "student"):
    if parent.is_dir() and str(parent) not in sys.path:
        sys.path.append(str(parent))

from panda_model import load_panda
from viewer_style import VIEWER_STYLE

DEMO_TITLES = {
    "regular": "Panda inverse kinematics",
    "singularity": "Panda singularity comparison",
    "initial-conditions": "Different starting configurations",
    "manipulability": "Manipulability comparison",
    "joint-limits": "Joint-limit comparison",
    "ikflow": "IKFlow: different starts, same goal",
}


def load_results(path):
    """Load a pickle-free playback archive created by ``save_results``.

    Args:
        path: Path to an experiment .npz archive.

    Returns:
        A dictionary mapping display names to ``run_ik`` results, with (N, 7)
        joint histories, (4, 4) targets, and (N,) error/objective arrays.
    """
    results = {}
    fields = (
        "theta", "target", "position_error", "orientation_error",
        "manipulability", "joint_limit_cost", "converged",
    )
    with np.load(path, allow_pickle=False) as data:
        for index, name in enumerate(data["names"]):
            results[str(name)] = {
                field: data[f"{index}_{field}"].copy() for field in fields
            }
            for field in ("targets", "waypoint_iterations", "demo",
                          "refinement_end", "refinement_converged", "tracking_converged",
                          "smallest_singular_value", "joint_update", "stop_reason", "rejected_update",
                          "motion_start", "challenge_start"):
                if f"{index}_{field}" in data:
                    results[str(name)][field] = data[f"{index}_{field}"].copy()
    return results


def visualize_results(results, dt=0.03, test=False, browser=None):
    """Animate named Panda joint histories in the browser.

    Args:
        results: Dictionary mapping names to ``run_ik`` result dictionaries.
            Each result contains "theta" of shape (N, 7), "target" of shape
            (4, 4), and shape (N,) error and objective arrays. Moving paths also
            use one fixed "target" pose throughout the history.
        dt: Rendering interval in seconds. Each fixed-goal IK update takes
            0.5 seconds, with linear joint interpolation within each update.
            Paired methods share an iteration index and hold their final
            pose after stopping; playback is not physical time.
        test: If True, display one headless frame and return for local tests.
        browser: Optional browser name accepted by Swift.

    Select a method or starting configuration, then use Play/Pause.
    Play restarts a completed history. Large axes mark the target tool pose; small
    axes mark the achieved tool pose. The curve shows the selected tool path:
    blue for standard IK, green for manipulability, orange for joint limits.
    Target axes remain at the fixed goal. All histories use the same
    robot base. Secondary-task and IKFlow demos show two translated copies
    of that workspace side-by-side. Comparisons start together and play in
    sync. IKFlow compares a raw seed or a different seed's refinement toward
    the same fixed goal. "Refined pose" pauses at the final refined pose.

    Returns:
        In test mode, a dictionary containing the environment, robot handle,
        controls, and playback state. Otherwise runs until interrupted.
    """
    if not results or dt <= 0:
        raise ValueError("Provide at least one history and a positive dt")
    for result in results.values():
        theta = np.asarray(result["theta"])
        if theta.ndim != 2 or theta.shape[1] != 7 or len(theta) == 0:
            raise ValueError("Each joint history must have shape (N, 7)")
        if not np.all(np.isfinite(theta)):
            raise ValueError("Joint histories must be finite")
        if "targets" in result and np.asarray(result["targets"]).shape != (len(theta), 4, 4):
            raise ValueError("Moving targets must have shape (N, 4, 4)")

    import spatialgeometry as sg
    import swift

    panda = load_panda()
    names = list(results)
    demo = str(results[names[0]].get("demo", ""))
    fixed_goal = all("targets" not in result for result in results.values())
    peak = 0
    if (demo in ("singularity", "manipulability")
            and "joint_update" in results.get("Standard IK", {})
            and max(len(result["theta"]) for result in results.values()) > 1):
        peak = int(np.argmax(np.max(np.abs(results["Standard IK"]["joint_update"]), axis=1)))
    paired = demo in ("manipulability", "joint-limits", "ikflow") and len(names) > 1
    state = {
        "selection": min(1, len(names) - 1) if demo in ("manipulability", "joint-limits") else 0,
        "iteration": 0, "playing": not test, "last_time": 0.0,
        "comparison_mode": "different", "playback_started": 0.0,
    }
    if demo == "ikflow":
        state["selection"] = next(
            (index for index, name in enumerate(names) if results[name]["converged"]), 0
        )
        if not fixed_goal:
            state["iteration"] = int(results[names[state["selection"]]].get("refinement_end", 0))

    def playback_last():
        if demo in ("manipulability", "joint-limits"):
            return max(len(result["theta"]) for result in results.values()) - 1
        if demo == "ikflow" and fixed_goal and paired:
            other = comparison_configuration()[2]
            return max(len(results[names[state["selection"]]]["theta"]), len(other["theta"])) - 1
        return len(results[names[state["selection"]]]["theta"]) - 1

    def joint_configuration(result, progress):
        # Interpolate playback only; the computed IK history is unchanged.
        progress = min(max(float(progress), 0.0), len(result["theta"]) - 1)
        index = int(progress)
        fraction = progress - index
        following = min(index + 1, len(result["theta"]) - 1)
        return result["theta"][index] + fraction * (result["theta"][following] - result["theta"][index])

    def comparison_configuration():
        selected = results[names[state["selection"]]]
        if demo in ("manipulability", "joint-limits"):
            other = results["Standard IK"]
            index = int(min(state["iteration"], len(other["theta"]) - 1))
            label = "Standard IK"
            return joint_configuration(other, state["iteration"]), label, other, index
        if state["comparison_mode"] == "raw":
            return selected["theta"][0], "Raw " + names[state["selection"]], selected, 0
        candidates = [
            name for name in names
            if name != names[state["selection"]] and results[name]["converged"]
        ]
        if not candidates:
            candidates = [name for name in names if name != names[state["selection"]]]
        selected_end = int(selected.get("refinement_end", len(selected["theta"]) - 1))
        name = max(candidates, key=lambda name: np.linalg.norm(
            results[name]["theta"][int(results[name].get("refinement_end", len(results[name]["theta"]) - 1))]
            - selected["theta"][selected_end]
        ))
        other = results[name]
        other_end = int(other.get("refinement_end", len(other["theta"]) - 1))
        progress = max(0, state["iteration"] - selected_end)
        position = min(state["iteration"] if fixed_goal else other_end + progress, len(other["theta"]) - 1)
        index = int(position)
        label = "refined" if index == other_end else "refining"
        return joint_configuration(other, position), name + f" ({label})", other, index
    environment = swift.Swift()
    environment.launch(
        headless=test, browser=browser, axes=False, ground_pattern="@grid"
    )
    status = swift.Label(label="Loading Panda meshes...", compact=True)
    environment.add_ui(swift.Label(label=VIEWER_STYLE, compact=True))
    environment.add_ui(status)
    if demo in ("singularity", "manipulability"):
        environment.set_camera_pose(
            [1.8, 2.0, 1.25] if paired else [1.35, 1.15, 1.05],
            [0.0, 0.75 if paired else 0.25, 0.35],
        )
    elif paired:
        environment.set_camera_pose([1.65, 1.7, 1.1], [0.3, 0.5, 0.4])
    else:
        environment.set_camera_pose([1.4, 1.1, 1.05], [0.35, 0.0, 0.4])
    panda.q = results[names[state["selection"]]]["theta"][state["iteration"]].copy()
    target_axes = sg.Axes(length=0.16, arrows=True, pose=results[names[0]]["target"])
    achieved_axes = sg.Axes(length=0.08, arrows=True, pose=panda.fkine(panda.q))
    environment.add_shape(target_axes)
    environment.add_shape(achieved_axes)
    paths = {}
    for name, result in results.items():
        positions = np.array([
            panda.fkine(theta).t for theta in result["theta"]
        ]).T
        if positions.shape[1] == 1:
            positions = np.repeat(positions, 2, axis=1)
        paths[name] = positions
    colors = {
        "Standard IK": [0.12, 0.47, 0.71],
        "Manipulability": [0.17, 0.63, 0.17],
        "Joint-limit avoidance": [1.0, 0.5, 0.05],
    }
    path = sg.Polyline(
        paths[names[state["selection"]]], linewidth=3,
        color=colors.get(names[state["selection"]], [0.12, 0.47, 0.71]),
    )
    environment.add_shape(path)
    desired_paths = {}
    for name, result in results.items():
        if "targets" in result:
            desired_paths[name] = result["targets"][:, :3, 3].T
        else:
            desired_paths[name] = np.repeat(result["target"][:3, 3, None], 2, axis=1)
    desired_path = sg.Polyline(desired_paths[names[0]], linewidth=2, color=[0.6, 0.6, 0.6])
    environment.add_shape(desired_path)
    comparison_handle = None
    comparison_target_axes = None
    if paired:
        from spatialmath import SE3

        comparison_robot = load_panda()
        offset = SE3(0.0, 1.0, 0.0).A
        comparison_robot.base = SE3(offset)
        comparison_robot.q = comparison_configuration()[0].copy()
        comparison_target_axes = sg.Axes(length=0.16, arrows=True, pose=offset @ results[names[0]]["target"])
        comparison_achieved_axes = sg.Axes(length=0.08, arrows=True)
        environment.add_shape(comparison_target_axes)
        environment.add_shape(comparison_achieved_axes)
        comparison_path = sg.Polyline(
            paths["Standard IK"] + offset[:3, 3, None] if demo in ("manipulability", "joint-limits")
            else paths[names[0]] + offset[:3, 3, None],
            linewidth=3, color=[0.12, 0.47, 0.71],
        )
        environment.add_shape(comparison_path)
        comparison_desired_path = sg.Polyline(
            desired_paths[names[0]] + offset[:3, 3, None],
            linewidth=2, color=[0.6, 0.6, 0.6],
        )
        environment.add_shape(comparison_desired_path)

    def display(index):
        result = results[names[state["selection"]]]
        state["iteration"] = min(max(float(index), 0), playback_last())
        index = int(min(state["iteration"], len(result["theta"]) - 1))
        theta = joint_configuration(result, state["iteration"])
        achieved_axes.T = panda.fkine(theta).A
        achieved_axes.update()
        target_axes.T = result["targets"][index] if "targets" in result else result["target"]
        target_axes.update()
        demo = str(result.get("demo", ""))
        title = DEMO_TITLES.get(demo, "Panda inverse kinematics")
        left_name = escape(names[state["selection"]])
        left_name += f" — iteration {index}/{len(result['theta']) - 1}"
        if names[state["selection"]] == "Standard IK" and peak and peak - 1 <= state["iteration"] <= peak:
            left_name += " (near singularity / large IK update)"
        if "motion_start" in result:
            left_name += " (initial hold)" if index < int(result["motion_start"]) else " (tracking)"
        if "refinement_end" in result:
            stage = "tracking" if index > int(result["refinement_end"]) else "refining"
            if index == int(result["refinement_end"]) and result["refinement_converged"]:
                stage = "refined pose"
            left_name += f" ({stage})"
        if "stop_reason" in result:
            if result["stop_reason"] and index == len(result["theta"]) - 1:
                left_name += f" ({escape(str(result['stop_reason']))})"
            elif not fixed_goal and (result["position_error"][index] >= 0.005 or result["orientation_error"][index] >= 0.005):
                left_name += " (tracking error)"
        elif not result["converged"]:
            left_name += " (path incomplete)" if result.get("refinement_converged", False) else " (did not converge)"
        labels = left_name
        if paired:
            other_theta, other_name, other, other_index = comparison_configuration()
            comparison_achieved_axes.T = comparison_robot.fkine(other_theta).A
            comparison_achieved_axes.update()
            other_target = other["targets"][other_index] if "targets" in other else other["target"]
            comparison_target_axes.T = offset @ other_target
            comparison_target_axes.update()
            if comparison_handle is not None:
                comparison_handle.q = other_theta
            other_key = next(name for name in names if results[name] is other)
            raw_seed = demo == "ikflow" and state["comparison_mode"] == "raw"
            comparison_points = (
                np.repeat(paths[other_key][:, :1], 2, axis=1) if raw_seed else paths[other_key]
            ) + offset[:3, 3, None]
            if not np.array_equal(comparison_path.points, comparison_points):
                comparison_path.points = comparison_points
                comparison_path.update()
            if raw_seed:
                comparison_targets = np.repeat(other_target[:3, 3, None] + offset[:3, 3, None], 2, axis=1)
            else:
                comparison_targets = desired_paths[other_key] + offset[:3, 3, None]
            if not np.array_equal(comparison_desired_path.points, comparison_targets):
                comparison_desired_path.points = comparison_targets
                comparison_desired_path.update()
            right_name = escape(other_name)
            right_name += f" — iteration {other_index}/{len(other['theta']) - 1}"
            if other_name == "Standard IK" and peak and peak - 1 <= state["iteration"] <= peak:
                right_name += " (near singularity / large IK update)"
            if "motion_start" in other:
                right_name += " (initial hold)" if other_index < int(other["motion_start"]) else " (tracking)"
            if "stop_reason" in other:
                if other["stop_reason"] and other_index == len(other["theta"]) - 1:
                    right_name += f" ({escape(str(other['stop_reason']))})"
                elif not fixed_goal and (other["position_error"][other_index] >= 0.005 or other["orientation_error"][other_index] >= 0.005):
                    right_name += " (tracking error)"
            elif not other["converged"] and state["comparison_mode"] != "raw":
                right_name += " (did not converge)"
            labels = f"Left: {left_name}<br>Right: {right_name}"
        status.label = f"<b>{title}</b><br>{labels}"
        return theta

    def animate(time, _values):
        if state["playing"] and fixed_goal:
            elapsed = max(0.0, monotonic() - state["playback_started"])
            # Equal time per IK update; larger joint changes move faster.
            index = min(playback_last(), elapsed / 0.5)
            if index != state["iteration"]:
                display(index)
            if index == playback_last():
                display(playback_last())
                state["playing"] = False
                play_button.label = "Play"
        elif state["playing"] and time - state["last_time"] + 1e-12 >= dt:
            index = min(state["iteration"] + 1, playback_last())
            state["last_time"] = time
            display(index)
            if index == playback_last():
                state["playing"] = False
                play_button.label = "Play"
        result = results[names[state["selection"]]]
        return joint_configuration(result, state["iteration"])

    handle = environment.add_robot(panda, readonly=True, callback=animate, name="Panda")
    if paired:
        comparison_handle = environment.add_robot(
            comparison_robot, readonly=True,
            callback=lambda time, values: comparison_configuration()[0], name="Comparison Panda",
        )

    def select_iteration(value):
        if state["playing"] and int(value) == state["iteration"]:
            return
        state["playing"] = False
        play_button.label = "Play"
        handle.q = display(value)

    def select_history(value):
        state["selection"] = int(value)
        path.points = paths[names[state["selection"]]]
        path.color = colors.get(names[state["selection"]], [0.12, 0.47, 0.71])
        path.update()
        desired_path.points = desired_paths[names[state["selection"]]]
        desired_path.update()
        result = results[names[state["selection"]]]
        select_iteration(0)

    def toggle(_value):
        if not state["playing"] and state["iteration"] == playback_last():
            select_iteration(0)
        state["playing"] = not state["playing"]
        state["last_time"] = environment.sim_time
        elapsed = 0.5 * state["iteration"]
        state["playback_started"] = monotonic() - elapsed
        play_button.label = "Pause" if state["playing"] else "Play"

    def restart(_value):
        select_iteration(0)
        state["playing"] = True
        state["last_time"] = environment.sim_time
        state["playback_started"] = monotonic()
        play_button.label = "Pause"

    def show_initial(_value):
        select_iteration(0)

    def show_final(_value):
        last = playback_last()
        select_iteration(last)

    def show_refined(_value):
        result = results[names[state["selection"]]]
        select_iteration(playback_last() if fixed_goal
                         else int(result.get("refinement_end", len(result["theta"]) - 1)))

    def select_comparison(value):
        state["comparison_mode"] = "different" if int(value) == 0 else "raw"
        handle.q = display(state["iteration"])

    selector = swift.Select(
        select_history, label="Left arm / initial condition", options=names,
        value=state["selection"],
    )
    play_button = swift.Button(toggle, label="Pause" if state["playing"] else "Play")
    for control in (
        selector, play_button, swift.Button(restart, label="Restart"),
    ):
        environment.add_ui(control)
    if demo == "ikflow" and paired:
        environment.add_ui(swift.Select(
            select_comparison, label="Right arm",
            options=["Different refined solution", "Raw seed"],
        ))
    if demo == "ikflow":
        environment.add_ui(swift.Button(show_refined, label="Refined pose"))
    handle.q = display(state["iteration"])

    if test:
        return {
            "environment": environment, "handle": handle, "state": state,
            "select_history": select_history, "select_iteration": select_iteration,
            "restart": restart, "show_final": show_final, "show_initial": show_initial,
            "animate": animate,
            "target_axes": target_axes,
            "status": status,
            "comparison_handle": comparison_handle,
            "comparison_target_axes": comparison_target_axes,
            "select_comparison": select_comparison,
            "toggle": toggle,
            "show_refined": show_refined,
        }
    state["playback_started"] = monotonic()
    environment.step(0.0)
    environment.run(dt=min(dt, 0.05))


def main():
    """Display a saved experiment using the provided playback controls."""
    parser = argparse.ArgumentParser()
    parser.add_argument("history", type=Path)
    parser.add_argument("--dt", type=float, default=0.03)
    parser.add_argument("--browser")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()
    view = visualize_results(
        load_results(args.history), dt=args.dt, browser=args.browser, test=args.test
    )
    if args.test:
        view["environment"].close()


if __name__ == "__main__":
    main()
