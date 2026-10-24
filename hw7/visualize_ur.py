"""Browser playback for the UR7e Jacobian IK question."""

import numpy as np
from time import monotonic

from ur7e_model import load_ur7e


def visualize_results(results, dt=0.02, test=False, browser=None):
    """Display named UR7e paths with Play, Restart, and selection controls.

    Args:
        results: Dictionary of histories returned by run_ur_ik.solve_target.
            Each contains theta (N, 6), targets and achieved (N, 4, 4), and
            error, singular-value, raw-update, and limited arrays (N,).
        dt: Positive rendering interval in seconds. Each IK update takes
            0.5 seconds, with linear joint interpolation within the update.
            This display timing is not physical robot speed.
        test: If True, create one headless frame and return the viewer handles.
        browser: Optional browser name accepted by Swift.

    Blue is the IK-generated tool path; gray marks the fixed goal. Large axes
    show the target and small axes show tool0. The label reports pose error
    and whether the provided loop limited a large raw joint update.

    Returns:
        Viewer handles and callbacks in test mode; otherwise runs until closed.
    """
    import spatialgeometry as sg
    import swift

    if not results or dt <= 0:
        raise ValueError("Provide histories and a positive playback interval")
    for result in results.values():
        theta = np.asarray(result["theta"])
        if theta.ndim != 2 or theta.shape[1] != 6 or not len(theta) or not np.isfinite(theta).all():
            raise ValueError("Joint histories must be finite arrays with shape (N, 6)")
        for key in ("targets", "achieved"):
            if np.asarray(result[key]).shape != (len(theta), 4, 4):
                raise ValueError("Pose histories must have shape (N, 4, 4)")
    robot, tool = load_ur7e()
    environment = swift.Swift()
    environment.launch(headless=test, browser=browser, axes=False, ground_pattern="@grid")
    from viewer_style import VIEWER_STYLE

    status = swift.Label(label="Loading UR7e meshes...", compact=True)
    environment.add_ui(swift.Label(label=VIEWER_STYLE, compact=True))
    environment.add_ui(status)
    environment.set_camera_pose([-1.5, -1.3, 0.9], [-0.45, -0.1, 0.25])
    names = list(results)
    state = {"selection": 0, "iteration": 0, "playing": not test, "playback_started": 0.0}
    result = results[names[0]]
    robot.q = result["theta"][0]
    target_axes = sg.Axes(length=0.12, arrows=True, pose=result["targets"][0])
    achieved_axes = sg.Axes(length=0.06, arrows=True, pose=robot.fkine(robot.q, end=tool))
    path = sg.Polyline(result["achieved"][:, :3, 3].T, linewidth=3, color=[0.12, 0.47, 0.71])
    desired = sg.Polyline(result["targets"][:, :3, 3].T, linewidth=2, color=[0.6, 0.6, 0.6])
    for shape in (target_axes, achieved_axes, path, desired):
        environment.add_shape(shape)

    def display(index):
        result = results[names[state["selection"]]]
        state["iteration"] = min(max(float(index), 0), len(result["theta"]) - 1)
        index = int(state["iteration"])
        fraction = state["iteration"] - index
        following = min(index + 1, len(result["theta"]) - 1)
        theta = result["theta"][index] + fraction * (result["theta"][following] - result["theta"][index])
        target_axes.T = result["targets"][index]
        achieved_axes.T = robot.fkine(theta, end=tool).A
        target_axes.update()
        achieved_axes.update()
        limited = " (display update limited)" if result["limited"][index] else ""
        status.label = (
            f"<b>{names[state['selection']]}</b> — iteration {index}/{len(result['theta']) - 1}<br>"
            f"Blue: achieved; gray: requested<br>"
            f"Position error: {1000 * result['position_error'][index]:.1f} mm<br>"
            f"Smallest singular value: {result['smallest_singular_value'][index]:.2e}<br>"
            f"Raw joint update: {result['raw_update'][index]:.3f} rad{limited}"
        )
        return theta

    def animate(time, _values):
        result = results[names[state["selection"]]]
        index = state["iteration"]
        if state["playing"]:
            index = min(len(result["theta"]) - 1, max(0.0, (monotonic() - state["playback_started"]) / 0.5))
            if index == len(result["theta"]) - 1:
                state["playing"] = False
                play.label = "Play"
        return display(index)

    handle = environment.add_robot(robot, readonly=True, callback=animate, name="UR7e")

    def restart(_value):
        handle.q = display(0)
        state["playback_started"] = monotonic()
        state["playing"] = True
        play.label = "Pause"

    def toggle(_value):
        result = results[names[state["selection"]]]
        if not state["playing"] and state["iteration"] == len(result["theta"]) - 1:
            restart(None)
            return
        state["playing"] = not state["playing"]
        state["playback_started"] = monotonic() - 0.5 * state["iteration"]
        play.label = "Pause" if state["playing"] else "Play"

    def select(value):
        state["selection"] = int(value)
        result = results[names[state["selection"]]]
        path.points = result["achieved"][:, :3, 3].T
        desired.points = result["targets"][:, :3, 3].T
        path.update()
        desired.update()
        state["playing"] = False
        play.label = "Play"
        handle.q = display(0)

    play = swift.Button(toggle, label="Play" if test else "Pause")
    for control in (
        swift.Select(select, label="Starting posture", options=names), play,
        swift.Button(restart, label="Restart"),
    ):
        environment.add_ui(control)
    handle.q = display(0)
    if test:
        return {"environment": environment, "handle": handle, "state": state,
                "select": select, "restart": restart, "animate": animate,
                "target_axes": target_axes, "status": status}
    state["playback_started"] = monotonic()
    environment.step(0.0)
    environment.run(dt=min(dt, 0.05))
