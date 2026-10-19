"""Play HW6 desired and integrated chair-swing trajectories."""

import numpy as np
from colorsys import hsv_to_rgb

def visualize_results(results, dt=0.03, speed=4.0, test=False, browser=None, start_time=0.0):
    """Display trajectory integration results with timestep selection.

    Args:
        results: Dictionary mapping names to run_trajectory.simulate results.
            Each contains times (N,), poses and desired (N, 4, 4), and
            position_error and orientation_error (N,). desired_path (M, 3)
            optionally provides densely sampled desired positions.
        dt: Positive interval between viewer updates in seconds.
        speed: Positive multiplier on playback time, default 4x. All histories advance
            in physical time, independent of their integration timesteps.
        test: If True, create one headless frame and return viewer handles.
        browser: Optional browser name accepted by Swift.
        start_time: Nonnegative playback start time in seconds. This does
            not change the integration's initial pose or reset its history.

    Gray shows the desired ride and path; blue shows the integrated chair.
    Its trail grows across laps, colored blue to red by elapsed simulation time.
    The blue chair retains its full SE(3) pose, without IK or
    projection onto the ride's joints. Its chain may drift away from the
    support as integration error builds up. Select a timestep, then Play
    or Restart. Switching timesteps keeps the current playback time;
    Restart returns to zero.

    Returns:
        Viewer handles and callbacks in test mode; otherwise runs until closed.
    """
    import spatialgeometry as sg
    import swift
    from spatialmath import SE3
    from swing_model import load_swing, swing_joint_values

    if (not results or dt <= 0 or not np.isfinite(speed) or speed <= 0
            or not np.isfinite(start_time) or start_time < 0):
        raise ValueError("Provide histories and positive playback settings")
    ride, chair = load_swing()
    names = list(results)
    histories = {name: np.array([swing_joint_values(pose) for pose in result["desired"]])
                 for name, result in results.items()}
    state = {"selection": 0, "time": 0.0, "index": 0, "playing": False, "last_time": 0.0}
    result = results[names[0]]
    ride.q = histories[names[0]][0]
    chair.base = SE3(result["poses"][0])
    environment = swift.Swift()
    environment.launch(headless=test, browser=browser, axes=False, ground_pattern="@grid")
    target_axes = sg.Axes(length=0.10, arrows=True, pose=result["desired"][0])
    achieved_axes = sg.Axes(length=0.05, arrows=True, pose=result["poses"][0])
    desired_path = sg.Polyline(result.get("desired_path", result["desired"][:, :3, 3]).T,
                               linewidth=2, color=[0.6, 0.6, 0.6])
    trail_count = 64
    trail_bounds = {
        name: np.searchsorted(result["times"],
                              np.linspace(result["times"][0], result["times"][-1], trail_count + 1))
        for name, result in results.items()
    }
    actual_paths = [
        sg.Polyline(np.repeat(result["poses"][0, :3, 3, None], 2, axis=1),
                    linewidth=3,
                    color=(*hsv_to_rgb((2 / 3) * (1 - index / (trail_count - 1)), 1, 1), 0))
        for index in range(trail_count)
    ]
    trail_samples = [None] * trail_count
    for shape in (target_axes, achieved_axes, desired_path, *actual_paths):
        environment.add_shape(shape)
    status = swift.Label(compact=True)

    def display(time):
        name = names[state["selection"]]
        result = results[name]
        state["time"] = min(max(float(time), 0.0), result["times"][-1])
        index = max(0, np.searchsorted(result["times"], state["time"], side="right") - 1)
        state["index"] = index
        for part, path in enumerate(actual_paths):
            first = trail_bounds[name][part]
            last = min(index, trail_bounds[name][part + 1])
            if last <= first:
                if trail_samples[part] is not None:
                    path.points = np.repeat(result["poses"][0, :3, 3, None], 2, axis=1)
                    path.opacity = 0
                    path.update()
                    trail_samples[part] = None
                continue
            samples = (name, first, last)
            if trail_samples[part] != samples:
                path.points = result["poses"][first:last + 1, :3, 3].T
                path.opacity = 1
                path.update()
                trail_samples[part] = samples
        theta = histories[name][index]
        target_axes.T = result["desired"][index]
        chair.base = SE3(result["poses"][index])
        achieved_axes.T = result["poses"][index]
        target_axes.update()
        achieved_axes.update()
        status.label = (f"<b>{name}</b><br>Gray: desired; blue chair: integrated<br>"
                        f"Trail time: 0 → {result['times'][-1]:.1f} s"
                        '<div style="height: 6px; background: linear-gradient(to right,'
                        'blue, cyan, lime, yellow, red); margin: 4px 0;"></div>'
                        f"t = {result['times'][index]:.2f} s; "
                        f"position error = {1000 * result['position_error'][index]:.1f} mm")
        return theta

    def animate(time, _values):
        if state["playing"]:
            display(state["time"] + speed * max(0, time - state["last_time"]))
            if state["time"] == results[names[state["selection"]]]["times"][-1]:
                state["playing"] = False
                play.label = "Play"
        state["last_time"] = time
        return histories[names[state["selection"]]][state["index"]]

    handle = environment.add_robot(ride, readonly=True, callback=animate, name="Desired swing")
    chair_handle = environment.add_robot(chair, readonly=True, name="Integrated chair")

    def restart(_value):
        handle.q = display(0.0)
        state["last_time"] = environment.sim_time
        state["playing"] = True
        play.label = "Pause"

    def toggle(_value):
        if not state["playing"] and state["time"] == results[names[state["selection"]]]["times"][-1]:
            restart(None)
            return
        state["playing"] = not state["playing"]
        state["last_time"] = environment.sim_time
        play.label = "Pause" if state["playing"] else "Play"

    def select(value):
        time = state["time"]
        state["selection"] = int(value)
        result = results[names[state["selection"]]]
        desired_path.points = result.get("desired_path", result["desired"][:, :3, 3]).T
        desired_path.update()
        state["playing"] = False
        play.label = "Play"
        handle.q = display(time)

    style = """<style>
.info, .control-panel { display: none !important; }
.sidenav { width: 290px; height: auto; top: 16px; left: 16px; padding: 14px;
border-radius: 12px; background: rgba(30,30,30,0.9); color: white; }
.sidenav .label-div:first-child { display: none; }
.sidenav .label-div p { width: 100% !important; text-align: left; color: white; }
.sidenav .button-button { width: 100%; cursor: pointer; }
</style>"""
    play = swift.Button(toggle, label="Play")
    for control in (swift.Label(label=style, compact=True), status,
                    swift.Select(select, label="Integration timestep", options=names),
                    play, swift.Button(restart, label="Restart")):
        environment.add_ui(control)
    environment.set_camera_pose([1.1, 1.0, 0.95], [0.0, 0.0, 0.43])
    if test:
        environment.step(0.0)
        handle.q = display(start_time)
        return {"environment": environment, "handle": handle, "state": state,
                "display": display, "select": select, "restart": restart,
                "animate": animate, "target_axes": target_axes,
                "desired_path": desired_path, "actual_paths": actual_paths,
                "ride": ride, "chair": chair,
                "chair_handle": chair_handle}
    handle.q = display(start_time)
    environment.run(dt=dt)


if __name__ == "__main__":
    from run_trajectory import main

    main(visualize=True)
