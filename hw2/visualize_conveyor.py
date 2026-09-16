"""View the box and camera motion from HW2 in a browser.

This is an optional, provided visualization and is not graded. Run it from the
repository root with:

    python hw2/visualize_conveyor.py

The animation uses the functions in your hw2.py file. You can change the
example values in main or call visualize_conveyor with your own values.
"""

import sys

import numpy as np


VIEWER_STYLE = """
<style>
  .info, .control-panel {
    display: none !important;
  }
  .sidenav {
    width: min(500px, calc(100vw - 32px));
    min-width: 0;
    max-width: none;
    height: auto;
    top: 16px;
    left: 16px;
    padding: 13px 16px 10px;
    overflow: hidden;
    border: 1px solid rgba(255, 255, 255, 0.18);
    border-radius: 12px;
    background: rgba(30, 30, 30, 0.88);
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.28);
    backdrop-filter: blur(8px);
  }
  .sidenav .label-div:first-child {
    display: none;
  }
  .sidenav .label-div {
    margin: 0 0 9px !important;
  }
  .sidenav .label-div p {
    width: 100% !important;
    margin: 0 !important;
    color: white;
    font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
    font-size: 13px !important;
    text-align: left;
  }
  .sidenav .slider-div {
    margin: 0;
  }
  .sidenav .slider-p {
    display: none;
  }
  .sidenav .slider {
    width: 100%;
  }
  .sidenav .slider-val-div {
    width: 100%;
    margin: 7px 0 0;
  }
  .sidenav .button-div {
    width: 100%;
    margin: 9px 0 0;
  }
  .sidenav .button-button {
    width: 100%;
    min-width: 0;
    max-width: none;
    border-radius: 6px;
    cursor: pointer;
  }
</style>
"""


def visualize_conveyor(
    box_speed=0.175,
    camera_angular_speed=0.10,
    box_distance=1.2,
    camera_height=1.0,
    duration=12.0,
    test=False,
    box_pose_function=None,
    camera_pose_function=None,
    relative_pose_function=None,
    box_twist_function=None,
    camera_twist_function=None,
):
    """Display the HW2 box and camera motion.

    Args:
        box_speed: linear speed of the box, in meters per second.
        camera_angular_speed: positive magnitude of the camera's clockwise
            angular speed, in radians per second.
        box_distance: distance across the belt from the post to the box, in
            meters.
        camera_height: height of the camera above the belt, in meters.
        duration: number of seconds shown by the time slider.
        test: If True, create one headless frame and return without opening a
            browser. Students do not need to use this argument.
        box_pose_function: function with the same inputs and output as
            box_pose_in_world. If omitted, the function from hw2.py is used.
        camera_pose_function: function with the same inputs and output as
            camera_pose_in_world. If omitted, the function from hw2.py is used.
        relative_pose_function: function with the same inputs and output as
            box_pose_in_camera. If omitted, the function from hw2.py is used.
        box_twist_function: function with the same inputs and output as
            box_twist_in_world. If omitted, the function from hw2.py is used.
        camera_twist_function: function with the same inputs and output as
            camera_twist_in_world. If omitted, the function from hw2.py is
            used.

    The red, green, and blue arrows show the x, y, and z axes of each frame.
    The yellow line shows the camera's positive z-axis, which is where it's
    looking. The text in the corner shows the position of the box in the camera
    frame.

    Returns:
        When test is True, returns the Swift environment and the displayed
        shapes. Otherwise, the function runs until the browser closes.
    """
    try:
        import spatialgeometry as sg
        import swift
        from spatialmath import SE3
    except ImportError as error:
        raise SystemExit(
            "Install the HW2 requirements before running the visualizer: "
            "python -m pip install -r hw2/requirements.txt"
        ) from error

    if (
        box_pose_function is None
        or camera_pose_function is None
        or relative_pose_function is None
        or box_twist_function is None
        or camera_twist_function is None
    ):
        from hw2 import (
            box_pose_in_camera,
            box_pose_in_world,
            box_twist_in_world,
            camera_pose_in_world,
            camera_twist_in_world,
        )

        if box_pose_function is None:
            box_pose_function = box_pose_in_world
        if camera_pose_function is None:
            camera_pose_function = camera_pose_in_world
        if relative_pose_function is None:
            relative_pose_function = box_pose_in_camera
        if box_twist_function is None:
            box_twist_function = box_twist_in_world
        if camera_twist_function is None:
            camera_twist_function = camera_twist_in_world

    environment = swift.Swift()
    environment.launch(
        realtime=True,
        headless=test,
        axes=False,
        ground_opacity=0.0,
        ground_pattern=False,
    )

    page_style = swift.Label(label=VIEWER_STYLE, compact=True)
    motion_text = swift.Label(compact=True)
    slider_step = 0.05
    time_slider = swift.Slider(
        min=0,
        max=duration,
        step=slider_step,
        value=0,
        label="",
        precision=2,
    )

    state = {
        "playing": True,
        "start_sim_time": 0.0,
        "start_time": 0.0,
        "time": 0.0,
        "automatic_slider_values": [],
    }

    def animation_time(sim_time):
        if state["playing"]:
            elapsed = sim_time - state["start_sim_time"]
            return (state["start_time"] + elapsed) % duration
        return state["time"]

    def box_pose_at(time):
        return np.asarray(box_pose_function(time, box_speed, box_distance))

    def camera_pose_at(time):
        return np.asarray(
            camera_pose_function(time, camera_angular_speed, camera_height)
        )

    def integrate_twist(twist, time):
        from scipy.linalg import expm

        velocity = twist[:3]
        omega = twist[3:]
        twist_hat = np.array([
            [0, -omega[2], omega[1], velocity[0]],
            [omega[2], 0, -omega[0], velocity[1]],
            [-omega[1], omega[0], 0, velocity[2]],
            [0, 0, 0, 0],
        ])
        return expm(twist_hat * time)

    box_size = np.array([0.32, 0.28, 0.24])
    optical_axis_length = max(0.8, box_distance)
    belt_length = max(3.0, abs(box_speed) * duration + 1.2)
    belt_center_x = box_speed * duration / 2

    belt = sg.Cuboid(
        scale=[belt_length, 0.80, 0.12],
        pose=SE3(belt_center_x, box_distance, -0.06),
        color=[0.22, 0.24, 0.27],
    )
    post = sg.Cylinder(
        radius=0.045,
        length=camera_height,
        pose=SE3(0, 0, camera_height / 2),
        color=[0.35, 0.35, 0.38],
    )
    box = sg.Cuboid(scale=box_size, color=[0.88, 0.48, 0.12])
    camera = sg.Cuboid(scale=[0.22, 0.14, 0.16], color=[0.15, 0.18, 0.22])
    lens = sg.Cylinder(radius=0.055, length=0.08, color=[0.08, 0.10, 0.12])
    optical_axis = sg.Cylinder(
        radius=0.007,
        length=optical_axis_length,
        color=[0.95, 0.78, 0.12],
    )
    world_axes = sg.Axes(length=0.35, arrows=True)
    box_axes = sg.Axes(length=0.30, arrows=True)
    camera_axes = sg.Axes(length=0.30, arrows=True)

    displayed_shapes = {
        "belt": belt,
        "post": post,
        "box": box,
        "camera": camera,
        "lens": lens,
        "optical_axis": optical_axis,
        "world_axes": world_axes,
        "box_axes": box_axes,
        "camera_axes": camera_axes,
    }

    environment.add_shape(belt)
    environment.add_shape(post)
    environment.add_shape(world_axes)

    def update_text(time):
        relative_pose = np.asarray(
            relative_pose_function(
                time,
                box_speed,
                camera_angular_speed,
                box_distance,
                camera_height,
            )
        )
        position = relative_pose[:3, 3]
        box_from_twist = (
            integrate_twist(np.asarray(box_twist_function(box_speed)), time)
            @ box_pose_at(0)
        )
        camera_from_twist = (
            integrate_twist(
                np.asarray(camera_twist_function(camera_angular_speed)), time
            )
            @ camera_pose_at(0)
        )
        box_twist_matches = np.allclose(box_from_twist, box_pose_at(time))
        camera_twist_matches = np.allclose(
            camera_from_twist, camera_pose_at(time)
        )
        twist_status = (
            "match" if box_twist_matches and camera_twist_matches
            else "do not match"
        )
        if np.isclose(position[0], 0):
            horizontal_location = "centered horizontally"
        elif position[0] > 0:
            horizontal_location = "right of the optical axis"
        else:
            horizontal_location = "left of the optical axis"
        motion_text.label = (
            f"t = {time:.2f} s<br>"
            f"box position in camera frame = "
            f"[{position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f}] m<br>"
            f"box is {horizontal_location}<br>"
            f"pose and twist calculations {twist_status}"
        )

    def update_animation(sim_time, _values):
        time = animation_time(sim_time)
        state["time"] = time
        slider_value = round(time / slider_step) * slider_step
        state["automatic_slider_values"].append(slider_value)
        state["automatic_slider_values"] = state["automatic_slider_values"][-4:]
        time_slider.value = slider_value
        update_text(time)
        return SE3(box_pose_at(time)) @ SE3(0, 0, box_size[2] / 2)

    def box_axes_pose(sim_time, _values):
        return SE3(box_pose_at(animation_time(sim_time)))

    def camera_body_pose(sim_time, _values):
        return SE3(camera_pose_at(animation_time(sim_time))) @ SE3(0, 0, -0.08)

    def camera_lens_pose(sim_time, _values):
        return SE3(camera_pose_at(animation_time(sim_time))) @ SE3(0, 0, 0.04)

    def camera_axes_pose(sim_time, _values):
        return SE3(camera_pose_at(animation_time(sim_time)))

    def optical_axis_pose(sim_time, _values):
        camera_pose = SE3(camera_pose_at(animation_time(sim_time)))
        return camera_pose @ SE3(0, 0, optical_axis_length / 2)

    environment.add_shape(box, callback=update_animation)
    environment.add_shape(box_axes, callback=box_axes_pose)
    environment.add_shape(camera, callback=camera_body_pose)
    environment.add_shape(lens, callback=camera_lens_pose)
    environment.add_shape(camera_axes, callback=camera_axes_pose)
    environment.add_shape(optical_axis, callback=optical_axis_pose)

    def select_time(value):
        # Swift reports animation-driven slider updates through the same
        # callback as a user drag. Ignore only values the animation just sent.
        for index, automatic_value in enumerate(state["automatic_slider_values"]):
            if np.isclose(value, automatic_value, atol=1e-6):
                del state["automatic_slider_values"][:index + 1]
                return
        state["automatic_slider_values"].clear()
        state["playing"] = False
        state["time"] = float(value)
        play_button.label = "Play"
        update_text(state["time"])

    time_slider.cb = select_time

    def toggle_animation(_value):
        state["playing"] = not state["playing"]
        if state["playing"]:
            state["start_time"] = state["time"]
            state["start_sim_time"] = environment.sim_time
            play_button.label = "Pause"
        else:
            state["time"] = animation_time(environment.sim_time)
            time_slider.value = state["time"]
            play_button.label = "Play"

    play_button = swift.Button(toggle_animation, label="Pause")

    environment.add_ui(page_style)
    environment.add_ui(motion_text)
    environment.add_ui(time_slider)
    environment.add_ui(play_button)

    environment.set_camera_pose(
        [belt_center_x + 2.5, -2.7, 2.2],
        [belt_center_x, box_distance / 2, 0.45],
    )

    if test:
        environment.step()
        return environment, displayed_shapes

    environment.run()


def main(test=False):
    """Run the provided conveyor example using the functions in hw2.py."""
    return visualize_conveyor(test=test)


if __name__ == "__main__":
    main(test="--test" in sys.argv)
