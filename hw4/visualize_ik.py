"""View inverse-kinematics solutions on an official UR7e model.

This is an optional, provided example and is not graded. Run it from the
repository root with:

    python hw4/visualize_ik.py

The first run downloads the official UR7e description and meshes. Use the
slider in the Swift browser window to move between the IK solutions, or press
Play to animate between them. The colored axes mark the end-effector pose
reached by the first candidate.
"""

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


VIEWER_STYLE = """
<style>
  /* Keep Swift's scene, but replace its full sidebar with a small overlay. */
  .info, .control-panel {
    display: none !important;
  }
  .sidenav {
    width: min(540px, calc(100vw - 32px));
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
  /* This hidden first label supplies these page styles. */
  .sidenav .label-div:first-child {
    display: none;
  }
  .sidenav .label-div {
    margin: 0 0 11px !important;
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


def visualize_ik(candidates, test=False):
    """Display candidate IK solutions on the UR7e.

    Args:
        candidates: NumPy array with shape ``(N, 6)``. Each row contains one
            candidate set of six joint angles in radians.
        test: If True, create one headless frame and return without opening a
            browser. Students do not need to use this argument.

    The colored axes show the end-effector pose computed from the first
    candidate using the UR7e model's forward kinematics. Use the slider to
    display one row of ``candidates``, or press Play to cycle through every
    row. The animation jumps directly between candidates, so every pose shown
    is one of the supplied IK solutions.

    Returns:
        When ``test`` is True, returns the UR7e model, its Swift handle, and
        ``candidates``. Otherwise, the function runs until the browser closes.
    """
    try:
        import spatialgeometry as sg
        import swift
        from roboticstoolbox.models.URDF.URDFRobot import URDFRobot
    except ImportError as error:
        raise SystemExit(
            "Install the HW4 requirements before running the visualizer: "
            "python -m pip install -r hw4/requirements.txt"
        ) from error

    ur7e = URDFRobot("ur7e")
    tool0 = next(link for link in ur7e.ee_links if link.name == "tool0")
    ur7e.q = candidates[0]

    environment = swift.Swift()
    environment.launch(
        headless=test,
        axes=True,
        ground_pattern="@grid",
    )

    page_style = swift.Label(label=VIEWER_STYLE, compact=True)
    angles = swift.Label(compact=True)
    solution_slider = swift.Slider(
        min=1,
        max=len(candidates),
        step=1,
        value=1,
        label="",
        precision=0,
    )

    state = {
        "playing": False,
        "solution": 0,
        "last_change": 0.0,
    }

    def display_angles(q):
        displayed = ", ".join(f"{angle:.2f}" for angle in q)
        angles.label = f"q = [{displayed}] radians"

    def animate_solution(time, _values):
        if not state["playing"]:
            return robot_handle.q

        if time - state["last_change"] >= 0.8:
            state["solution"] = (state["solution"] + 1) % len(candidates)
            state["last_change"] = time
            solution_slider.value = state["solution"] + 1
            display_angles(candidates[state["solution"]])

        return candidates[state["solution"]]

    robot_handle = environment.add_robot(
        ur7e,
        readonly=True,
        callback=animate_solution,
    )

    def toggle_animation(_value):
        state["playing"] = not state["playing"]
        if state["playing"]:
            state["last_change"] = environment.sim_time
            play_button.label = "Pause"
        else:
            play_button.label = "Play"

    play_button = swift.Button(toggle_animation, label="Play")

    def show_solution(value):
        index = int(value) - 1
        # Updating the slider during playback also reports a slider event.
        # That event reflects the animation itself, not a manual selection.
        if state["playing"] and index == state["solution"]:
            return
        state["playing"] = False
        state["solution"] = index
        play_button.label = "Play"
        robot_handle.q = candidates[index]
        display_angles(candidates[index])

    solution_slider.cb = show_solution
    show_solution(1)

    environment.add_ui(page_style)
    environment.add_ui(angles)
    environment.add_ui(solution_slider)
    environment.add_ui(play_button)
    environment.add_shape(
        sg.Axes(
            length=0.15,
            arrows=True,
            pose=ur7e.fkine(candidates[0], end=tool0),
        )
    )
    environment.set_camera_pose([1.4, 1.4, 1.1], [0.0, 0.0, 0.35])

    if test:
        environment.step()
        return ur7e, robot_handle, candidates

    environment.run()


def main(test=False):
    """Run the provided EAIK example in the reusable visualizer."""
    from hw4 import eaik_example

    _, candidates = eaik_example()
    return visualize_ik(candidates, test=test)


if __name__ == "__main__":
    main(test="--test" in sys.argv)
