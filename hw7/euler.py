"""P3: ZYX Euler-angle singularities and manipulability."""

from pathlib import Path
import sys

import numpy as np

HW7_DIR = Path(__file__).resolve().parent
for parent in (HW7_DIR.parent, HW7_DIR.parent / "hw2" / "student"):
    if parent.is_dir() and str(parent) not in sys.path:
        sys.path.append(str(parent))

from jacobian_func_skeleton import spatial_jacobian
from hw2.kin_func_skeleton import R3_to_so3, so3_to_SO3


def euler_jacobian(theta):
    """Calculate the spatial angular Jacobian for ZYX Euler angles.

    Args:
        theta: NumPy array with shape ``(3,)`` containing angles in radians
            for ``R = Rz(theta[0]) @ Ry(theta[1]) @ Rx(theta[2])``.

    Returns:
        NumPy array with shape ``(3, 3)`` mapping the angle rates to the
        spatial angular velocity.
    """
    # TODO
    return ...


def manipulability(jacobian):
    """Calculate the product of all three singular values of a Jacobian.

    Args:
        jacobian: NumPy array with shape ``(3, 3)``.

    Returns:
        Nonnegative scalar. Include zero singular values, so a singular
        Jacobian has zero manipulability. ``np.linalg.svd`` may be useful.
    """
    # TODO
    return ...


def main():
    """Plot manipulability for two choices of the first and third angles."""
    import matplotlib.pyplot as plt
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    args = parser.parse_args()
    jacobian_function = euler_jacobian
    measure = manipulability
    if args.reference:
        from run_panda_ik import load_implementation

        reference = load_implementation(True)
        jacobian_function = reference.euler_jacobian
        measure = reference.manipulability

    theta2 = np.linspace(0, 2 * np.pi, 400, endpoint=False)
    figure, axes = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(8, 7))
    for axis, (theta1, theta3), color in zip(
        axes, ((0.0, 0.0), (0.4, -0.7)), ("tab:blue", "tab:orange"),
    ):
        values = []
        for angle in theta2:
            jacobian = jacobian_function(np.array([theta1, angle, theta3]))
            values.append(measure(jacobian))
        axis.plot(theta2, values, color=color,
                  label=f"theta1 = {theta1:g}, theta3 = {theta3:g}")
        axis.set_ylabel("Manipulability")
        axis.axvline(np.pi / 2, color="tab:red", linestyle="--", label="Singular angle")
        axis.axvline(3 * np.pi / 2, color="tab:red", linestyle="--")
        axis.legend(loc="upper right")
        axis.grid(True)
    axes[-1].set_xticks([0, np.pi / 2, np.pi, 3 * np.pi / 2, 2 * np.pi],
                       ["0", "pi/2", "pi", "3pi/2", "2pi"])
    axes[-1].set_xlabel("Second Euler angle (radians)")
    figure.suptitle("ZYX Euler Jacobian manipulability")
    figure.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
