"""Public sanity checker for HW4: UR7e Inverse Kinematics.

Run with:
    python hw4/check_hw4.py
or from inside the hw4 directory:
    python check_hw4.py
"""

import sys
from pathlib import Path
import numpy as np

# Robust import path setup independent of current working directory
HW4_DIR = Path(__file__).resolve().parent
if str(HW4_DIR) not in sys.path:
    sys.path.insert(0, str(HW4_DIR))

# The repository keeps one shared copy of kin_func_skeleton.py under hw2/.
KIN_PARENT = HW4_DIR.parent / "hw2" / "student"
if not KIN_PARENT.is_dir():
    KIN_PARENT = HW4_DIR.parent
if str(KIN_PARENT) not in sys.path:
    sys.path.insert(0, str(KIN_PARENT))

import hw4
import ur7e_model


def run_checks() -> bool:
    all_passed = True

    def run_check(name: str, test_fn):
        nonlocal all_passed
        try:
            test_fn()
            print(f"[PASS] {name}")
        except AssertionError as e:
            all_passed = False
            msg = str(e) if str(e) else "Assertion failed"
            print(f"[FAIL] {name}: {msg}")
        except Exception as e:
            all_passed = False
            print(f"[FAIL] {name}: Raised {type(e).__name__}: {e}")

    # Shared target for the verification and pipeline examples.
    q_goal = np.array([0.2, -0.8, 1.2, -0.5, 0.4, -0.3], dtype=np.float64)
    q_curr = np.array([0.1, -0.7, 1.1, -0.4, 0.3, -0.2], dtype=np.float64)
    desired_end_effector_pose = ur7e_model.robot.fwdKin(q_goal)

    # 1. Check one zero and one nonzero forward-kinematics example.
    def check_ur7e_fk():
        zero_pose = hw4.ur7e_fk(np.zeros(6))
        if not isinstance(zero_pose, np.ndarray) or zero_pose.shape != (4, 4):
            raise AssertionError("Return a NumPy array with shape (4, 4)")
        if not np.allclose(zero_pose, ur7e_model.UR7E_ZERO_POSE, atol=1e-8):
            raise AssertionError("At q = 0, return UR7E_ZERO_POSE")

        actual = hw4.ur7e_fk(q_goal)
        expected = ur7e_model.robot.fwdKin(q_goal)
        if not np.allclose(actual, expected, atol=1e-8):
            raise AssertionError("The nonzero forward-kinematics example is incorrect")

    run_check("1. ur7e_fk matches zero and nonzero examples", check_ur7e_fk)

    # 2. q_goal reaches the desired pose, while the zero configuration does not.
    def check_verify_ik_solutions():
        candidates = np.array([q_goal, np.zeros(6)])
        verified = hw4.verify_ik_solutions(desired_end_effector_pose, candidates)
        expected = np.array([q_goal])
        if verified.shape != expected.shape or not np.allclose(verified, expected):
            raise AssertionError("Keep q_goal and reject the zero configuration")

    run_check("2. verify_ik_solutions keeps one valid candidate", check_verify_ik_solutions)

    # 3. These three simple rows are preferred by three different norms.
    def check_closest_solutions_by_norm():
        solutions = np.array(
            [
                [1.8, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.8, 0.8, 0.8, 0.0, 0.0, 0.0],
                [0.7, 0.7, 0.7, 0.7, 0.7, 0.7],
            ]
        )
        actual = hw4.closest_solutions_by_norm(solutions, np.zeros(6))
        expected = (solutions[0], solutions[1], solutions[2])
        if len(actual) != 3 or any(
            not np.allclose(got, want) for got, want in zip(actual, expected)
        ):
            raise AssertionError("Expected rows 0, 1, and 2 for the three norms")

    run_check("3. each norm selects its expected row", check_closest_solutions_by_norm)

    # 4. The same first-joint angle can be represented two pi apart.
    def check_select_solutions_by_objective():
        q_base = np.array([-0.1 * np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])
        q_equivalent = q_base.copy()
        q_equivalent[0] += 2 * np.pi
        q_invalid = q_base.copy()
        q_invalid[0] += 4 * np.pi
        solutions = np.array([q_base, q_equivalent, q_invalid])
        q_current = np.array([1.8 * np.pi, 0.0, 0.0, 0.0, 0.0, 0.0])

        closest_current, closest_zero, valid = (
            hw4.select_solutions_by_objective(solutions, q_current)
        )
        if valid.shape != (2, 6) or not np.allclose(valid, solutions[:2]):
            raise AssertionError("Keep only solutions between -2*pi and 2*pi")
        if not np.allclose(closest_current, q_equivalent):
            raise AssertionError("Select the valid solution closest to q_current")
        if not np.allclose(closest_zero, q_base):
            raise AssertionError("Select the valid solution closest to zero")

    run_check(
        "4. select_solutions_by_objective applies both objectives",
        check_select_solutions_by_objective,
    )

    # 5. Compare the pipeline with the same helper calls written out directly.
    def check_solve_ur7e_pose():
        candidates = ur7e_model.robot.IK(desired_end_effector_pose).Q
        verified = hw4.verify_ik_solutions(desired_end_effector_pose, candidates)
        expected = hw4.select_solutions_by_objective(verified, q_curr)
        actual = hw4.solve_ur7e_pose(desired_end_effector_pose, q_curr)
        if len(actual) != 3 or any(
            not np.allclose(got, want) for got, want in zip(actual, expected)
        ):
            raise AssertionError("Generate, verify, and select in that order")

    run_check("5. solve_ur7e_pose applies the complete IK pipeline", check_solve_ur7e_pose)

    return all_passed


if __name__ == "__main__":
    success = run_checks()
    sys.exit(0 if success else 1)
