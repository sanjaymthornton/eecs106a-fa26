# EECS C106A Fall 2026 Homework Skeletons

This public repository contains the starter code and public checks for EECS C106A homework. Keep one private repository for the semester, pull assignments from this repository, and submit your work through Gradescope.

## Assignments

| Assignment | Instructions | File to complete |
| --- | --- | --- |
| HW0: Python Bootcamp | [HW0 README](hw0/README.md) | `hw0/HW0.ipynb` |
| HW1: Vehicle Coordinate Transformations | Assignment handout and [starter docstring](hw1/hw1.py) | `hw1/hw1.py` |

The assignment handout provides the full problem statement and any additional submission requirements.

## One-time setup

1. On GitHub, create an **empty private repository** named `eecs106a-fa26`. Do not initialize it with a README, license, or `.gitignore`.

   **Note**: When creating the repository, set its visibility to private. If
   you accidentally create it as public, open the repository's **Settings**,
   select **General**, scroll down to **Danger Zone**, choose **Change
   repository visibility**, select **Change to private**, and complete the
   confirmation steps.

2. Clone this staff repository once:

   ```bash
   git clone https://github.com/ucb-ee106/fa26-hw-skeleton.git eecs106a-fa26
   cd eecs106a-fa26
   ```

3. Keep the public course repository as the `staff` remote and add your private repository as `origin`:

   ```bash
   git remote rename origin staff
   git remote add origin https://github.com/YOUR_GITHUB_USERNAME/eecs106a-fa26.git
   git push -u origin main
   ```

4. Confirm the remotes before doing coursework:

   ```bash
   git remote -v
   ```

`origin` must be your private repository. `staff` must be this public repository. Students should push their work to `origin`, not `staff`.

## Python environment

The course environment is tested with Python 3.13.5. Use Python 3.13 and one virtual environment for the semester. From the repository root on macOS or Linux:

```bash
python3 --version  # should report Python 3.13.x
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Activate this environment whenever you open a new terminal. Install the dependencies for the assignment you are working on:

```bash
python -m pip install -r hw0/requirements.txt  # HW0
python -m pip install -r hw1/requirements.txt  # HW1
```

For HW0, run `python -m jupyterlab` and open `hw0/HW0.ipynb`.

For HW1, implement `get_corners` in `hw1/hw1.py`, then run `python hw1/check_hw1.py`
and `python hw1/car_vis.py`. Use `python hw1/car_vis.py --path-only` to view the
trajectory before implementing the function, or add `--save parking.png` to
save a plot without opening a window.

## Getting a newly released assignment

Commit your current work first, then run:

```bash
git pull --no-rebase staff main
git push origin main
```

Install the new assignment's dependencies and follow its handout and repository instructions. If Git reports conflicts, resolve them while preserving your answers before committing and pushing the merge. Ask course staff for help if you are unsure.

## Submitting to Gradescope

Run the assignment's local checks and save your work. Commit and push only the files you intend to submit. For example, for HW1:

```bash
git status
git add hw1/hw1.py
git commit -m "Complete HW1"
git push origin main
```

For HW0, stage `hw0/HW0.ipynb` instead. Keep each assignment in its original directory.

Open Gradescope directly, choose the assignment and GitHub submission method, then select your private `eecs106a-fa26` repository and its `main` branch. Verify that the intended files and latest commit are included, and review the grading results. After making corrections, commit, push, and resubmit.

Saving locally or pushing to GitHub does not create a Gradescope submission. Passing public checks does not guarantee full credit on the additional grading tests.

## Repository hygiene

Keep your coursework repository private. Do not commit credentials, access tokens, or unrelated private files. Generated `.OTTER_LOG`, `__pycache__/`, notebook checkpoints, and virtual environments are ignored automatically; they are not assignment deliverables.
