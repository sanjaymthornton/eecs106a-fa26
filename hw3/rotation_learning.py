"""Starter code for Rotation Representations in Robot Learning.

Every representation in this problem is the same pipeline with two pieces
swapped out:

    labels = encode(rotations)      # Rotation -> (N, d) numbers to regress
    model  = fit(X[train], labels[train])
    R_hat  = decode(predict(model, X_sweep))   # (N, d) raw -> (N, 3, 3)

So the only thing you write is an ``encode``/``decode`` pair per
representation. The PROVIDED code trains, evaluates, and plots all of them
through one generic runner.

The main provided calls are:

    model  = fit(X, Y)
    Yhat   = predict(model, X)
    errors = rotation_errors(R1, R2)

Do not modify the PROVIDED sections. Write your solution in the YOUR CODE
section.
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as Rot

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from hw2.kin_func_skeleton import R3_to_so3, so3_to_R3, so3_to_SO3, SO3_to_so3, renormalize_SO3

TWO_LAPS = 4 * np.pi


# ============================= PROVIDED: NETWORK =============================

_H = 64  # hidden width

def _shapes(dout):
    return [(2, _H), (_H,), (_H, _H), (_H,), (_H, dout), (dout,)]

def _unpack(w, dout):
    ps, o = [], 0
    for sh in _shapes(dout):
        n = int(np.prod(sh)); ps.append(w[o:o + n].reshape(sh)); o += n
    return ps

def _loss_grad(w, X, Y, dout):
    """Mean squared error and its gradient for the two-hidden-layer MLP.

    The network is

        h1  = tanh(X  @ W1 + b1)        (N, H)
        h2  = tanh(h1 @ W2 + b2)        (N, H)
        out = h2 @ W3 + b3              (N, dout)

    and the loss is the mean squared error over all N * dout output entries,
    L = mean((out - Y) ** 2).

    ``fit`` hands the whole parameter set to SciPy's L-BFGS-B as a single flat
    vector, so ``w`` holds (W1, b1, W2, b2, W3, b3) concatenated in the order
    listed by ``_shapes``; ``_unpack`` splits it apart as ``ps[0] ... ps[5]``,
    and the returned gradient is concatenated in that same order so the two
    line up entry for entry. L-BFGS-B is called with ``jac=True``, which is
    why the loss and its gradient are returned together from one combined
    forward and backward pass.

    The backward pass reuses a single array ``d``, which always holds
    dL/d(pre-activation) for the layer currently being differentiated. It
    starts as the gradient with respect to the network output,

        dL/d(out) = 2 * (out - Y) / (N * dout),

    where the 1 / (N * dout) matches the averaging in the loss. Then for each
    layer, working from the output backwards:

        dL/d(bias)   = d summed over the batch
        dL/d(weight) = (that layer's input).T @ d

    and stepping back one layer multiplies by the weights, ``d @ W.T``, and
    then by the tanh derivative ``1 - tanh(z) ** 2``. That derivative is
    written as ``1 - h ** 2`` because ``h`` is the stored tanh output, which
    is also why the forward pass keeps ``h1`` and ``h2`` around.

    Args:
        w: Flat parameter vector.
        X: Inputs with shape ``(N, 2)``.
        Y: Targets with shape ``(N, dout)``.
        dout: Output width, needed to recover the parameter shapes.

    Returns:
        A tuple of the scalar mean squared error and the flat gradient, which
        has the same length as ``w``.
    """
    ps = _unpack(w, dout)
    # Forward pass. h1 and h2 are retained for use in the backward pass.
    h1 = np.tanh(X @ ps[0] + ps[1]); h2 = np.tanh(h1 @ ps[2] + ps[3])
    out = h2 @ ps[4] + ps[5]
    d = 2 * (out - Y) / (X.shape[0] * Y.shape[1])  # dL/d(out), shape (N, dout)
    g5, g4 = d.sum(0), h2.T @ d                    # dL/db3, dL/dW3
    d = (d @ ps[4].T) * (1 - h2 ** 2)              # back through W3, then tanh
    g3, g2 = d.sum(0), h1.T @ d                    # dL/db2, dL/dW2
    d = (d @ ps[2].T) * (1 - h1 ** 2)              # back through W2, then tanh
    g1, g0 = d.sum(0), X.T @ d                     # dL/db1, dL/dW1
    return np.mean((out - Y) ** 2), np.concatenate([g.ravel() for g in (g0, g1, g2, g3, g4, g5)])

def fit(X, Y, seed=0):
    """Train the MLP f: R^2 -> R^d by minimizing mean squared error. Returns a model."""
    dout = Y.shape[1]
    r = np.random.default_rng(seed)
    w0 = np.concatenate([(r.normal(0, np.sqrt(2 / sh[0]), sh) if len(sh) == 2
                          else np.zeros(sh)).ravel() for sh in _shapes(dout)])
    res = minimize(_loss_grad, w0, args=(X, Y, dout), jac=True,
                   method='L-BFGS-B', options={'maxiter': 20000, 'maxfun': 10 ** 7})
    return (res.x, dout)

def predict(model, X):
    """Network outputs for inputs X (N,2) -> (N,d)."""
    w, dout = model
    ps = _unpack(w, dout)
    return np.tanh(np.tanh(X @ ps[0] + ps[1]) @ ps[2] + ps[3]) @ ps[4] + ps[5]


# ============================= PROVIDED: ERROR METRICS =============================

def angular_error_deg(R1, R2):
    """Angle in degrees between two rotation matrices with shape ``(3, 3)``.

    The rotation carrying ``R1`` to ``R2`` is ``R1.T @ R2``, whose angle
    ``theta`` satisfies ``trace(R1.T @ R2) = 1 + 2 * cos(theta)``. Inverting
    that and converting to degrees gives the error reported everywhere in
    this problem, so a value of 180 means the two rotations are as far apart
    as rotations can be.
    """
    cosine = (np.trace(R1.T @ R2) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))

def rotation_errors(R1, R2):
    """Angular error in degrees for corresponding pairs of rotations.

    Vectorized equivalent of calling ``angular_error_deg`` on each pair;
    ``einsum`` contracts to ``trace(R1.T @ R2)`` for every pair at once.

    Args:
        R1: NumPy array with shape ``(N, 3, 3)``.
        R2: NumPy array with shape ``(N, 3, 3)``.

    Returns:
        NumPy array with shape ``(N,)``.
    """
    cosine = (np.einsum("nij,nij->n", R1, R2) - 1.0) / 2.0
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# ============================= PROVIDED: REPRESENTATIONS =============================

@dataclass(frozen=True)
class Representation:
    """One way of writing a rotation as a vector of numbers.

    Attributes:
        name: Short identifier, also used for the output plot filename.
        encode: Maps a SciPy ``Rotation`` holding N rotations to ``(N, d)``
            regression targets.
        decode: Maps ``(N, d)`` raw network outputs to ``(N, 3, 3)`` valid
            rotation matrices.
        label_names: One legend entry per output dimension.
        title: Title for the label panel of the plot.
    """
    name: str
    encode: Callable
    decode: Callable
    label_names: tuple
    title: str

def on_matrices(encoder):
    """Adapt an encoder written for ``(N, 3, 3)`` arrays to take a ``Rotation``."""
    return lambda rotations: encoder(rotations.as_matrix())


# ============================= PROVIDED: PLOTTING =============================

def plot_result(s, result, output_dir):
    """Plot one representation's labels and its decoded sweep error."""
    import matplotlib.pyplot as plt

    representation = result["representation"]
    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for index, name in enumerate(representation.label_names):
        axes[0].plot(s, result["labels"][:, index], label=name)
    axes[0].set_title(representation.title)
    axes[0].set_ylabel("label value")
    axes[0].legend(ncol=min(len(representation.label_names), 3), fontsize="small")
    axes[1].plot(s, result["error"],
                 label=f"angular error (max {result['error'].max():.2f} deg)")
    axes[1].set_title("Decoded rotation error")
    axes[1].set_ylabel("angular error (deg)")
    axes[1].set_xlabel("s")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(Path(output_dir) / f"{representation.name}_results.png", dpi=150)
    plt.close(figure)

def plot_quaternion_norms(s, raw, canonical, output_dir):
    """Plot predicted quaternion norm for the raw-label and canonical-label models."""
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    for axis, result, label in (
        (axes[0], raw, "raw-label model"),
        (axes[1], canonical, r"canonical model ($w \geq 0$)"),
    ):
        norm = np.linalg.norm(result["raw"], axis=1)
        axis.plot(s, norm, label=f"{label} (min {norm.min():.3f})")
        axis.set_ylabel("predicted norm")
        axis.legend()
    axes[0].set_title("Predicted quaternion norm")
    axes[1].set_xlabel("s")
    figure.tight_layout()
    figure.savefig(Path(output_dir) / "quaternion_norms.png", dpi=150)
    plt.close(figure)


# ============================= YOUR CODE =============================

def rotations_from_s(s):
    """Construct the target rotations along the two-lap path.

    Fill in ``angles``: a shape ``(N, 3)`` array whose three columns are the
    intrinsic XYZ Euler angles of the robot's orientation, namely roll
    ``0.4 * sin(s)``, pitch ``0.3 * sin(2 * s)``, and yaw ``s``. Note that
    capitalization matters in the provided line below: ``Rot.from_euler``
    reads ``"XYZ"`` as intrinsic and ``"xyz"`` as extrinsic. ``np.stack``
    may be useful.

    Args:
        s: NumPy array with shape ``(N,)`` containing path-progress values.

    Returns:
        A SciPy ``Rotation`` object containing ``N`` rotations.
    """
    angles = np.stack([0.4 * np.sin(s), 0.3 * np.sin(2 * s), s], axis=1)
    if angles is None:
        raise NotImplementedError
    return Rot.from_euler("XYZ", angles)

def make_dataset(seed=0):
    """Construct the paired two-lap data, split, and dense sweep.

    Fill in ``s0``: the 500 base path values, drawn uniformly from
    ``[0, 2*pi)`` with ``rng.uniform``. The line below logs each of them on
    both laps, giving the 1000 paired samples the rest of the problem needs.

    Args:
        seed: Integer seed passed to ``np.random.default_rng``.

    Returns:
        A dictionary with the following entries:

        - ``s``: shape ``(1000,)`` path values, each base value and that
          value plus ``2*pi``.
        - ``X``: shape ``(1000, 2)`` positions ``[cos(s), sin(s)]``.
        - ``rot``: the ``Rotation`` object holding the 1000 orientations.
          Each representation encodes its own labels from this one object.
        - ``R``: shape ``(1000, 3, 3)`` rotation matrices.
        - ``train`` and ``eval``: shape ``(900,)`` and ``(100,)`` integer
          index arrays, e.g. ``X[train]`` are the training inputs.
        - ``s_sweep``: 4000 evenly spaced values in ``[0, 4*pi)``.
        - ``X_sweep``: shape ``(4000, 2)`` positions built from ``s_sweep``.
        - ``rot_sweep``: the ``Rotation`` object for the dense sweep.
        - ``R_sweep``: shape ``(4000, 3, 3)`` sweep rotation matrices.
    """
    rng = np.random.default_rng(seed)
    s0 = rng.uniform(0, 2 * np.pi, 500)
    if s0 is None:
        raise NotImplementedError
    s = np.concatenate([s0, s0 + 2 * np.pi])
    rotations = rotations_from_s(s)
    indices = rng.permutation(len(s))
    s_sweep = np.linspace(0, TWO_LAPS, 4000, endpoint=False)
    sweep_rotations = rotations_from_s(s_sweep)
    return {
        "s": s,
        "X": np.stack([np.cos(s), np.sin(s)], axis=1),
        "rot": rotations,
        "R": rotations.as_matrix(),
        "train": indices[:900],
        "eval": indices[900:],
        "s_sweep": s_sweep,
        "X_sweep": np.stack([np.cos(s_sweep), np.sin(s_sweep)], axis=1),
        "rot_sweep": sweep_rotations,
        "R_sweep": sweep_rotations.as_matrix(),
    }

def decode_matrix(predictions):
    """Decode raw flattened-matrix predictions.

    Args:
        predictions: NumPy array with shape ``(N, 9)``.

    Returns:
        NumPy array with shape ``(N, 3, 3)`` containing valid rotations.
        Reshape each row and project it onto SO(3) with ``renormalize_SO3``.
    """
    rotations = np.reshape(predictions, (-1, 3, 3))
    return renormalize_SO3(rotations)

def decode_euler(predictions):
    """Decode intrinsic XYZ Euler-angle predictions.

    Args:
        predictions: NumPy array with shape ``(N, 3)``.

    Returns:
        NumPy array with shape ``(N, 3, 3)``. Capitalization matters: use
        ``Rot.from_euler("XYZ", predictions)``.
    """
    return Rot.from_euler("XYZ", predictions).as_matrix()

def encode_exponential(rotations):
    """Encode rotation matrices as exponential coordinates.

    Args:
        rotations: NumPy array with shape ``(N, 3, 3)``.

    Returns:
        NumPy array with shape ``(N, 3)``. For each matrix ``R``, compute
        ``SO3 -> R3`` conversion (use kin_func_skeleton), which is ``log(R)^vee``.
    """
    return np.stack([so3_to_R3(SO3_to_so3(r)) for r in rotations])

def decode_exponential(vectors):
    """Decode exponential coordinates into rotation matrices.

    Args:
        vectors: NumPy array with shape ``(N, 3)``.

    Returns:
        NumPy array with shape ``(N, 3, 3)``. For each vector ``v``, compute
        ``R3 -> SO3`` (use kin_func_skeleton).
    """
    return np.stack([so3_to_SO3(R3_to_so3(v)) for v in vectors])

def normalize_quaternions(quaternions):
    """Normalize predicted quaternions before decoding them.

    Args:
        quaternions: NumPy array with shape ``(N, 4)`` in SciPy
            ``[x, y, z, w]`` order.

    Returns:
        NumPy array with shape ``(N, 4)`` whose rows have unit norm.
        Ensure we don't divide by zero!
    """
    result = []
    for i in range(len(quaternions)):
        q = quaternions[i]
        if np.linalg.norm(q) == 0:
            result.append([0, 0, 0, 0])
        else:
            mag = np.linalg.norm(q)
            result.append(q / mag)
    return np.array(result)

def canonicalize_quaternions(quaternions):
    """Apply the ``w >= 0`` quaternion-label convention.

    Args:
        quaternions: NumPy array with shape ``(N, 4)`` in SciPy
            ``[x, y, z, w]`` order.

    Returns:
        A new NumPy array with shape ``(N, 4)``. Multiply rows whose last
        component is negative by ``-1``. Do not modify the input array.
    """
    return np.where(quaternions[:, 3:4] < 0, -quaternions, quaternions)

def decode_quaternions(predictions):
    """Decode raw quaternion predictions.

    Args:
        predictions: NumPy array with shape ``(N, 4)`` in SciPy
            ``[x, y, z, w]`` order.

    Returns:
        NumPy array with shape ``(N, 3, 3)``. Normalize the predictions with
        ``normalize_quaternions`` before passing them to ``Rot.from_quat``.
    """
    normalized = normalize_quaternions(predictions)
    return Rot.from_quat(normalized).as_matrix()

def encode_6d(rotations):
    """Encode rotation matrices using the 6D representation.

    Args:
        rotations: NumPy array with shape ``(N, 3, 3)``.

    Returns:
        NumPy array with shape ``(N, 6)``. Each row contains the first
        column of its rotation followed by the second column.
    """
     

def decode_6d(vectors):
    """Decode 6D predictions using the stated Gram--Schmidt procedure.

    Args:
        vectors: NumPy array with shape ``(N, 6)``. The first and last three
            values are the two predicted column vectors.

    Returns:
        NumPy array with shape ``(N, 3, 3)``. Normalize the first vector,
        remove its component from the second and normalize the result, then
        use their cross product as the third rotation-matrix column.
    """
    # TODO YOUR CODE HERE
    raise NotImplementedError


# ============================= PROVIDED: EXPERIMENTS =============================
# The registry, training loop, metrics, and dense probe are complete below.
# Do not modify this section.
#
# ``impl`` is the module holding the encode/decode implementations. It defaults
# to this file; the staff reference passes itself instead, so that it can reuse
# everything here without copying it.

def representations(impl):
    """Build the five representations from an implementation module."""
    return (
        Representation(
            "matrix",
            lambda rotations: rotations.as_matrix().reshape(-1, 9),
            impl.decode_matrix,
            tuple(rf"$R_{{{i}{j}}}$" for i in range(1, 4) for j in range(1, 4)),
            "9D labels: flattened rotation matrix",
        ),
        Representation(
            "euler",
            lambda rotations: rotations.as_euler("XYZ"),
            impl.decode_euler,
            ("roll", "pitch", "yaw"),
            "Intrinsic XYZ Euler labels",
        ),
        Representation(
            "exponential",
            on_matrices(impl.encode_exponential),
            impl.decode_exponential,
            (r"$\omega_x$", r"$\omega_y$", r"$\omega_z$"),
            r"Exponential-coordinate labels $\log(R)^\vee$",
        ),
        Representation(
            "quaternion",
            lambda rotations: rotations.as_quat(),
            impl.decode_quaternions,
            ("$q_x$", "$q_y$", "$q_z$", "$q_w$"),
            "Quaternion labels",
        ),
        Representation(
            CANONICAL,
            lambda rotations: impl.canonicalize_quaternions(rotations.as_quat()),
            impl.decode_quaternions,
            ("$q_x$", "$q_y$", "$q_z$", "$q_w$"),
            r"Quaternion labels, canonicalized to $w \geq 0$",
        ),
        Representation(
            "sixd",
            on_matrices(impl.encode_6d),
            impl.decode_6d,
            (r"$r_{1x}$", r"$r_{1y}$", r"$r_{1z}$",
             r"$r_{2x}$", r"$r_{2y}$", r"$r_{2z}$"),
            "6D labels: first two rotation-matrix columns",
        ),
    )

def run_experiment(representation, data):
    """Train one representation and measure its error over the dense sweep.

    Args:
        representation: The ``Representation`` to train.
        data: The dictionary returned by ``make_dataset``.

    Returns:
        A dictionary with the ``representation`` itself, the trained
        ``model``, two scalar evaluation scores ``mse`` and ``msae``, the
        ``(4000, d)`` ``raw`` sweep outputs, the ``(4000, d)`` ground-truth
        sweep ``labels``, and the ``(4000,)`` sweep ``error``.

        The two scores measure different things. ``mse`` is the mean squared
        error the network actually minimizes, computed on the raw
        representation coordinates, so its units differ from row to row and
        it says nothing directly about rotations. ``msae`` decodes the
        predictions to rotations first and is the mean squared angular error
        in square radians, so it is comparable across representations.
    """
    print("Running experiment: " + representation.name)
    train, evaluate = data["train"], data["eval"]
    labels = representation.encode(data["rot"])
    model = fit(data["X"][train], labels[train])
    raw_eval = predict(model, data["X"][evaluate])
    mse = np.mean((raw_eval - labels[evaluate]) ** 2)
    decoded_eval = representation.decode(raw_eval)
    msae = np.mean(np.radians(rotation_errors(decoded_eval, data["R"][evaluate])) ** 2)
    raw = predict(model, data["X_sweep"])
    decoded = representation.decode(raw)
    return {
        "representation": representation,
        "model": model,
        "mse": mse,
        "msae": msae,
        "raw": raw,
        "labels": representation.encode(data["rot_sweep"]),
        "error": rotation_errors(decoded, data["R_sweep"]),
    }

def seam_probe(result, data, impl, half_width=0.05, count=20000):
    """Resample densely around the seam, where the predicted norm is smallest.

    The sweep grid is far too coarse to resolve the sign-convention seam, so
    the reported sweep max understates the true error. This zooms in on it.

    Returns:
        A dictionary with the ``(count,)`` probe path values ``s`` and the
        ``(count,)`` probe ``error``.
    """
    norm = np.linalg.norm(result["raw"], axis=1)
    seam = data["s_sweep"][np.argmin(norm)]
    probe_s = (seam + np.linspace(-half_width, half_width, count)) % TWO_LAPS
    probe_X = np.stack([np.cos(probe_s), np.sin(probe_s)], axis=1)
    decoded = result["representation"].decode(predict(result["model"], probe_X))
    truth = impl.rotations_from_s(probe_s).as_matrix()
    return {"s": probe_s, "error": rotation_errors(decoded, truth)}


# ============================= PROVIDED: SCRIPT =============================

CANONICAL = "quaternion_canonical"  # the w >= 0 variant, reported separately

def print_results(results):
    """Print the eval error and sweep error statistics as a table."""
    print(f"{'representation':22s} {'eval MSE':>10s} {'eval MSAE (rad^2)':>18s} "
          f"{'sweep mean':>12s} {'sweep max':>11s}")
    for name, result in results.items():
        print(f"{name:22s} {result['mse']:10.2e} {result['msae']:18.2e} "
              f"{result['error'].mean():11.2f}\u00b0 {result['error'].max():10.2f}\u00b0")

def main(output_dir=Path("."), impl=None):
    """Run every representation and write plots."""
    impl = impl or sys.modules[__name__]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = impl.make_dataset(seed=0)
    results = {rep.name: run_experiment(rep, data)
               for rep in representations(impl)}
    probe = seam_probe(results[CANONICAL], data, impl)

    # The five representations form the summary table; the canonical-label
    # quaternion variant and the seam probe belong to the quaternion question
    # rather than to the comparison across representations.
    print_results({name: result for name, result in results.items()
                   if name != CANONICAL})
    print("\nAdditional quaternion experiments")
    print_results({CANONICAL: results[CANONICAL]})
    print(f"dense probe around the seam: max = "
          f"{probe['error'].max():.2f}\u00b0 over "
          f"{len(probe['error'])} samples spanning "
          f"{probe['s'].max() - probe['s'].min():.2f} rad")

    for result in results.values():
        plot_result(data["s_sweep"], result, output_dir)
    plot_quaternion_norms(data["s_sweep"], results["quaternion"],
                          results[CANONICAL], output_dir)
    return results, probe

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    arguments = parser.parse_args()
    main(arguments.output_dir)
