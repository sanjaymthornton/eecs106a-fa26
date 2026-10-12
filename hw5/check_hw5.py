"""Every scored HW5 case. Run python hw5/check_hw5.py from the repository root."""
from contextlib import contextmanager
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyransac3d


FUNCTION_NAMES = ('fit_plane_ls', 'plane_distances', 'fit_plane_ransac')


def plane_points(normal, offset, seed=5, count=24, noise=0.0):
    """Construct finite test inputs near a specified plane."""
    normal = np.asarray(normal, dtype=float)
    normal = normal / np.linalg.norm(normal)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(normal, axis)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    rng = np.random.default_rng(seed)
    xy = rng.uniform(-1, 1, (count, 2))
    points = -offset * normal + xy[:, :1] * u + xy[:, 1:] * v
    return points + rng.normal(0, noise, (count, 1)) * normal


def ls_cases():
    """Yield every scored least-squares case."""
    yield 'horizontal plane with offset', {'points': plane_points([0, 0, 1], -2)}
    yield 'vertical plane', {'points': plane_points([1, 0, 0], 0.7)}
    yield 'tilted plane', {'points': plane_points([1, -2, 3], -0.4)}
    yield 'exactly three points', {'points': np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2]])}
    yield 'noisy orthogonal fit', {'points': plane_points([2, 1, -3], 0.6, noise=0.025)}
    points = plane_points([1, 2, 3], -0.2, noise=0.015)
    yield 'translated noisy cloud', {'points': points + [10, -7, 4]}
    yield 'reordered points', {'points': points[::-1].copy()}
    yield 'duplicate points', {'points': np.repeat(points, 2, axis=0)}
    yield 'float32 input', {'points': points.astype(np.float32)}
    yield 'nonuniform planar sampling', {'points': np.array([[0, 0, 4], [1, 0, 4], [0, 2, 4], [9, 3, 4], [-2, 1, 4]])}


def distance_cases():
    """Yield every scored distance case."""
    points = np.array([[0, 0, 2], [1, 0, 2.25], [0, 1, 1.5]])
    for name, normal, offset in (
        ('on and both sides', [0, 0, 1], -2),
        ('scaled plane', [0, 0, 4], -8),
        ('sign-reversed plane', [0, 0, -3], 6),
        ('vertical plane', [1, 0, 0], -0.5),
        ('tilted plane', [1, -2, 3], -1.5),
        ('small nonzero normal', [0, 0, 1e-8], -2e-8),
    ):
        yield name, {'points': points.copy(), 'normal': normal, 'offset': offset}
    yield 'single point', {'points': np.array([[3, 2, 1]]), 'normal': [2, 1, -1], 'offset': 0.3}
    yield 'empty batch', {'points': np.empty((0, 3)), 'normal': [0, 0, 1], 'offset': 2}
    yield 'integer points', {'points': np.array([[0, 0, 0], [1, 2, 3]]), 'normal': [1, 1, 1], 'offset': -2}
    yield 'float32 points', {'points': points.astype(np.float32), 'normal': [0.5, -0.25, 1], 'offset': 0.1}


def ransac_cases():
    """Yield real package fits and controlled consensus-refit cases."""
    for name, normal, offset, noise, threshold, seed in (
        ('horizontal plane and outliers', [0, 0, 1], -0.4, 0.0, 0.001, 106),
        ('vertical plane and outliers', [1, 0, 0], 0.2, 0.0, 0.002, 9),
        ('tilted noisy plane and outliers', [1, -2, 3], -0.3, 0.002, 0.012, 17),
        ('custom tight threshold', [2, 1, 3], 0.5, 0.0002, 0.0015, 23),
        ('all points planar', [1, 2, -1], 0.1, 0.0, 0.001, 5),
    ):
        points = plane_points(normal, offset, noise=noise)
        if name != 'all points planar':
            points = np.vstack([points, [[4, 3, 5], [-3, 4, -5], [5, -4, 3]]])
        yield name, {'points': points, 'threshold': threshold, 'iterations': 80, 'seed': seed}, None
    yield 'three points and one iteration', {
        'points': np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2]]),
        'threshold': 0.001, 'iterations': 1, 'seed': 106,
    }, None
    for name, points, indices, threshold in (
        ('refit rather than retain package equation',
         [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 2]], [0, 1, 2, 3], 0.05),
        ('original indices differ from refined mask',
         [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0.2, 0.3, 1.01], [0, 0, 2]], [0, 1, 2, 3], 0.05),
        ('inclusive threshold boundary',
         [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1], [0, 0, 1.125], [0, 0, 0.875], [0, 0, 1.25]], [0, 1, 2, 3], 0.125),
        ('noncontiguous original indices',
         [[0, 0, 1], [0, 0, 3], [1, 0, 1], [0, 1, 1], [0, 0, -2], [1, 1, 1]], [0, 2, 3, 5], 0.02),
    ):
        yield name, {'points': np.array(points), 'threshold': threshold, 'iterations': 7, 'seed': 31}, {
            'equation': [0, 0, 1, -0.8], 'indices': indices,
        }


def cases(function_name):
    """Return public inputs and any controlled package response."""
    if function_name == 'fit_plane_ransac':
        yield from ransac_cases()
    else:
        source = ls_cases if function_name == 'fit_plane_ls' else distance_cases
        for name, arguments in source():
            yield name, arguments, None


@contextmanager
def package_response(response):
    """Supply a public consensus set and record the package-call arguments."""
    calls = []
    original_fit = pyransac3d.Plane.fit

    def fit(plane, pts, thresh=0.05, maxIteration=1000, **kwargs):
        calls.append({'points': np.asarray(pts).copy(), 'threshold': thresh, 'iterations': maxIteration})
        if response is None:
            return original_fit(plane, pts, thresh=thresh, maxIteration=maxIteration, **kwargs)
        return response['equation'].copy(), response['indices'].copy()

    with patch.object(pyransac3d.Plane, 'fit', fit):
        yield calls


def output_matches(function_name, actual, expected):
    """Check shapes, finite values, and geometrically equivalent planes."""
    if function_name == 'plane_distances':
        distances = np.asarray(actual)
        want = np.asarray(expected)
        return (distances.shape == want.shape and np.isfinite(distances).all()
                and (distances >= 0).all() and np.allclose(distances, want, rtol=1e-7, atol=1e-9))
    if not isinstance(actual, (tuple, list)) or len(actual) != len(expected):
        return False
    normal, offset = np.asarray(actual[0]), np.asarray(actual[1])
    if normal.shape != (3,) or offset.shape != () or not np.isfinite(normal).all() or not np.isfinite(offset):
        return False
    if not np.isclose(np.linalg.norm(normal), 1, rtol=1e-7, atol=1e-7):
        return False
    want = np.asarray(expected[0])
    sign = 1 if normal @ want >= 0 else -1
    if not (np.allclose(sign * normal, want, rtol=1e-7, atol=1e-7)
            and np.isclose(sign * offset, expected[1], rtol=1e-7, atol=1e-7)):
        return False
    if function_name == 'fit_plane_ransac':
        mask, indices = np.asarray(actual[2]), np.asarray(actual[3])
        return (mask.shape == np.asarray(expected[2]).shape and mask.dtype == np.bool_
                and np.array_equal(mask, expected[2]) and indices.ndim == 1
                and indices.dtype.kind in 'iu' and np.array_equal(np.sort(indices), np.sort(expected[3])))
    return True


def evaluate(function_name, function, arguments, response, expected):
    """Run one case without replacing any student-assigned helper."""
    before = np.asarray(arguments['points']).copy()
    with package_response(response) as calls:
        actual = function(**arguments)
    passed = output_matches(function_name, actual, expected)
    if function_name == 'fit_plane_ransac':
        passed = passed and np.array_equal(arguments['points'], before)
        passed = passed and len(calls) == 1 and np.array_equal(calls[0]['points'], before)
        passed = passed and calls[0]['threshold'] == arguments['threshold'] and calls[0]['iterations'] == arguments['iterations']
    return passed, actual, calls


def run_checks():
    """Run all scored cases using the student's own functions."""
    import plane_fitting

    expected_results = json.loads(Path(__file__).with_name('expected_results.json').read_text())
    all_passed = True
    for function_name in FUNCTION_NAMES:
        for index, (name, arguments, response) in enumerate(cases(function_name)):
            try:
                passed, _, _ = evaluate(function_name, getattr(plane_fitting, function_name),
                                        arguments, response, expected_results[function_name][index])
            except Exception as error:
                passed = False
                print(f'{type(error).__name__}: {error}')
            all_passed &= passed
            print(f"[{'PASS' if passed else 'FAIL'}] {function_name}: {name}")
    return all_passed


if __name__ == '__main__':
    raise SystemExit(0 if run_checks() else 1)
