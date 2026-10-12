"""Provided I/O and visualization helpers. No fitting algorithms."""
from pathlib import Path
import numpy as np

def load_cloud(path):
    """Load numeric Nx3 XYZ data without executing pickled objects."""
    points = np.load(Path(path), allow_pickle=False)
    if points.ndim != 2 or points.shape[1] != 3 or points.dtype.kind not in 'fiu':
        raise ValueError('Expected a numeric N x 3 array of XYZ coordinates.')
    points = np.asarray(points, dtype=np.float64)
    valid = np.isfinite(points).all(axis=1)
    removed = int((~valid).sum())
    points = points[valid]
    if len(points) < 3:
        raise ValueError('Need at least three finite points.')
    return points, removed

def normalize_plane(normal, offset):
    length = np.linalg.norm(normal)
    if not np.isfinite(length) or length < 1e-12:
        raise ValueError('Invalid plane normal.')
    normal = np.asarray(normal, dtype=float) / length
    offset = float(offset) / length
    # Consistent sign for comparisons, with no assumption about gravity.
    if normal[np.argmax(np.abs(normal))] < 0:
        normal, offset = -normal, -offset
    return normal, offset

def plane_patch(points, normal, offset):
    """A bounded rectangle on a plane; safe for vertical planes too."""
    normal, offset = normalize_plane(normal, offset)
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    u = np.cross(normal, axis)
    u /= np.linalg.norm(u)
    v = np.cross(normal, u)
    center = points.mean(axis=0)
    center = center - (center @ normal + offset) * normal
    uv = (points - center) @ np.stack([u, v], axis=1)
    lo, hi = uv.min(axis=0), uv.max(axis=0)
    pad = (hi - lo) * 0.015
    lo, hi = lo - pad, hi + pad
    corners = np.array([[lo[0], lo[1]], [hi[0], lo[1]],
                        [hi[0], hi[1]], [lo[0], hi[1]]])
    vertices = center + corners[:, :1] * u + corners[:, 1:] * v
    return vertices.astype(np.float32), np.array([[0, 1, 2], [0, 2, 3]], dtype=np.uint32)

def load_colors(path, original_points):
    """Load per-point RGB, retaining exactly the finite XYZ rows used by load_cloud."""
    colors = np.load(Path(path), allow_pickle=False)
    if colors.shape != original_points.shape or colors.dtype.kind not in 'fiu':
        raise ValueError('Colors must be numeric N x 3, with one RGB row per original XYZ row.')
    if not np.isfinite(colors).all() or colors.min() < 0 or colors.max() > 255:
        raise ValueError('RGB colors must be finite and in [0, 255].')
    if colors.dtype.kind == 'f' and colors.max() <= 1:
        colors = colors * 255
    colors = np.rint(colors).astype(np.uint8)
    return colors[np.isfinite(original_points).all(axis=1)]

