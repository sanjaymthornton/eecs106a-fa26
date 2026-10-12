"""Provided report tool: python analyze.py --method naive|ransac|both."""
import argparse
import csv
from pathlib import Path
import numpy as np
from support import load_cloud
from plane_fitting import fit_plane_ls, plane_distances, fit_plane_ransac


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=('naive', 'ransac', 'both'), default='both')
    args = parser.parse_args()
    points, _ = load_cloud(Path(__file__).with_name('cloud.npy'))
    rows = []
    try:
        naive = fit_plane_ls(points) if args.method in ('naive', 'both') else None
        for threshold in (0.002, 0.010, 0.050):
            fits = []
            if naive is not None:
                fits.append(('naive', *naive, plane_distances(points, *naive) <= threshold))
            if args.method in ('ransac', 'both'):
                n, d, mask, _ = fit_plane_ransac(points, threshold, 500, 106)
                fits.append(('ransac', n, d, mask))
            for method, n, d, mask in fits:
                residuals = plane_distances(points, n, d)
                rows.append(dict(method=method, threshold_m=threshold, inliers=int(mask.sum()),
                    median_distance_mm=float(np.median(residuals)*1000),
                    nx=float(n[0]), ny=float(n[1]), nz=float(n[2]), d_m=float(d)))
    except NotImplementedError as exc:
        parser.exit(1, f'Complete plane_fitting.py first: {exc}\n')
    output = Path(__file__).parent / 'results'
    output.mkdir(exist_ok=True)
    with (output / 'fit_metrics.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('method   threshold(m)  inliers   median distance(mm)')
    for r in rows:
        print(f"{r['method']:8s} {r['threshold_m']:10.3f} {r['inliers']:8d} {r['median_distance_mm']:20.3f}")
    print(f'Saved {output / "fit_metrics.csv"}')


if __name__ == '__main__':
    main()
