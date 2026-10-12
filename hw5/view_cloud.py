"""Provided Viser viewer. Run python view_cloud.py, then open localhost:8080."""
import argparse
import io
from pathlib import Path
import threading
import time
import numpy as np
import viser
import plane_fitting as student
from support import load_cloud, load_colors, normalize_plane, plane_patch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    folder = Path(__file__).parent
    points, removed = load_cloud(folder / 'cloud.npy')
    colors = load_colors(folder / 'cloud_colors.npy', np.load(folder / 'cloud.npy', allow_pickle=False))
    print(f'{len(points):,} cropped points; {removed} nonfinite rows excluded.')
    server = viser.ViserServer(host='127.0.0.1', port=args.port, label='HW5 · Plane fitting')
    server.scene.world_axes.visible = False
    server.scene.set_up_direction('-y')
    lock = threading.RLock()
    state = {'naive': None, 'ransac': None, 'errors': {}, 'threshold': 0.010}
    view = server.gui.add_dropdown('View', ('Point cloud', 'Naive plane fit', 'RANSAC plane fit', 'Compare both planes'), initial_value='Point cloud')
    equations = server.gui.add_markdown('', visible=False)
    show_plane = server.gui.add_checkbox('Show plane', True)
    highlight = server.gui.add_checkbox('Color inliers / outliers', False)
    opacity = server.gui.add_slider('Plane opacity', 0.0, 0.8, 0.05, 0.25)
    point_size = server.gui.add_slider('Point size', 0.0005, 0.006, 0.0005, 0.0015)
    threshold = server.gui.add_number('Threshold (m)', 0.010, min=0.0001, max=1.0, step=0.001)
    iterations = server.gui.add_number('Iterations', 500, min=1, max=10000, step=100)
    seed = server.gui.add_number('Seed', 106, min=0, step=1)
    refit = server.gui.add_button('Refit')
    top = server.gui.add_button('View top down')
    side = server.gui.add_button('View side view')
    export = server.gui.add_button('Download fits (.npz)')
    message = server.gui.add_markdown('', visible=False)
    cloud = server.scene.add_point_cloud('/cloud', points.astype(np.float32), colors=colors,
        point_size=point_size.value, point_shape='circle', precision='float32')
    center = (points.min(0) + points.max(0)) / 2
    span = max(float(np.linalg.norm(np.ptp(points, axis=0))), 0.1)
    meshes, borders = {}, {}
    palette = {'naive': (245, 159, 50), 'ransac': (75, 140, 255)}

    def outline(vertices):
        return vertices[np.array([[0, 1], [1, 2], [2, 3], [3, 0]])]

    def camera(client, from_side=False):
        # Fixed capture axes, independent of student fitting correctness.
        up = np.array([0.0, 0.0, -1.0]) if from_side else np.array([0.0, -1.0, 0.0])
        if from_side:
            # Near table level: median depth only frames the cloud, not a fit.
            target = center.copy()
            target[2] = np.median(points[:, 2])
            position = target + span * np.array([1.25, 0.0, -0.025])
        else:
            target = center
            position = target + span * np.array([0.0, 0.0, -1.25])
        with server.atomic():
            client.camera.position = tuple(position)
            client.camera.look_at = tuple(target)
            client.camera.up_direction = tuple(up)

    @server.on_client_connect
    def connected(client):
        camera(client)

    def refresh(_=None):
        with lock, server.atomic():
            selected = {'Point cloud': (), 'Naive plane fit': ('naive',),
                        'RANSAC plane fit': ('ransac',), 'Compare both planes': ('naive', 'ransac')}[view.value]
            text, missing = [], []
            for key in ('naive', 'ransac'):
                fit = state[key]
                visible = key in selected and fit is not None and show_plane.value
                if key in meshes:
                    meshes[key].visible = borders[key].visible = visible
                    meshes[key].opacity = opacity.value
                if key not in selected:
                    continue
                if fit is None:
                    missing.append(f"{key.title()}: {state['errors'].get(key, 'not available')}")
                    continue
                n, d = fit[:2]
                label = 'Naive (orange)' if key == 'naive' else 'RANSAC (blue)'
                text.append(f'**{label}**\n\n{n[0]:.6f} x {n[1]:+.6f} y  \n{n[2]:+.6f} z {d:+.6f} = 0')
            equations.content, equations.visible = '\n\n'.join(text), bool(text)
            display_colors = colors.copy()
            if highlight.value and selected:
                chosen = 'ransac' if 'ransac' in selected and state['ransac'] is not None else selected[0]
                fit = state[chosen]
                if fit is not None:
                    try:
                        if chosen == 'ransac':
                            mask = fit[2]
                        else:
                            mask = np.asarray(student.plane_distances(points, *fit[:2])) <= state['threshold']
                        if mask.shape != (len(points),):
                            raise ValueError('Expected one distance or mask entry per point.')
                        display_colors[mask] = (50, 190, 165)
                        display_colors[~mask] = (250, 105, 85)
                    except Exception as exc:
                        missing.append(f'Inlier colors: {exc}')
            cloud.colors, cloud.point_size = display_colors, point_size.value
            message.content, message.visible = '\n\n'.join(missing), bool(missing)

    def compute(_=None):
        with lock:
            refit.disabled = True
            try:
                state['threshold'] = float(threshold.value)
                for key in ('naive', 'ransac'):
                    try:
                        result = (student.fit_plane_ls(points) if key == 'naive' else
                                  student.fit_plane_ransac(points, state['threshold'], int(iterations.value), int(seed.value)))
                        n, d = normalize_plane(*result[:2])
                        if key == 'ransac':
                            mask = np.asarray(result[2])
                            if mask.shape != (len(points),) or mask.dtype != np.bool_:
                                raise ValueError('RANSAC mask must be an N-element boolean array.')
                            state[key] = (n, d, mask, np.asarray(result[3], dtype=int))
                        else:
                            state[key] = (n, d)
                        vertices, faces = plane_patch(points, n, d)
                        if key in meshes:
                            meshes[key].vertices = vertices
                            borders[key].points = outline(vertices)
                        else:
                            meshes[key] = server.scene.add_mesh_simple('/' + key, vertices, faces,
                                color=palette[key], opacity=opacity.value, side='double', visible=False)
                            borders[key] = server.scene.add_line_segments('/' + key + '_border', outline(vertices),
                                colors=palette[key], thickness=2.0, thickness_units='screen', visible=False)
                        state['errors'].pop(key, None)
                    except Exception as exc:
                        state[key] = None
                        state['errors'][key] = str(exc) or type(exc).__name__
                refresh()
            finally:
                refit.disabled = False

    for control in (view, show_plane, highlight, opacity, point_size):
        control.on_update(refresh)
    refit.on_click(compute)

    @top.on_click
    def top_view(event):
        if event.client is not None:
            camera(event.client)

    @side.on_click
    def side_view(event):
        if event.client is not None:
            camera(event.client, from_side=True)

    @export.on_click
    def download(event):
        with lock:
            data = {'points': points, 'colors': colors, 'threshold': state['threshold']}
            for key in ('naive', 'ransac'):
                if state[key] is not None:
                    data[key + '_plane'] = np.r_[state[key][0], state[key][1]]
            if state['ransac'] is not None:
                data['inliers'] = state['ransac'][2]
            stream = io.BytesIO()
            np.savez(stream, **data)
        if event.client is not None:
            event.client.send_file_download('plane_fits.npz', stream.getvalue())

    compute()
    print('Edit plane_fitting.py, then restart this process to load code changes.')
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        server.stop()


if __name__ == '__main__':
    main()
