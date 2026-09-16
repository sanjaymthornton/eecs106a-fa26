import numpy as np


def box_pose_in_world(t, box_speed, box_distance):
    '''
    Problem (b)
        Returns the 4x4 rigid body pose g_01 at time t. The box begins at
        (0, box_distance, 0), its frame is aligned with the world frame, and
        it moves in the positive x direction at a constant speed.

        Args:
            t: time at which the configuration is computed, in seconds.
            box_speed: linear speed of the box, in meters per second.
            box_distance: distance across the belt from the post to the box,
                in meters.
        Returns:
            4x4 rigid pose of frame {1} as seen from frame {0} as a numpy
            array.
    '''
    g = np.eye(4)
    g[0, 3] = box_speed * t
    g[1, 3] = box_distance
    g[2, 3] = 0
    return g


def box_twist_in_world(box_speed):
    '''
    Problem (c)
        Returns the 6x1 spatial twist xi_1 describing the motion of the box.
        The twist is written in frame {0} in the order
        [v_x, v_y, v_z, omega_x, omega_y, omega_z].

        Args:
            box_speed: linear speed of the box, in meters per second.
        Returns:
            6x1 twist describing the motion of frame {1} relative to frame
            {0} as a numpy array.
    '''
    xi = np.array([
        box_speed,
        0,
        0,
        0,
        0,
        0
    ])
    return xi


def camera_pose_in_world(t, camera_angular_speed, camera_height):
    '''
    Problem (d)
        Returns the 4x4 rigid body pose g_02 at time t. The camera is located
        at (0, 0, camera_height). At time zero, its x-axis points in the world
        x direction, its y-axis points down, and its z-axis points in the
        world y direction. It pans clockwise about the world z-axis at a
        constant angular speed when viewed from above.

        Args:
            t: time at which the configuration is computed, in seconds.
            camera_angular_speed: positive magnitude of the camera's clockwise
                angular speed, in radians per second.
            camera_height: height of the camera above the belt, in meters.
        Returns:
            4x4 rigid pose of frame {2} as seen from frame {0} as a numpy
            array.

        Functions you might find useful:
            numpy.sin
            numpy.cos
            numpy.matmul
            numpy.eye
    '''
    g = np.eye(4)
    theta = camera_angular_speed * t

    # Clockwise rotation: the bottom sine is negative in the rotation
    rot_cos = np.cos(theta)
    rot_sin = np.sin(theta)
    rot = np.array([[rot_cos, rot_sin, 0],
                  [-1 * rot_sin, rot_cos, 0],
                  [0, 0, 1]])
    init_positions = np.array([[1, 0, 0],
                               [0, 0, 1],
                               [0, -1, 0]])
    final_R = np.matmul(rot, init_positions)
    g[0:3, 0:3] = final_R
    g[0:3, 3] = np.array([0, 0, camera_height])
    return g


def camera_twist_in_world(camera_angular_speed):
    '''
    Problem (e)
        Returns the 6x1 spatial twist xi_2 describing the motion of the
        camera. The twist is written in frame {0} in the order
        [v_x, v_y, v_z, omega_x, omega_y, omega_z]. A positive value for
        camera_angular_speed describes clockwise motion when viewed from
        above.

        Args:
            camera_angular_speed: positive magnitude of the camera's clockwise
                angular speed, in radians per second.
        Returns:
            6x1 twist describing the motion of frame {2} relative to frame
            {0} as a numpy array.
    '''
    xi = np.array([
        0,
        0,
        0,
        0,
        0,
        -camera_angular_speed,
    ])
    return xi


def box_pose_in_camera(
    t,
    box_speed,
    camera_angular_speed,
    box_distance,
    camera_height,
):
    '''
    Problem (f)
        Returns the 4x4 rigid body pose g_21 at time t. This pose describes
        the box frame {1} as seen from the camera frame {2}.

        Args:
            t: time at which the configuration is computed, in seconds.
            box_speed: linear speed of the box, in meters per second.
            camera_angular_speed: positive magnitude of the camera's clockwise
                angular speed, in radians per second.
            box_distance: distance across the belt from the post to the box,
                in meters.
            camera_height: height of the camera above the belt, in meters.
        Returns:
            4x4 rigid pose of frame {1} as seen from frame {2} as a numpy
            array.

        Functions you might find useful:
            numpy.matmul
            numpy.linalg.inv

        Note: Feel free to use one or more of the other functions you have
        implemented in this file.
    '''
    g = np.eye(4)
    box_pose = box_pose_in_world(t, box_speed, box_distance)
    camera_pose = camera_pose_in_world(t, camera_angular_speed, camera_height)
    g = np.matmul(np.linalg.inv(camera_pose), box_pose)
    return g
