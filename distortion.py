import numpy as np
from scipy.optimize import least_squares


def reprojection_error(params, intrinsic_param, R, pic_coor, world_coor):
    k1, k2, k3 = params
    num_points = len(pic_coor)
    errors = []

    for i in range(num_points):
        X, Y = world_coor[i]
        x, y = pic_coor[i]

        camera_coor = np.dot(R, np.array([X, Y, 1]))
        camera_coor = camera_coor / camera_coor[-1]
        Xc, Yc = camera_coor[0], camera_coor[1]

        fx = intrinsic_param[0, 0]
        fy = intrinsic_param[1, 1]
        cx = intrinsic_param[0, 2]
        cy = intrinsic_param[1, 2]

        Z = 1
        u_ideal = fx * Xc / Z + cx
        v_ideal = fy * Yc / Z + cy

        r = np.sqrt((Xc / Z) ** 2 + (Yc / Z) ** 2)

        u_distorted = u_ideal * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)
        v_distorted = v_ideal * (1 + k1 * r ** 2 + k2 * r ** 4 + k3 * r ** 6)

        errors.append(x - u_distorted)
        errors.append(y - v_distorted)

    # current_error = np.linalg.norm(errors)
    # print(f"error: {current_error:.3f}")

    return errors


def get_distortion(A, R, pix_coords, camera_coords):
    pixel_coords = pix_coords
    camera_coords = camera_coords

    theta = 0.9
    theta = np.radians(theta)
    camera_coords = np.tan(camera_coords * theta)

    initial_param = np.array([0, 0, 0])
    intrinsic_param = A

    result = least_squares(reprojection_error, initial_param, args=(intrinsic_param, R, pixel_coords, camera_coords),
                           loss='cauchy')  # loss = 'huber'
    k1, k2, k3 = result.x

    np.set_printoptions(precision=3, suppress=True)

    return k1, k2, k3
