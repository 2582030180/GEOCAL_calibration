import numpy as np
from scipy.optimize import least_squares


# 建立内参矩阵，计算重投影误差，便于优化矩阵参数
def reprojection_error(params, R, pix_coor, real_coor):
    fx, fy, uc, vc = params  # 不考虑畸变
    num_points = len(pix_coor)
    errors = []

    for i in range(num_points):
        X, Y = real_coor[i]
        x, y = pix_coor[i]

        # 构造内参矩阵
        A = np.array([
            [fx, 0, uc],
            [0, fy, vc],
            [0, 0, 1]
        ])

        # 变为齐次坐标，还是假设在Z=1平面上？
        camera_point = np.array([X, Y, 1])

        # 用内参矩阵进行变换
        image_point = np.dot(A, np.dot(R, camera_point))
        # 归一化
        image_point = image_point / image_point[-1]

        # 计算两个方向上的误差
        errors.append(x - image_point[0])
        errors.append(y - image_point[1])

    # 打印当前参数和误差
    # current_error = np.linalg.norm(errors)
    # print(f"error: {current_error:.3f}")

    return errors


def get_intrinsic(R, pix_coords, camera_coords):
    pixel_coords = pix_coords
    camera_coords = camera_coords

    theta = 0.9
    theta = np.radians(theta)
    camera_coords = np.tan(camera_coords * theta)

    initial_param = np.array([1000, 1000, 1023, 790])

    result = least_squares(reprojection_error, initial_param, args=(R, pixel_coords, camera_coords),
                           loss='cauchy')  # loss = 'huber'

    fx, fy, uc, vc = result.x

    A = np.array([
        [fx, 0, uc],
        [0, fy, vc],
        [0, 0, 1]
    ])

    np.set_printoptions(precision=3, suppress=True)

    # print(A)
    return A
