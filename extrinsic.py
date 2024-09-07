import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.linalg import expm, logm, norm
from scipy.optimize import least_squares


# 向量到反对称矩阵
# from vector to skew-symmetric matrix
def vector_to_matrix(v):
    new_matrix = np.array([[0, -v[2], v[1]],
                           [v[2], 0, -v[0]],
                           [-v[1], v[0], 0]])
    return new_matrix


# 李代数到李群，旋转向量到旋转矩阵
# from so(3) to SO(3)
def lie_algebra_to_group(omega):
    omega = np.asarray(omega)

    theta = norm(omega)
    if theta < 1e-10:
        return np.eye(3)

    a = omega / theta
    a_hat = vector_to_matrix(a)

    R = np.cos(theta) * np.eye(3) + (1 - np.cos(theta)) * np.outer(a, a) + np.sin(theta) * a_hat

    return R


# 重投影误差
# 'omega' is a three-dimensional vector indicating rotation axis and rotation angle
def reprojection_error(omega, A, pix_coors, world_coors):
    R = lie_algebra_to_group(omega)
    num_points = len(pix_coors)
    errors = []

    for i in range(num_points):
        X, Y = world_coors[i]
        x, y = pix_coors[i]

        world_coor = np.array([X, Y, 1])  # 把世界坐标点变成三维的，Z=1

        projected = np.dot(A, np.dot(R, world_coor))

        projected = projected / projected[2]

        error = pix_coors[i] - projected[:2]
        errors.extend(error)

    # print(f"Current params: R=\n{R}")

    return errors


def get_extrinsic(A, pix_coords, world_coords):
    pixel_coords = pix_coords
    world_coords = world_coords

    theta = 0.9
    theta = np.radians(theta)
    world_coords = np.tan(world_coords * theta)

    initial_omega = np.array([0.1, 0.1, 0.1])  # 假设小的旋转
    intrinsic_param = A

    result = least_squares(reprojection_error, initial_omega,
                           args=(intrinsic_param, pixel_coords, world_coords), loss='cauchy')  # loss = 'huber'

    optimized_rotation_v = result.x
    optimized_rotation_m = lie_algebra_to_group(optimized_rotation_v)

    # print("rotation vector is :\n", optimized_rotation_v)
    # print("rotation matrix is :\n", optimized_rotation_m)

    return optimized_rotation_m
