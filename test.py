import json
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
import yaml


def read_data(input_file):
    read_path = input_file
    with open(read_path, 'r') as file:
        data = file.read()

    pixel_coords = []  # 像素坐标系，以像素为单位
    camera_coords = []  # 相机坐标系

    # 正则表达式读取
    pattern = re.compile(r'\(([^)]+)\):\(([^)]+)\)')
    matches = pattern.findall(data)

    for match in matches:
        pixel_coord = tuple(map(float, match[0].split(',')))
        camera_coord = tuple(map(int, match[1].split(',')))
        pixel_coords.append(pixel_coord)
        camera_coords.append(camera_coord)

    pixel_coords = np.array(pixel_coords)
    camera_coords = np.array(camera_coords)

    return pixel_coords, camera_coords


def test(output_folder, input_file):

    with open('config.yaml', 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    # output_folder = config['output_path']['output_folder']
    json_file_path = os.path.join(output_folder, "calibration_data.json")
    with open(json_file_path, 'r') as json_file:
        calibration_data = json.load(json_file)

    k1 = calibration_data['k1']
    k2 = calibration_data['k2']
    k3 = calibration_data['k3']
    A = np.array(calibration_data['A'])
    R = np.array(calibration_data['R'])

    theta = 0.9
    theta = np.radians(theta)
    # 世界坐标系下的点 (假设 Z = 1)
    pixel_coords, world_coords = read_data(input_file)
    world_points = []
    for i in range(len(world_coords)):
        X, Y = world_coords[i]
        world_points.append([np.tan(theta * X), np.tan(theta * Y)])

    world_points = np.array(world_points)
    # 将点转换为齐次坐标
    world_points_homogeneous = np.hstack([world_points, np.ones((world_points.shape[0], 1))])

    # 转换到相机坐标系
    camera_points = R @ world_points_homogeneous.T

    # 归一化图像坐标
    normalized_image_points = camera_points[:2, :] / camera_points[2, :]

    # 应用畸变校正
    x = normalized_image_points[0, :]
    y = normalized_image_points[1, :]
    r2 = x ** 2 + y ** 2
    radial_distortion = 1 + k1 * r2 + k2 * r2 ** 2 + k3 * r2 ** 3
    x_distorted = x * radial_distortion
    y_distorted = y * radial_distortion

    # 转换为像素坐标
    image_points = A @ np.vstack([x_distorted, y_distorted, np.ones_like(x_distorted)])
    image_points /= image_points[2, :]  # 齐次归一化
    #######改变图片画布大小########
    canvas_size = (720, 540)
    output_image = Image.new('RGB', canvas_size, color=(0, 0, 0))
    draw = ImageDraw.Draw(output_image)
    # 将像素点绘制到黑色画布上
    for point in zip(image_points[0, :], image_points[1, :]):
        u = int(round(point[0]))
        v = int(round(point[1]))

        if 0 <= u < canvas_size[0] and 0 <= v < canvas_size[1]:
            draw.point((u, v), fill=(255, 255, 255))

            if u - 1 >= 0:
                draw.point((u - 1, v), fill=(255, 255, 255))
            if u + 1 < canvas_size[0]:
                draw.point((u + 1, v), fill=(255, 255, 255))
            if v - 1 >= 0:
                draw.point((u, v - 1), fill=(255, 255, 255))
            if v + 1 < canvas_size[1]:
                draw.point((u, v + 1), fill=(255, 255, 255))

    output_image_path = os.path.join(output_folder, "calibration_effect.bmp")
    output_image.save(output_image_path, format='BMP')

    print(f"Image with distorted points saved to {output_image_path}")
