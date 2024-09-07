import json
import os.path
import numpy as np
import re
import yaml
from intrinsic import get_intrinsic
from distortion import get_distortion
from extrinsic import get_extrinsic


# 读取两个坐标系下的坐标
def read_data(input_file):
    read_path = input_file
    # read_path = os.path.join(output_folder, "final_coordinates.txt")
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


def refine_all(R, pixel_coords, camera_coords):
    A = get_intrinsic(R, pixel_coords, camera_coords)
    k1, k2, k3 = get_distortion(A, R, pixel_coords, camera_coords)
    R = get_extrinsic(A, pixel_coords, camera_coords)

    return A, R, k1, k2, k3


def calibrate(output_folder, input_file):
    with open('config.yaml', 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    # output_folder = config['output_path']['output_folder']
    pixel_coords, camera_coords = read_data(input_file)

    # default value
    R = np.eye(3)

    A = get_intrinsic(R, pixel_coords, camera_coords)
    k1, k2, k3 = get_distortion(A, R, pixel_coords, camera_coords)
    R = get_extrinsic(A, pixel_coords, camera_coords)

    # 根据需求反复微调几次
    counter = 0
    for i in range(2):
        counter += 1
        print(counter)
        A, R, k1, k2, k3 = refine_all(R, pixel_coords, camera_coords)

    print("Intrinsic matrix is\n", A)
    print("\nDistortion coefficient are\n", k1, k2, k3)
    print("\nExtrinsic matrix is\n", R)
    para_path = os.path.join(output_folder, "calibration result.txt")
    with open(para_path, 'w') as file:
        file.write(f"Intrinsic matrix is\n {A}\n\n")
        file.write(f"Distortion coefficient are: \nk1={k1}, k2={k2}, k3={k3} \n\n")
        file.write(f"Extrinsic matrix is\n {R}\n")

    # 结果存储成json文件便于后续测试
    calibration_data = {
        "k1": k1,
        "k2": k2,
        "k3": k3,
        "A": A.tolist(),
        "R": R.tolist()
    }
    json_file_path = os.path.join(output_folder, "calibration_data.json")
    with open(json_file_path, 'w') as json_file:
        json.dump(calibration_data, json_file, indent=4)

