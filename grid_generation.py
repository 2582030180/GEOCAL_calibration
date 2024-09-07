import os
import yaml
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import math
import sys
import json
from collections import defaultdict


class Point2d:
    def __init__(self, px, py):
        self.px = px
        self.py = py


class NetPoint:
    def __init__(self, oriloc, newloc, searchv, neighbor, flag):
        self.oriloc = oriloc  # presentation in original coordinates
        self.newloc = newloc  # presentation in new axis, center point is (0,0)
        self.searchv = searchv  # List of 4 search vectors  left-up-right-down
        self.neighbors = neighbor  # List of neighbors
        self.flag = flag  # whether it has been searched, 0 is not 1 is yes


def normalization(vec):
    x_max = max(v[0] for v in vec)
    x_min = min(v[0] for v in vec)
    y_max = max(v[1] for v in vec)
    y_min = min(v[1] for v in vec)

    normalized = []
    for px, py in vec:
        nx = 2.0 * (px - x_min) / (x_max - x_min) - 1.0
        ny = 2.0 * (py - y_min) / (y_max - y_min) - 1.0
        normalized.append((nx, ny))

    return normalized


##### 改变半径参数 #######
def find_neighbors(point):
    r = config['grid_generation']['r']
    tree = KDTree(coordinates)
    indices_within_radius = tree.query_ball_point(point, r)
    points_within_radius = coordinates[indices_within_radius]  # 在它周围半径内的所有点
    k = len(points_within_radius)

    if k == 0:
        return []

    tree2 = KDTree(np.array(points_within_radius))
    _, indices = tree2.query(point, k=k)

    if isinstance(indices, np.int64) or isinstance(indices, int):
        indices = [indices]

    nbs = points_within_radius[indices]
    filtered_neighbors = [p for p in nbs if not np.array_equal(p, point)]

    # 计算每个点的角度
    start_angle = -3 * math.pi / 4
    angles = []
    for p in filtered_neighbors:
        angle = math.atan2(p[1] - point[1], p[0] - point[0])
        # 转换到正角度并调整起始角度
        angle = (angle - start_angle) % (2 * math.pi)

        while any(abs(existing_angle - angle) < 1e-7 for existing_angle, _ in angles):
            angle += 1e-7
        angles.append((angle, p))
    # 按角度排序
    angles.sort()

    # 返回排序后的点
    sorted_neighbors = [p for angle, p in angles]

    return sorted_neighbors  # 所有的邻居，数目可能小于4


def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def is_within_angle_range(v, angle_range, direction_vector):
    # 计算v与direction_vector之间的角度
    dot_product = np.dot(v, direction_vector)
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle = np.arccos(dot_product)
    # 将角度转换为弧度
    max_angle_radians = np.radians(angle_range)
    return angle <= max_angle_radians


def is_within_distance_range(point_a, point_b, min_distance, max_distance):
    distance = np.linalg.norm((point_a[0] - point_b[0], point_a[1] - point_b[1]))
    # print(" a is ",point_a, "b is ",point_b)
    return min_distance <= distance <= max_distance


##### 改变范围参数 ######
def judge_range(point_a, point_b, direction_vector, angle_range=30):
    min_distance = config['grid_generation']['min_distance']
    max_distance = config['grid_generation']['max_distance']
    # print(" a is ",point_a, "b is ",point_b)
    # if counter > 3330:
    #     max_distance = 55
    #     min_distance = 30

    if is_within_distance_range(point_a, point_b, min_distance, max_distance):
        vector_ab = normalize_vector(point_b - point_a)
        if is_within_angle_range(vector_ab, angle_range, direction_vector):
            return True

    return False


def generate_all_point(root):
    global counter
    global grid_points
    global searched_point
    global point_needed
    stack = [root]
    # print("root is", root.newloc.px, root.newloc.py)
    # print("neighbors are:", root.neighbors)
    while stack:
        print(counter)
        current = stack.pop()
        # print(current.oriloc.px, current.oriloc.py)
        # print("neighbor:", current.neighbors)
        all_found = 0
        for nei in current.neighbors:
            for sear in searched_point:
                if np.array_equal(nei, sear):
                    all_found += 1
        # print(all_found == len(root.neighbors))
        if all_found == len(current.neighbors):

            if counter != len(grid_points) - 1:
                counter += 1
                stack.append(grid_points[counter])
            continue

        root_c = (current.oriloc.px, current.oriloc.py)
        for i in range(len(current.neighbors)):
            c = current.neighbors[i]

            found = False
            for point in searched_point:
                if np.array_equal(c, point):
                    found = True
                    break
            if found:
                continue

            ll = None
            s = current.searchv
            for j in range(len(current.searchv)):

                if judge_range(root_c, c, current.searchv[j]):

                    if abs(current.searchv[j][0]) <= abs(current.searchv[j][1]):
                        if root.searchv[j][1] >= 0:
                            # 下方邻居
                            ll = Point2d(current.newloc.px, current.newloc.py + 1)
                            tempp = normalize_vector(c - root_c)
                            s[j] = tempp
                        else:
                            # 上方邻居
                            ll = Point2d(current.newloc.px, current.newloc.py - 1)
                            tempp = normalize_vector(c - root_c)
                            s[j] = tempp
                    else:
                        if current.searchv[j][0] <= 0:
                            # 左侧邻居
                            ll = Point2d(current.newloc.px - 1, current.newloc.py)
                            tempp = normalize_vector(c - root_c)
                            s[j] = tempp
                        else:
                            # 右侧邻居
                            ll = Point2d(current.newloc.px + 1, current.newloc.py)
                            tempp = normalize_vector(c - root_c)
                            s[j] = tempp
                    break
                else:
                    continue

            if ll:

                n = find_neighbors(c)

                searched_point.append(c)

                c = Point2d(current.neighbors[i][0], current.neighbors[i][1])
                new = NetPoint(c, ll, s, n, 1)
                grid_points.append(new)

                point_needed[(grid_points[-1].oriloc.px, grid_points[-1].oriloc.py)] = (grid_points[-1].newloc.px,
                                                                                        grid_points[-1].newloc.py)
                # print(point_needed)  # 用来画新坐标图

            else:

                continue

        if counter != len(grid_points) - 1:
            counter += 1
            stack.append(grid_points[counter])


def grid_generation(output_folder):
    global config
    with open('config.yaml', 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    # output_folder = config['output_path']['output_folder']

    # store all the points position to coordinates, their original (0,0) is in the top-left corner
    coordinates_file_path = os.path.join(output_folder, "centroids_coordinates.txt")
    global coordinates
    coordinates = []

    with open(coordinates_file_path, 'r') as file:
        for line in file:
            x, y = map(float, line.strip().split(','))
            coordinates.append((x, y))

    coordinates = np.array(coordinates)

    # find center point  #######改变主点参数#######
    # 手动输入中心点时启用
    # center_like = config['grid_generation']['center_like']

    # 自动默认离画幅中心最近的点为中心点
    image_path = os.path.join(output_folder, "output_image_with_centroids_sub.bmp")
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    height, width = image.shape[:2]
    cx = height // 2
    cy = width // 2
    center_like = np.array([cx, cy])

    distances = cdist(coordinates, np.array([center_like]))
    closest_index = np.argmin(distances)
    center = coordinates[closest_index]
    coordinates = coordinates[~np.all(coordinates == center, axis=1)]

    # print(counter,grid_points,point_needed,searched_point,search_root)
    global counter
    counter = 0
    # root is center point
    ori_loc = Point2d(center[0], center[1])
    new_loc = Point2d(0, 0)
    neighbors = find_neighbors(center)  # 几个坐标的列表
    search_v = [np.array(n - center) for n in neighbors]  # 与邻居等数量的搜索向量
    search_v = [(float(arr[0]), float(arr[1])) for arr in search_v]
    search_v = normalization(search_v)  # 搜索向量归一化  也是几个坐标的列表
    # initialize root
    search_root = NetPoint(ori_loc, new_loc, search_v, neighbors, 1)
    global grid_points
    # 一张网格，存储所有网点实例
    grid_points = [search_root]  # store all the instances of NetPoint
    global searched_point
    searched_point = [center]
    global point_needed
    point_needed = {
        (grid_points[-1].oriloc.px, grid_points[-1].oriloc.py): (grid_points[-1].newloc.px, grid_points[-1].newloc.py)}


    generate_all_point(search_root)

    # 去掉重复值
    value_to_keys = defaultdict(list)

    for key, value in point_needed.items():
        value_to_keys[value].append(key)

    duplicate = [(value, keys) for value, keys in value_to_keys.items() if len(keys) > 1]
    reverse_dict = {}
    for k, v in point_needed.items():
        reverse_dict[tuple(v)] = k

    # # 获取四个邻近坐标
    for item in duplicate:
        n1 = reverse_dict.get((item[0][0] + 1, item[0][1]))
        n2 = reverse_dict.get((item[0][0] - 1, item[0][1]))
        n3 = reverse_dict.get((item[0][0], item[0][1] + 1))
        n4 = reverse_dict.get((item[0][0], item[0][1] - 1))

        # 自己的坐标
        x1, y1 = item[1][0]
        x2, y2 = item[1][1]
        # print([n1, n2, n3, n4])
        temp = [n1, n2, n3, n4]
        for each in range(len(temp)):
            if temp[each] is None:
                temp[each] = (0, 0)
        # 邻居的坐标
        x3, y3 = temp[0]
        x4, y4 = temp[1]
        x5, y5 = temp[2]
        x6, y6 = temp[3]

        distance_sum1 = (x1 - x3) ** 2 + (y1 - y3) ** 2 + (x1 - x4) ** 2 + (y1 - y4) ** 2 + (x1 - x5) ** 2 + (
                y1 - y5) ** 2 + (x1 - x6) ** 2 + (y1 - y6) ** 2
        distance_sum2 = (
                    (x2 - x3) ** 2 + (y2 - y3) ** 2 + (x2 - x4) ** 2 + (y2 - y4) ** 2 + (x2 - x5) ** 2 + (y2 - y5) ** 2
                    + (x2 - x6) ** 2 + (y2 - y6) ** 2)

        if distance_sum1 >= distance_sum2:
            del point_needed[item[1][0]]
        else:
            del point_needed[item[1][1]]

    # 预测缺失值

    # 创建一个集合来存储所有存在的组合
    existing_combinations = set((v[0], v[1]) for v in point_needed.values())

    # 创建一个列表来存储不存在的组合
    missing_combinations = []

    min_first = float('inf')
    max_first = float('-inf')
    min_second = float('inf')
    max_second = float('-inf')
    for value in point_needed.values():
        first, second = value
        if first < min_first:
            min_first = first
        if first > max_first:
            max_first = first
        if second < min_second:
            min_second = second
        if second > max_second:
            max_second = second

    # 遍历所有可能的组合     #####改变参数######
    for i in range(min_first + 1, max_first):
        for j in range(min_second + 1, max_second):
            if (i, j) not in existing_combinations:
                missing_combinations.append((i, j))

    # 获取四个邻近坐标
    for item in missing_combinations:
        n1 = reverse_dict.get((item[0] + 1, item[1]))
        n2 = reverse_dict.get((item[0] - 1, item[1]))
        n3 = reverse_dict.get((item[0], item[1] + 1))
        n4 = reverse_dict.get((item[0], item[1] - 1))

        mm = 0
        temp = [n1, n2, n3, n4]
        for each in range(len(temp)):
            if temp[each] is None:
                mm = 1
                break
        if mm == 1:
            continue

        point_needed[(round((n3[0] + n4[0]) / 2, 3), round((n1[1] + n2[1]) / 2, 3))] = item

    coordinates_file_path = os.path.join(output_folder, "final_coordinates.txt")
    with open(coordinates_file_path, 'w') as file:
        for key, value in point_needed.items():
            file.write(f"{key}:{value}\n")

    # 在原图上画新坐标

    image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    pil_image = Image.fromarray(image_color)
    draw = ImageDraw.Draw(pil_image)
    #####改变字体大小显示#######
    font_size = config['grid_generation']['font_size']
    font = ImageFont.truetype("arial.ttf", font_size)

    # 在每个坐标位置写出对应的value值
    for coord, value in point_needed.items():
        x, y = coord
        text = f"({value[0]}, {value[1]})"
        text_bbox = draw.textbbox((0, 0), text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        draw.text((x - text_width // 2, y + 2), text, font=font, fill=(255, 255, 255))

    output_path = os.path.join(output_folder, "output_image_with_coordinates.png")
    pil_image.save(output_path)


