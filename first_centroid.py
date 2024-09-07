import os
import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw


def first_centroid(folder_path):

    with open('config.yaml', 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)

    output_folder = config['first_centroid']['output_folder']
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    else:
        for file in os.listdir(output_folder):
            file_path = os.path.join(output_folder, file)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)

    threshold = config['first_centroid']['threshold_value']
    min_area = config['first_centroid']['min_area']
    img_counter = 0

    for image_path in folder_path:
        original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # #####改变阈值######
        modified_image = np.where(original_image > threshold, original_image, 0)

        height, width = original_image.shape
        center_x_start = 3 * width // 9
        center_x_end = 6 * width // 9
        center_y_start = 3 * height // 9
        center_y_end = 6 * height // 9

        _, contours, _ = cv2.findContours(modified_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        ######改变提取的最小面积#######
        centroids = []

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            if (center_x_start <= x <= center_x_end and center_y_start <= y <= center_y_end) or \
                    (center_x_start <= x + w <= center_x_end and center_y_start <= y + h <= center_y_end):
                if cv2.contourArea(contour) < min_area:
                    continue

            mask = np.zeros_like(original_image)
            cv2.drawContours(mask, [contour], -1, 255, thickness=cv2.FILLED)

            masked_image = cv2.bitwise_and(original_image, original_image, mask=mask)
            M = cv2.moments(masked_image)
            if M['m00'] != 0:
                cx = M['m10'] / M['m00']
                cy = M['m01'] / M['m00']
                centroids.append((cx, cy))

        image_size = (original_image.shape[1], original_image.shape[0])
        output_image = np.zeros((image_size[1], image_size[0], 3), dtype=np.uint8)
        output_image = Image.fromarray(output_image)
        draw = ImageDraw.Draw(output_image)

        for centroid in centroids:
            draw.ellipse((centroid[0] - 1, centroid[1] - 1, centroid[0] + 1, centroid[1] + 1), fill=(255, 255, 255))

        img_counter += 1

        output_image_path = os.path.join(output_folder, f"centroid_{img_counter}.bmp")
        output_image.save(output_image_path, format='BMP')



