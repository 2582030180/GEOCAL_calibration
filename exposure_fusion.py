import os
import sys
import glob
import numpy as np
import cv2
import argparse
import yaml


def show_image(message, src):
    cv2.namedWindow(message, 0)
    cv2.imshow(message, src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def gauss_curve(src, mean, sigma):
    dst = np.exp(-(src - mean)**2 / (2 * sigma**2))
    return dst


class ExposureFusion(object):
    def __init__(self, sequence, best_exposedness=0.5, sigma=0.2, eps=1e-12, exponents=(1.0, 1.0, 1.0), layers=7):
        self.sequence = sequence  # [N, H, W, 3], (0..1), float32
        self.img_num = sequence.shape[0]
        self.best_exposedness = best_exposedness
        self.sigma = sigma
        self.eps = eps
        self.exponents = exponents
        self.layers = layers

    @staticmethod
    def cal_contrast(src):
        gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
        laplace_kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)
        contrast = cv2.filter2D(gray, -1, laplace_kernel, borderType=cv2.BORDER_REPLICATE)
        return np.abs(contrast)

    @staticmethod
    def cal_saturation(src):
        mean = np.mean(src, axis=-1)
        channels = [(src[:, :, c] - mean)**2 for c in range(3)]
        saturation = np.sqrt(np.mean(channels, axis=0))
        return saturation

    @staticmethod
    def cal_exposedness(src, best_exposedness, sigma):
        exposedness = [gauss_curve(src[:, :, c], best_exposedness, sigma) for c in range(3)]
        exposedness = np.prod(exposedness, axis=0)
        return exposedness

    def cal_weight_map(self):
        weights = []
        for idx in range(self.sequence.shape[0]):
            contrast = self.cal_contrast(self.sequence[idx])
            saturation = self.cal_saturation(self.sequence[idx])
            exposedness = self.cal_exposedness(self.sequence[idx], self.best_exposedness, self.sigma)
            weight = np.power(contrast, self.exponents[0]) * np.power(saturation, self.exponents[1]) * np.power(exposedness, self.exponents[2])
            # Gauss Blur
            # weight = cv2.GaussianBlur(weight, (21, 21), 2.1)
            weights.append(weight)
        weights = np.stack(weights, 0) + self.eps
        # normalize
        weights = weights / np.expand_dims(np.sum(weights, axis=0), axis=0)
        return weights

    def naive_fusion(self):
        weights = self.cal_weight_map()  # [N, H, W]
        weights = np.stack([weights, weights, weights], axis=-1)  # [N, H, W, 3]
        naive_fusion = np.sum(weights * self.sequence * 255, axis=0)
        naive_fusion = np.clip(naive_fusion, 0, 255).astype(np.uint8)
        return naive_fusion

    def build_gaussian_pyramid(self, high_res):
        gaussian_pyramid = [high_res]
        for idx in range(1, self.layers):
            gaussian_pyramid.append(cv2.GaussianBlur(gaussian_pyramid[-1], (5, 5), 0.83)[::2, ::2])
        return gaussian_pyramid

    def build_laplace_pyramid(self, gaussian_pyramid):
        laplace_pyramid = [gaussian_pyramid[-1]]
        for idx in range(1, self.layers):
            size = (gaussian_pyramid[self.layers - idx - 1].shape[1], gaussian_pyramid[self.layers - idx - 1].shape[0])
            upsampled = cv2.resize(gaussian_pyramid[self.layers - idx], size, interpolation=cv2.INTER_LINEAR)
            laplace_pyramid.append(gaussian_pyramid[self.layers - idx - 1] - upsampled)
        laplace_pyramid.reverse()
        return laplace_pyramid

    def multi_resolution_fusion(self):
        weights = self.cal_weight_map()  # [N, H, W]
        weights = np.stack([weights, weights, weights], axis=-1)  # [N, H, W, 3]

        image_gaussian_pyramid = [self.build_gaussian_pyramid(self.sequence[i] * 255) for i in range(self.img_num)]
        image_laplace_pyramid = [self.build_laplace_pyramid(image_gaussian_pyramid[i]) for i in range(self.img_num)]
        weights_gaussian_pyramid = [self.build_gaussian_pyramid(weights[i]) for i in range(self.img_num)]

        fused_laplace_pyramid = [np.sum([image_laplace_pyramid[n][l] *
                                         weights_gaussian_pyramid[n][l] for n in range(self.img_num)], axis=0) for l in range(self.layers)]

        result = fused_laplace_pyramid[-1]
        for k in range(1, self.layers):
            size = (fused_laplace_pyramid[self.layers - k - 1].shape[1], fused_laplace_pyramid[self.layers - k - 1].shape[0])
            upsampled = cv2.resize(result, size, interpolation=cv2.INTER_LINEAR)
            result = upsampled + fused_laplace_pyramid[self.layers - k - 1]
        result = np.clip(result, 0, 255).astype(np.uint8)

        return result


def exposure_fusion(output_folder):
    with open('config.yaml', 'r', encoding='utf-8') as config_file:
        config = yaml.safe_load(config_file)
    # output_folder = config['output_path']['output_folder']
    root_path = config['exposure_fusion']['root_path']

    sequence_path = [os.path.join(root_path, fname) for fname in os.listdir(root_path)]
    sequence = np.stack([cv2.imread(path) for path in sequence_path], axis=0)

    mef = ExposureFusion(sequence.astype(np.float32) / 255.0)
    naive_fusion_result = mef.naive_fusion()
    multi_res_fusion = mef.multi_resolution_fusion()

    output_image_path = os.path.join(output_folder, "exposure_fusion.bmp")
    cv2.imwrite(output_image_path,multi_res_fusion)
