#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Date    : 2022/12
# @Author  : Qinbing Fu and Zhiqiang Li
# @PaperName: Harmonizing motion and contrast vision for robust looming detection

import copy
import numpy as np
import cv2


class LGMD_P_ON_OFF:
    """
    path                   : path of the folder where the dataset is located
    beta_1                 : weight coefficient of inhibition unit
    I_size                 : size of convolution kernel
    alpha_1                : baseline sensitivity value of contrast normalization
    alpha_2,alpha_3        : influence coefficient of contrast and motion computation in ON channel
    alpha_4,alpha_5        : influence coefficient of contrast and motion computation in OFF channel
    theta_1,theta_2,theta_3: influence coefficient of ON/OFF channel
    gauss_size,gauss_div   : size and standard deviation of gaussian convolution kernel
    C_w                    : a constant used to calculate w
    delta_C                : a small real number used to prevent calculation errors
    C_de,T_de              : a constant used to calculate T_g
    T_g                    : threshold of grouping mechanism
    K_I,K_C,K_g            : Three convolution kernels in computation
    """

    def __init__(self, path):
        self.video = cv2.VideoCapture(path)
        self.height, self.width, self.frame_numbers, self.fps = self.get_video_inf()
        self.gray_video = np.zeros([self.height, self.width, self.frame_numbers])
        self.P = np.zeros([self.height, self.width, self.frame_numbers])
        self.P_on = np.zeros([self.height, self.width, self.frame_numbers])
        self.P_off = np.zeros([self.height, self.width, self.frame_numbers])
        self.E_on = np.zeros([self.height, self.width, self.frame_numbers])
        self.E_off = np.zeros([self.height, self.width, self.frame_numbers])
        self.I_on = np.zeros([self.height, self.width, self.frame_numbers])
        self.I_off = np.zeros([self.height, self.width, self.frame_numbers])
        self.S_on = np.zeros([self.height, self.width, self.frame_numbers])
        self.S_on_hat = np.zeros([self.height, self.width, self.frame_numbers])
        self.S_on_out = np.zeros([self.height, self.width, self.frame_numbers])
        self.S_off = np.zeros([self.height, self.width, self.frame_numbers])
        self.S_off_hat = np.zeros([self.height, self.width, self.frame_numbers])
        self.S_off_out = np.zeros([self.height, self.width, self.frame_numbers])
        self.S = np.zeros([self.height, self.width, self.frame_numbers])
        self.C_on = np.zeros([self.height, self.width, self.frame_numbers])
        self.C_off = np.zeros([self.height, self.width, self.frame_numbers])
        self.Ce = np.zeros([self.height, self.width, self.frame_numbers])
        self.G = np.zeros([self.height, self.width])
        self.mp = np.zeros([self.frame_numbers])
        self.mp_hat = np.zeros([self.frame_numbers])
        self.M_on = np.zeros([self.height, self.width, self.frame_numbers])
        self.M_off = np.zeros([self.height, self.width, self.frame_numbers])
        self.M_hat_on = np.zeros([self.height, self.width, self.frame_numbers])
        self.M_hat_off = np.zeros([self.height, self.width, self.frame_numbers])
        self.beta = 0.4
        self.I_size = 3
        self.alpha_1 = 3
        self.alpha_2 = 1
        self.alpha_3 = 1
        self.alpha_4 = 1
        self.alpha_5 = 1
        self.theta_1 = 1
        self.theta_2 = 1
        self.theta_3 = 1
        self.gauss_size = 9
        self.gauss_div = 5
        self.C_w = 4
        self.delta_C = 0.01
        self.C_de = 0.5
        self.T_de = 0.7  # 0.7
        self.K_I = np.zeros([self.I_size, self.I_size])
        self.basic_weight = 0.125
        self.K_C = np.zeros([3, 3])
        self.K_g = np.zeros([3, 3])
        self.init_kernel()

    # Get the width, height, frame number and frame rate of the
    # experimental video during class initialization.
    def get_video_inf(self):
        height = int(self.video.get(cv2.CAP_PROP_FRAME_HEIGHT))
        width = int(self.video.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_numbers = int(self.video.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.video.get(cv2.CAP_PROP_FPS)
        return height, width, frame_numbers, fps

    # Class initialization with three convolution kernels
    def init_kernel(self):
        for i in range(3):
            for j in range(3):
                self.K_g[i, j] = 1 / 9
                self.K_C[i, j] = -1 / 8
                if i == 1 and j == 1:
                    self.K_C[i, j] = 1

        a = 0
        b = 0
        while (a < self.I_size / 2):
            b = 0
            while (b < self.I_size / 2):
                self.K_I[a, b] = self.K_I[self.I_size - 1 - a, b] = self.K_I[a, self.I_size - 1 - b] = self.K_I[
                    self.I_size - 1 - a, self.I_size - 1 - b] = self.basic_weight * (np.sqrt(a * a + b * b) + 1)
                b = b + 1
            a = a + 1
        self.K_I[a - 1, b - 1] = 0

    # Half-wave rectification of image frames
    @staticmethod
    def half_wave_ON(matrix):
        temp_matrix = copy.deepcopy(matrix)
        temp_matrix[temp_matrix < 0] = 0
        return temp_matrix

    @staticmethod
    def half_wave_OFF(matrix):
        temp_matrix = copy.deepcopy(matrix)
        temp_matrix[temp_matrix > 0] = 0
        return temp_matrix

    # The entry to run this code
    def run(self):
        for ftp in range(self.frame_numbers):
            ret, color_frame = self.video.read()
            self.gray_video[:, :, ftp] = cv2.cvtColor(color_frame, cv2.COLOR_RGB2GRAY)
            if ftp != 0:
                frame_diff = self.gray_video[:, :, ftp] - self.gray_video[:, :, ftp - 1]
                self.looming_perception(frame_diff, ftp)
        return self.mp

    # Main computation process of motion and contrast vision
    def looming_perception(self, frame_diff, t):
        # Computational Retina
        self.P[:, :, t] = frame_diff

        # Computational Lamina
        self.P_on[:, :, t] = LGMD_P_ON_OFF.half_wave_ON(self.P[:, :, t])
        self.P_off[:, :, t] = -LGMD_P_ON_OFF.half_wave_OFF(self.P[:, :, t])

        # Computational Medulla
            # dynamic contrast normalization mechanism
        self.M_hat_on[:, :, t] = cv2.GaussianBlur(self.P_on[:, :, t], (self.gauss_size, self.gauss_size),
                                                  self.gauss_div)
        self.M_hat_off[:, :, t] = cv2.GaussianBlur(self.P_off[:, :, t], (self.gauss_size, self.gauss_size),
                                                   self.gauss_div)
        self.M_on[:, :, t] = np.tanh(self.P_on[:, :, t] / (self.M_hat_on[:, :, t] + self.alpha_1))
        self.M_off[:, :, t] = np.tanh(self.P_off[:, :, t] / (self.M_hat_off[:, :, t] + self.alpha_1))

            # parallel ON/OFF contrast channels
        self.C_on[:, :, t] = np.abs(cv2.filter2D(self.M_on[:, :, t], -1, self.K_C))
        self.C_off[:, :, t] = np.abs(cv2.filter2D(self.M_off[:, :, t], -1, self.K_C))

            # parallel ON motion pathways
        self.E_on[:, :, t] = self.M_on[:, :, t]
        self.I_on[:, :, t] = cv2.filter2D(self.E_on[:, :, t - 1], -1, self.K_I)
        self.S_on[:, :, t] = self.E_on[:, :, t] - self.beta * self.I_on[:, :, t]

            # parallel OFF motion pathways
        self.E_off[:, :, t] = self.M_off[:, :, t]
        self.I_off[:, :, t] = cv2.filter2D(self.E_off[:, :, t - 1], -1, self.K_I)
        self.S_off[:, :, t] = self.E_off[:, :, t] - self.beta * self.I_off[:, :, t]

            # Local summation of motion and contrast signals
        self.S_on_out[:, :, t] = LGMD_P_ON_OFF.half_wave_ON(
            self.alpha_2 * self.S_on[:, :, t] - self.alpha_3 * self.C_on[:, :, t])
        self.S_off_out[:, :, t] = LGMD_P_ON_OFF.half_wave_ON(
            self.alpha_4 * self.S_off[:, :, t] - self.alpha_5 * self.C_off[:, :, t])

        self.S[:, :, t] = self.theta_1 * self.S_on_out[:, :, t] + self.theta_2 * self.S_off_out[:, :,t] + self.theta_3 * self.S_on_out[:, :,t] * self.S_off_out[:, :, t]
        # Computational Lobula
        self.Ce[:, :, t] = cv2.filter2D(self.S[:, :, t], -1, self.K_g)
        w = np.max(self.Ce[:, :, t]) / self.C_w + self.delta_C
        self.G[:, :] = self.S[:, :, t] * self.Ce[:, :, t] / w
        self.G[self.G * self.C_de < self.T_de] = 0
        k = np.sum(self.G)
        self.mp[t] = np.power(1 + np.exp(-k / (self.height * self.width * 0.1)), -1)
