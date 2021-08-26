import time

import cv2
import numpy as np

import settings


class FindCarNumber:
    def __init__(self, image):
        self.image = image

    def find_card_pos(self):
        """
        搜索车牌位置算法
        """
        t1 = time.time()
        # ==============================================

        # 高斯除噪
        image = cv2.GaussianBlur(
            self.image,
            settings.CARDPOS_gaussian_ksize,
            settings.CARDPOS_gaussian_sigma
        )
        # cv2.imshow("blur", image)

        # 只保留蓝色区域
        lower_blue = np.array(settings.CARDPOS_lower_blue)
        higher_blue = np.array(settings.CARDPOS_higher_blue)

        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        image = cv2.inRange(image, lower_blue, higher_blue)
        # cv2.imshow("blue", image)

        # 闭操作
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, settings.CARDPOS_close_kernel)  # 定义方框大小
        image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)  # 闭操作
        # cv2.imshow("close", image)

        # 寻找轮廓
        contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        # 将轮廓规整为长方形
        rectangles = []
        for c in contours:
            x = []
            y = []
            for point in c:
                y.append(point[0][0])
                x.append(point[0][1])
            r = [min(y), min(x), max(y), max(x)]
            rectangles.append(r)

        # print("reda: %d" % len(rectangles))

        # 恢复色彩
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        # 过滤器
        filtered = []
        for r in rectangles:
            x = abs(r[0] - r[2])  # 边长
            y = abs(r[1] - r[3])

            # === 过滤杂音(依照画面比例) ===
            noise_size_min = settings.CARDPOS_noise_size_min
            noise_size_max = settings.CARDPOS_noise_size_max

            noise = x * y
            screen = image.shape[0] * image.shape[1]

            noise_on_screen = noise / screen

            if noise_on_screen == 0:  # 排除空噪音
                continue
            elif not noise_size_min < noise_on_screen < noise_size_max:  # 排除设定噪音范围
                # print("Noise")
                continue

            # === 过滤非车牌比例 ===
            standard = settings.CARDPOS_standard
            tolerance = settings.CARDPOS_tolerance

            deviation = abs(x / y - standard)
            if deviation > tolerance:
                # print("Not Car Number")
                continue

            # === 添加过滤后的结果 ===
            # print("Success")
            filtered.append(r)

        # 记录车牌位置
        object_list = []
        offset = 0

        for r in filtered:  # 遍历轮廓列表
            object_list.append(([r[0] - offset, r[1] - offset], [r[2] + offset, r[3] + offset]))

        # 更新车牌位置表
        self.cards_pos = object_list

        # ==============================================
        return {
            "pos": object_list,
            "usedTime": time.time() - t1
        }

    @staticmethod
    def cut_text(image):
        """
        剪切车牌字符
        :return:
        """

        cut_info = {
            "region": 0.15,
            "region-city": 0.15,
            "spacer": 0.02,
            "char-1": 0.13,
            "char-2": 0.13,
            "char-3": 0.13,
            "char-4": 0.13,
            "char-5": 0.13,
        }

        image_height = image.shape[0]
        image_width = image.shape[1]

        last_p = 0  # 上次轮询到的百分比
        cut_images = cut_info  # 剪切好的图片
        for key in cut_info:
            now_p = last_p + cut_info[key]  # 当前轮询到的百分比
            left_x = int(image_width * last_p)
            right_x = int(image_width * now_p)

            # cv2.rectangle(image, [left_x, 0], [right_x, image_height], (0, 255, 0), 1)

            cut = image[0:image_height, left_x:right_x]
            cut_images[key] = cut

            last_p = now_p
            # cv2.imshow(key, cut)

        # cv2.imshow("image", image)
        # cv2.waitKey()

        return cut_images
