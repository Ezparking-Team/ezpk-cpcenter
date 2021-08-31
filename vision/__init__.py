import random

import cv2
import numpy as np

from vision.find_car_number import FindCarNumber


class Vision:
    def __init__(self, image_path):
        self.image = cv2.imread(image_path)

    def get_car_number_pos(self):
        """获取车牌位置"""
        c = FindCarNumber(image=self.image).find_card_pos()
        return c["pos"]

    def cropped_image(self, pos):
        """裁剪图片至车牌"""
        try:
            pos = pos[0]
        except IndexError:
            raise ValueError("Plate not found!")

        cropped = self.image[pos[0][1]:pos[1][1], pos[0][0]:pos[1][0]]

        return cropped

    @staticmethod
    def divided_image(cropped):
        """分割车牌"""
        try:
            divided = FindCarNumber.cut_text_peak(cropped)
        except IndexError as e:  # 无法使用峰值法，则用裁剪法
            print(e)
            divided = FindCarNumber.cut_text(cropped)

        for key in divided:
            # 调整大小
            # divided[key] = cv2.resize(
            #     divided[key],
            #     (32, 32),
            #     interpolation=cv2.INTER_CUBIC
            # )

            # 高斯
            # divided[key] = cv2.GaussianBlur(
            #     divided[key],
            #     (3, 3), 1
            # )

            # 保留蓝色区域
            lower_blue = np.array([0, 30, 180])
            higher_blue = np.array([360, 200, 360])

            divided[key] = cv2.cvtColor(divided[key], cv2.COLOR_BGR2HSV)
            divided[key] = cv2.inRange(divided[key], lower_blue, higher_blue)

            # 保存图片
            if key == "region":
                cv2.imwrite("demo_images/regions/%s.jpg" % random.randint(1, 999999999), divided[key])
            else:
                cv2.imwrite("demo_images/chars/%s.jpg" % random.randint(1, 999999999), divided[key])

        return divided
