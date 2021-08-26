import cv2

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
        pos = pos[0]

        cropped = self.image[pos[0][1]:pos[1][1], pos[0][0]:pos[1][0]]

        return cropped

    @staticmethod
    def divided_image(cropped):
        """分割车牌"""
        return FindCarNumber.cut_text(cropped)
