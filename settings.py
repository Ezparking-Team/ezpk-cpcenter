import json

from dotenv import load_dotenv, find_dotenv
import os

load_dotenv(verbose=True)

# ======== 基础设置 ========
SECRET_KEY = os.getenv("SECRET_KEY")

# ======== 视频设置 ========
TIME_FORMAT = os.getenv("TIME_FORMAT")

# 车牌定位器配置
CARDPOS_lower_blue = [100, 100, 50]  # 蓝色车牌最低色值
CARDPOS_higher_blue = [150, 150, 255]  # 蓝色车牌最高色值

CARDPOS_gaussian_ksize = (3, 3)  # 高斯降噪 卷积核大小
CARDPOS_gaussian_sigma = 1  # 高斯降噪 Sigma

CARDPOS_close_kernel = (20, 5)  # 闭操作方框大小，一般车牌在水平轴较长，所以在配置中要强调x轴长度、

CARDPOS_noise_size_min = 0.01  # 非杂音区域定义, 在此区域外的目标会被过滤
CARDPOS_noise_size_max = 0.08  # 单位为小数百分比（在图像中的占比）

CARDPOS_standard = 3  # 标准车牌长宽比，一般为长宽比余数，例如 3/1 为 3
CARDPOS_tolerance = 1  # 长宽比容差率，差错越小越精确
CARDPOS_offset = 3  # 车牌选择区域大小偏差，用于解决闭操作后区域过大或过小的问题

# 车牌识别设置
CARDRECO_shoot_time = 1000  # 当检测到车牌超过该毫秒值，即拍照
# CARDRECO_shoot_rest = 10000  # 拍照一次后的休眠时间（休眠期间不拍照）
