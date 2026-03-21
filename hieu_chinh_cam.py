import cv2
import numpy as np
import json
import time


json_file = './src/assets/stereo_calib_data_ver_3.json'

CamR = cv2.VideoCapture(1)
CamL = cv2.VideoCapture(0)
# FCAM_WIDTH = 1280 #hd
# FCAM_HEIGHT = 720
FCAM_WIDTH = 640 
FCAM_HEIGHT = 480
CamR.set(cv2.CAP_PROP_FRAME_WIDTH, FCAM_WIDTH)
CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, FCAM_HEIGHT)
CamL.set(cv2.CAP_PROP_FRAME_WIDTH, FCAM_WIDTH)
CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, FCAM_HEIGHT)

# CamR.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# CamR.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# CamL.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# CamL.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Hiệu chỉnh
try:
    with open(json_file, 'r', encoding='utf-8') as f:
        calib_data = json.load(f)
    MLS = np.array(calib_data['MLS'])
    dLS = np.array(calib_data['dLS'])
    MRS = np.array(calib_data['MRS'])
    dRS = np.array(calib_data['dRS'])
    R = np.array(calib_data['R'])
    T = np.array(calib_data['T'])
    RL = np.array(calib_data['RL'])
    RR = np.array(calib_data['RR'])
    PL = np.array(calib_data['PL'])
    PR = np.array(calib_data['PR'])
    Q = np.array(calib_data['Q'])
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file hiệu chỉnh ${json_file}.")
    exit()  

# Hai ma trận hiệu chỉnh cho camera trái và phải
Left_Stereo_Map = cv2.initUndistortRectifyMap(MLS, dLS, RL, PL, (FCAM_WIDTH, FCAM_HEIGHT), cv2.CV_16SC2)
Right_Stereo_Map = cv2.initUndistortRectifyMap(MRS, dRS, RR, PR, (FCAM_WIDTH, FCAM_HEIGHT), cv2.CV_16SC2)


def get_camera(mode="both"):
    """
    mode: "both", "left", "right", "camera"
    - "both": trả về ảnh ghép 2 bên
    - "left": trả về ảnh trái đã hiệu chỉnh
    - "right": trả về ảnh phải đã hiệu chỉnh
    - "camera": trả về tuple (Left_nice, Right_nice)
    """
    global Left_Stereo_Map, Right_Stereo_Map, CamR, CamL

    while True:
        retR, frameR = CamR.read()
        retL, frameL = CamL.read()
        if not retR or not retL:
            time.sleep(0.01)
            continue

        # Hiệu chỉnh ảnh
        Left_nice = cv2.remap(frameL, Left_Stereo_Map[0], Left_Stereo_Map[1],
                              interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)
        Right_nice = cv2.remap(frameR, Right_Stereo_Map[0], Right_Stereo_Map[1],
                               interpolation=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT)

        if mode == "both":
            both = np.hstack((Left_nice, Right_nice))
            for y in range(0, both.shape[0], 40):
                cv2.line(both, (0, y), (both.shape[1], y), (0, 255, 0), 1)
            return both

        elif mode == "left":
            return Left_nice

        elif mode == "right":
            return Right_nice
            # return frameR

        elif mode == "camera":
            return Left_nice, Right_nice

        else:
            print("Tham số mode không hợp lệ! Chọn: 'both', 'left', 'right', 'camera'")
            return None
