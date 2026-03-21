import requests
import datetime
import time
import os
import cv2 
import io


SERVER_IP = "10.86.222.173"
SERVER_PORT = "8080"
PROJECT_NAME = "may_diet_co"
API_ENDPOINT = "api/upload-lich-su"

url = f"http://{SERVER_IP}:{SERVER_PORT}/{PROJECT_NAME}/{API_ENDPOINT}" 

def sendResult(ma_phien, vi_tri, result_img,so_co):
    """
    Gửi kết quả (ảnh + dữ liệu) lên server.
    Ảnh truyền vào là mảng numpy (kết quả từ AI).
    """
    try:
        # Mã hóa ảnh numpy -> bộ nhớ nhị phân (PNG)
        is_success, buffer = cv2.imencode(".png", result_img)
        if not is_success:
            print("⚠️ Không thể mã hóa ảnh để gửi!")
            return

        image_bytes = io.BytesIO(buffer)

        # Dữ liệu cần gửi
        data_payload = {
            "ma_dinh_danh": "JETSON004",
            "so_co_phat_hien": so_co,
            "so_co_diet": 7,
            "vi_tri": vi_tri,
            "ma_phien": ma_phien
        }


        files_payload = {
            "anh": ("result.png", image_bytes, "image/png")
        }

        print(f"Đang gửi POST request đến: {url}")
        print(f"Dữ liệu: {data_payload}")

        response = requests.post(url, data=data_payload, files=files_payload, timeout=10)

        print("--- KẾT QUẢ TỪ SERVER ---")
        print(f"Status Code: {response.status_code}")
        print(f"Response Body: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"Không thể kết nối tới server tại {url}")
        print("Vui lòng kiểm tra IP và đảm bảo server đang chạy.")
    except requests.exceptions.Timeout:
        print("Request bị timeout (quá 10 giây)")
    except Exception as e:
        print(f"Lỗi khi gửi kết quả: {e}")