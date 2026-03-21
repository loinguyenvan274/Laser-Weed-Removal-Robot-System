import socket
import threading
import time
import queue
import cv2
# from .setup_wifi import SERVER_IP as _SERVER_IP
from .hieu_chinh_cam import get_camera
# from .yolo_v8 import process_single_image
from .yolov8_final import process_single_image

from .con_arduino import arduino
import requests
import io

#===============================thuat toán sort ===================
import math

# =========================
# TÍNH KHOẢNG CÁCH EUCLID
# =========================
def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


# =========================
# NEAREST NEIGHBOR
# =========================
def nearest_neighbor(points, start_index=0):
    n = len(points)
    visited = [False] * n
    path = []

    current = start_index
    visited[current] = True
    path.append(points[current])

    for _ in range(n - 1):
        nearest = None
        min_dist = float('inf')

        for i in range(n):
            if not visited[i]:
                d = distance(points[current], points[i])
                if d < min_dist:
                    min_dist = d
                    nearest = i

        current = nearest
        visited[current] = True
        path.append(points[current])

    return path


# =========================
# TỔNG ĐỘ DÀI ĐƯỜNG ĐI
# =========================
def total_distance(path):
    total = 0
    for i in range(len(path) - 1):
        total += distance(path[i], path[i + 1])
    return total


# =========================
# 2-OPT (SỬA ĐƯỜNG CHÉO)
# =========================
def two_opt(path):
    improved = True

    while improved:
        improved = False

        for i in range(1, len(path) - 2):
            for j in range(i + 1, len(path)):
                if j - i == 1:
                    continue  # cạnh liền nhau thì bỏ

                new_path = path[:]
                new_path[i:j] = reversed(path[i:j])

                if total_distance(new_path) < total_distance(path):
                    path = new_path
                    improved = True

    return path

# =========================
# HÀM CHÍNH
# =========================
def shortest_path(points, start_index=0):
    if len(points) == 0:
        return [] 
    path = nearest_neighbor(points, start_index)
    path = two_opt(path)
    return path
# =============================================================================================== end thuạt toán

# ===== Cấu hình =====
SERVER_IP = "10.86.222.173"
SERVER_PORT = 5000
SERVER_PORT_HTTP = "8080"
PROJECT_NAME = "/"
API_ENDPOINT = "api/upload-lich-su"
MAY_ID = "JETSON_1"

print(f"===========SERVER IP:{SERVER_IP}===========")
url = f"http://{SERVER_IP}:{SERVER_PORT_HTTP}/{PROJECT_NAME}/{API_ENDPOINT}"

# ===== Biến toàn cục =====
g_stop_event = threading.Event()
g_pause_event = threading.Event()
g_lock = threading.Lock()
g_trang_thai = "NGUNG_HOAT_DONG"
g_task_queue = queue.Queue()


# ===== Hàm hỗ trợ gửi an toàn =====
def safe_send(sock, msg):
    try:
        sock.sendall(msg.encode('utf-8'))
        return True
    except BrokenPipeError:
        print("[safe_send] Lỗi: server đã đóng kết nối.")
    except Exception as e:
        print(f"[safe_send] Lỗi gửi: {e}")
    return False


# ===== Cập nhật trạng thái =====
def cap_nhat_trang_thai(new_status):
    global g_trang_thai
    with g_lock:
        if g_trang_thai != new_status:
            g_trang_thai = new_status
            print(f"[UPDATE-STATUS] Cập nhật trạng thái: {g_trang_thai}")


# ===== Gửi trạng thái định kỳ =====
def gui_trang_thai(sock):
    while not g_stop_event.is_set():
        with g_lock:
            current_status = g_trang_thai
        msg = f"STATUS:{current_status}\n"
        if not safe_send(sock, msg):
            break
        time.sleep(3)


# ===== Gửi kết quả ảnh + dữ liệu lên server =====
def sendResult(ma_phien, vi_tri, result_img, so_co):
    try:
        is_success, buffer = cv2.imencode(".png", result_img)
        if not is_success:
            print("[API-send-to-server] Không thể mã hóa ảnh để gửi!")
            return

        image_bytes = io.BytesIO(buffer)
        data_payload = {
            "ma_dinh_danh": MAY_ID,
            "so_co_phat_hien": so_co,
            "so_co_diet": 7,
            "vi_tri": vi_tri,
            "ma_phien": ma_phien
        }
        files_payload = {
            "anh": ("result.png", image_bytes, "image/png")
        }

        print(f"[API-send-to-server] Gửi POST tới: {url}")
        response = requests.post(url, data=data_payload, files=files_payload, timeout=10)
        print(f"[API-send-to-server] Status: {response.status_code}, Body: {response.text}")

    except requests.exceptions.ConnectionError:
        print(f"[API-send-to-server] Không thể kết nối tới server tại {url}")
    except requests.exceptions.Timeout:
        print("[API-send-to-server] Request timeout")
    except Exception as e:
        print(f"[API-send-to-server] Lỗi: {e}")


# ===== Chạy nhiệm vụ chính =====
def chay_nhiem_vu(sock, ma_phien, quang_duong_muc_tieu):
    print(f"[START] Bắt đầu Phiên {ma_phien}. Mục tiêu: {quang_duong_muc_tieu}m")
    cap_nhat_trang_thai("DANG_HOAT_DONG")
    arduino.reset_connection()

    q_hien_tai = 0
    q_muc_tieu = quang_duong_muc_tieu

    while q_hien_tai < q_muc_tieu:
        if g_stop_event.is_set():
            print(f"Dừng thủ công tại {q_hien_tai}m")
            safe_send(sock, f"STOPPED:{q_hien_tai:.1f}\n")
            arduino.reset_connection()
            break

        if g_pause_event.is_set():
            arduino.guiThongTin(MAY_ID, 'PAUSE')
            cap_nhat_trang_thai("TAM_DUNG")
            print("Tạm dừng...")
            while g_pause_event.is_set() and not g_stop_event.is_set():
                time.sleep(1)
            if not g_stop_event.is_set():
                cap_nhat_trang_thai("DANG_HOAT_DONG")
                print("Tiếp tục nhiệm vụ")

        # ==== Xử lý ảnh ====
        time.sleep(1)
        image = get_camera('left')
        result_img, weed_coords, weed_count = process_single_image(img_input=image)
        cv2.imwrite('./result.jpg', result_img)
        weed_coords = shortest_path(weed_coords)
        print(f"toa do:{weed_coords}")
        # ==== Gửi dữ liệu sang Arduino ====
        arduino.guiThongTin('001', arduino.formatCodeToaDo(weed_coords))
        if arduino.checkHoanTat('101') is None:
            print('[101] Không phản hồi Arduino')

        # ==== Gửi kết quả lên server ====
        sendResult(str(ma_phien), str(q_hien_tai), result_img, weed_count)

        # ==== Di chuyển ====
        quang_duong_di = '20'
        arduino.guiThongTin('002', quang_duong_di)
        if arduino.checkHoanTat('102', 10) is None:
            q_hien_tai = float(arduino.getTinCuoiCung().strip()) / 100.0
            print('[102] Không phản hồi Arduino')
        else:
            q_hien_tai = float(arduino.getTinCuoiCung().strip()) / 100.0
        print(f"[#] Quãng đường hiện tại: {q_hien_tai}m")

    if not g_stop_event.is_set():
        safe_send(sock, f"COMPLETED:{q_hien_tai:.1f}\n")
        print(f"Hoàn thành Phiên {ma_phien}")

    cap_nhat_trang_thai("NGUNG_HOAT_DONG")
    g_stop_event.clear()
    g_pause_event.clear()


# ===== Nhận lệnh từ server =====
def nhan_lenh(sock):
    buffer = ""
    while not g_stop_event.is_set():
        try:
            data = sock.recv(1024).decode('utf-8')
            if not data:
                print("[Nhan-lenh] Server đóng kết nối.")
                g_stop_event.set()
                g_pause_event.set()
                break

            buffer += data
            while '\n' in buffer:
                line, buffer = buffer.split('\n', 1)
                line = line.strip()
                if not line:
                    continue

                print(f"Nhận lệnh: {line}")
                if line.startswith("START:"):
                    try:
                        parts = line.split(':')
                        ma_phien = int(parts[1])
                        quang_duong_muc_tieu = float(parts[2].replace(',', '.'))
                        g_task_queue.put((ma_phien, quang_duong_muc_tieu))
                    except Exception as e:
                        print(f"Lỗi phân tích START: {e}")
                elif line == "STOP":
                    print("Nhận STOP")
                    g_stop_event.set()
                    g_pause_event.clear()
                elif line == "PAUSE":
                    print("Nhận PAUSE")
                    g_pause_event.set()
                    cap_nhat_trang_thai("TAM_DUNG")
                elif line == "RESUME":
                    print("Nhận RESUME")
                    g_pause_event.clear()

        except Exception as e:
            print(f"Lỗi khi nhận lệnh: {e}")
            g_stop_event.set()
            break


# ===== Hàm chính =====
def main():
    while True:
        try:
            print("[Socket] Kết nối tới server...")
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.connect((SERVER_IP, SERVER_PORT))
            print(f"[Socket] Kết nối thành công {SERVER_IP}:{SERVER_PORT}")

            safe_send(sock, f"MAY_ID:{MAY_ID}\n")

            threading.Thread(target=gui_trang_thai, args=(sock,), daemon=True).start()
            threading.Thread(target=nhan_lenh, args=(sock,), daemon=True).start()

            while True:
                try:
                    ma_phien, quang_duong_muc_tieu = g_task_queue.get(timeout=1)
                    g_stop_event.clear()
                    g_pause_event.clear()
                    chay_nhiem_vu(sock, ma_phien, quang_duong_muc_tieu)
                    print("[MAIN] Chờ phiên mới...")
                except queue.Empty:
                    if not safe_send(sock,"CHECK_CONNECT: is oke?"):
                        break
                    continue

        except Exception as e:
            print(f"[Socket] Không kết nối được: {e}")
        finally:
            sock.close()
            g_stop_event.set()
            g_pause_event.set()
            print("[Socket] Thử kết nối lại sau 5s...")
            time.sleep(5)


if __name__ == "__main__":
    main()
