# HỆ THỐNG MÁY DIỆT CỎ BẰNG LASER


##  Tóm tắt dự án

Đồ án tập trung giải quyết vấn đề cấp thiết về **ô nhiễm môi trường do lạm dụng thuốc diệt cỏ** và **thiếu hụt nhân lực** trong nông nghiệp bằng cách phát triển **Robot diệt cỏ tự hành thông minh** hướng tới **nông nghiệp chính xác (Precision Agriculture)**.

Hệ thống sử dụng:
- **Trí tuệ nhân tạo (AI)** + **Thị giác máy tính** trên thiết bị nhúng biên (Jetson Nano).
- Mô hình học sâu **YOLOv8-segmentation** để phát hiện cỏ dại thời gian thực.
- **Camera Stereo** để tính toán độ sâu và xác định tọa độ 3D chính xác.
- **Laser** công suất cao để tiêu diệt cỏ mà không cần thuốc trừ cỏ hóa học.
- **Web Server** (Java Servlet + Apache Tomcat) để giám sát và điều khiển từ xa.

Kết quả thực nghiệm: Robot di chuyển ổn định, nhận diện & xử lý cỏ hiệu quả, giao diện web trực quan, hỗ trợ truy vết lịch sử và thống kê hiệu suất.

---

##  Mục tiêu chính

- Xây dựng xe tự hành di chuyển theo luống, nhận diện cỏ dại thời gian thực trên Jetson Nano.
- Ứng dụng **Stereo Vision** để chuyển đổi tọa độ ảnh 2D → tọa độ không gian 3D (x, y, z) dẫn hướng laser.
- Xây dựng hệ thống **Web giám sát** (Java Servlet + JSP + Tomcat) với giao tiếp lai **Socket TCP** (độ trễ thấp) + **HTTP RESTful API** (truyền dữ liệu lớn).
- Lưu trữ dữ liệu phiên làm việc, lịch sử diệt cỏ và thống kê bằng **MySQL**.

---

## 🛠 Công nghệ & Kiến trúc hệ thống

### 1. Phần cứng (Hardware)
- **Bộ xử lý trung tâm**: NVIDIA Jetson Nano (472 GFLOPS AI, 4GB RAM, GPU 128 CUDA).
- **Vi điều khiển**: Arduino (điều khiển servo Pan-Tilt, động cơ DC, relay laser).
- **Cảm biến**:
  - 2 webcam XWF-1080P (tạo hệ thống Camera Stereo).
  - Cảm biến dò line TCRT5000 (đếm vạch, duy trì quỹ đạo).
  - La bàn số + Encoder.
- **Cơ cấu chấp hành**:
  - Động cơ DC + bánh xe (đường kính 34 cm).
  - Servo Pan-Tilt + Laser công suất cao.
- **Giao tiếp nội bộ**: UART (Jetson ↔ Arduino), USB 3.0/CSI (camera).

**Sơ đồ khối phần cứng** (xem báo cáo trang 14-17).

### 2. Phần mềm & AI
- **Nhận diện cỏ**: YOLOv8-segmentation (train trên 700 ảnh rau bạc hà + cỏ dại).
  - Train: 75% (525 ảnh), Val: 13% (90 ảnh), Test: 12% (85 ảnh).
  - Batch size = 8.
  - Kết quả: mAP@0.5 Box ~85%, Mask ~80%; F1-score cao ở confidence ~0.514–0.515.
- **Stereo Vision**:
  - Hiệu chỉnh camera bằng bàn cờ (OpenCV `stereoCalibrate`).
  - Semi-Global Matching (SGM) để tạo Disparity Map.
  - Công thức tính độ sâu: \( z = \frac{b \cdot f_x}{d} \) (b = baseline, d = disparity).
- **Xử lý tọa độ**: K-means clustering tìm trọng tâm segment → Inverse Kinematics servo → bắn laser.
- **Giao tiếp lai**:
  - **TCP Socket**: Điều khiển realtime (Start/Stop/Pause, heartbeat, trạng thái).
  - **HTTP RESTful API**: Upload hình ảnh bằng chứng + thống kê.

### 3. Web Server & Quản lý
- Apache Tomcat + Java Servlet + JSP.
- Cơ sở dữ liệu MySQL (ERD xem trang 36).
- Chức năng:
  - Danh sách máy, trạng thái kết nối realtime.
  - Lịch sử diệt cỏ theo phiên (ảnh + tọa độ).
  - Biểu đồ thống kê (số lượng cỏ, độ che phủ, số điểm bắn).
  - Streaming video (tương lai).

---

## Kết quả thực nghiệm (tóm tắt)

- **Hiệu suất Jetson Nano**: Chạy mượt mô hình YOLOv8-segmentation realtime.
- **Độ chính xác laser**: Cao (đánh giá chi tiết trang 39).
- **AI model**: Precision/Recall >80%, mAP@0.5 tốt (confusion matrix trang 41).
- **Web**: Giao diện thân thiện, đồng bộ tốt (các hình giao diện trang 36-37).
- **Số điểm bắn laser**: Tính theo diện tích segment + vòng tròn ảnh hưởng laser + K-means (tối ưu năng lượng & hiệu quả diệt cỏ).

**Bảng thống kê chi phí** (xem chi tiết trang 42 báo cáo).

---

## Hướng phát triển tương lai (theo báo cáo)

- **IoT**: Thêm cảm biến môi trường, nâng cấp camera CSI chính hãng, tích hợp 5G/LoRa.
- **Web Server**: Thêm streaming video realtime, dashboard nâng cao, hỗ trợ nhiều robot.
- **Model AI**: Mở rộng dataset (nhiều loại cây trồng/cỏ), tối ưu quantization cho Jetson, thử YOLOv10/v11, thêm tracking (DeepSORT).

---

## Nội dung báo cáo đầy đủ

Báo cáo PDF (`PBL4_bao_cao_2025_cuoi_cung.pdf`) gồm 46 trang với đầy đủ:
- Mục lục & mục lục hình ảnh (trang 5-8).
- Giới thiệu (trang 9-10).
- Giải pháp chi tiết (phần cứng, truyền thông, camera stereo, YOLOv8) (trang 11-33).
- Kết quả thực nghiệm & đánh giá (trang 34-41).
- Bảng chi phí & hướng phát triển (trang 42-44).
- Tài liệu tham khảo (trang 45).

Tất cả hình ảnh, công thức toán học (Stereo Vision, Inverse Kinematics, loss curves, Precision-Recall, F1-confidence, Disparity Map...) đều có trong PDF gốc.

---

## Tài liệu tham khảo
(Chi tiết xem trang 45 của báo cáo)

---

**Liên hệ:**  
- Nguyễn Văn Lợi (thành viên thiết kế phần cứng & IoT)  
- Phan Đình Hồi (Web & Database)  
- Nguyễn Thanh Hậu (AI & Model)

---

**Cảm ơn bạn đã đọc!**  
Dự án chứng minh khả năng thay thế thuốc trừ cỏ hóa học bằng công nghệ Laser + AI, góp phần bảo vệ môi trường và nâng cao năng suất nông nghiệp Việt Nam.