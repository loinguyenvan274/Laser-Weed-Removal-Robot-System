import serial
import time
import math

COM_ARDUINO = '/dev/ttyUSB0'
# COM_ARDUINO = '/dev/ttyACM0'
BAUDRATE = 9600

def xoayVeViTri(x_px,y_px,Z):
    h_x =  512.454 #
    h_y = 421.881 #
    
    x_px_o = 471 #
    y_px_o = 396 #

    vt_goc_x_o = 77 #
    vt_goc_y_o = 119 #

    do_lech_x = abs(x_px_o - x_px)
    do_lech_y = abs(y_px_o - y_px)

    h_y_x = math.sqrt(do_lech_x*do_lech_x + h_y*h_y)
    
    goc_x = math.degrees(math.atan(do_lech_x/h_x))
    goc_y = math.degrees(math.atan(do_lech_y/h_y_x))

    

    if x_px > x_px_o:
        goc_x *= -1
    
    if y_px < y_px_o:
        goc_y *= -1

    vi_tri_goc_x = vt_goc_x_o + goc_x
    vi_tri_goc_y = vt_goc_y_o + goc_y
    
    return int(round(vi_tri_goc_x)),int(round(vi_tri_goc_y))

class Arduino:
    def __init__(self, port= COM_ARDUINO, baudrate= BAUDRATE, timeout=1):
        """
        Khởi tạo kết nối UART tới Arduino.
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.tinNhanCuoiCung = ''
        self._connect()
        
    def _connect(self):
        """Hàm nội bộ mở kết nối serial"""
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
        time.sleep(2) 
        print(f"Kết nối thành công với {self.port} ở {self.baudrate} baud.")

    def reset_connection(self):
        """
        Đóng kết nối cũ và mở lại kết nối với Arduino.
        """
        print("Reset kết nối với Arduino...")
        if self.ser and self.ser.is_open:
            self.ser.close()
        time.sleep(1)
        self._connect()
        print("Kết nối đã được reset.")

    def formatCodeToaDo(self, coords, default_z=0):
        """
        Định dạng danh sách tọa độ thành chuỗi tín hiệu.
        Nếu coords chỉ có 2 giá trị [x, y], sẽ tự thêm z = default_z.
        """  
        formatted = []
        for item in coords:
            if len(item) == 2:
                x, y = item
                z = default_z
            elif len(item) >= 3:
                x, y, z = item[:3]
            else:
                raise ValueError(f"Tọa độ không hợp lệ: {item}")
            g_x, g_y = xoayVeViTri(x,y,z)
            formatted.append(f"[{g_x},{g_y}]")
        return formatted
            
    def guiThongTin(self, ma_lenh, parts):
        if isinstance(parts, (list, tuple)):
            message = ma_lenh + ":"+ ":".join(parts) + "\n"
        else:
            message = ma_lenh + ":"+ parts + "\n"
        self.ser.write(message.encode('utf-8'))
        print(f"Đã gửi: {message.strip()}")

    def checkHoanTat(self, maCho, watingTime=10):
        timestamp = time.time()
        while time.time() < timestamp + watingTime:
            line = self.ser.readline().decode('utf-8').strip()
            if line:
                print(f"Nhận: {line}")
                ma_lenh = line[:3]
                if ma_lenh == maCho:
                    print(f"Mã lệnh [{maCho}] được nhận!")
                    self.tinNhanCuoiCung = line[4:]
                    return 'OKE'
                else:
                    print(f" Mã lệnh khác: {ma_lenh}")
        return None

    def getTinCuoiCung(self):
        return self.tinNhanCuoiCung

    def close(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print(" Đã đóng kết nối.")
arduino = Arduino()
