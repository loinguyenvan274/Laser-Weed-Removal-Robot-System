#include <Servo.h>

/*==========================================================Start code giao tiếp với Serial===================================================*/
String nhiemVuContents = "";    
String maLenh = "";

void serialEvent() {
  while (Serial.available()) {
    char inChar = (char)Serial.read();
    if (inChar == '\n') {
      maLenh = nhiemVuContents.substring(0, 3);
      nhiemVuContents = nhiemVuContents.substring(4);
    } else {
      nhiemVuContents += inChar;
    }
  }
}


/*==========================================================Start phần test ==============================================================*/
unsigned long t_xuat_gia_tri = 0;
void xemThongTin(int delayt){
  if(millis()> t_xuat_gia_tri + delayt){
    t_xuat_gia_tri = millis(); 
    Serial.print("quan duong da di la: ");
    // Serial.println(doDuong.getQuangDuong()); 
  }
}

/*==========================================================Start Phan chieu Laze======================================================================*/
class PBChieuLaze{
  private:
  const int PAN_PIN = 9;   
  const int TILT_PIN = 10; 

  const int LAZE_PIN = 2;
  const int MIN_ANGLE = 32;
  const int MAX_ANGLE = 140;

  Servo panServo;
  Servo tiltServo;

  public:
  void setup(){
    panServo.attach(PAN_PIN);
    tiltServo.attach(TILT_PIN);
    pinMode(LAZE_PIN, OUTPUT);
    digitalWrite(LAZE_PIN, LOW);
  }

  void  readToaDo(String coord) {
    coord.trim(); 
    if (coord.startsWith("[") && coord.endsWith("]")) {
      coord = coord.substring(1, coord.length() - 1); 
      int commaIndex = coord.indexOf(',');

      int g_x = coord.substring(0, commaIndex).toInt();
      int g_y = coord.substring(commaIndex + 1).toInt();

      chieu(g_x, g_y);
      
      Serial.print("Received: ");
      Serial.print(g_x); Serial.print(", ");
      Serial.println(g_y);
    }
  }

  void thucThiChieu(String nhiemVuContents){
    int start = 0;
    int end = nhiemVuContents.indexOf(':');
    while (end != -1) {
      readToaDo(nhiemVuContents.substring(start, end));
      start = end + 1;
      end = nhiemVuContents.indexOf(':', start);
    }
    readToaDo(nhiemVuContents.substring(start));
    Serial.println("101:da chieu xong");
  }

  void chieu(int g_x,int g_y){
    if((MIN_ANGLE < g_x && g_x < MAX_ANGLE)&& (MIN_ANGLE < g_y && g_y < MAX_ANGLE))
    {
      panServo.write(g_x);
      tiltServo.write(g_y);
      delay(200);
      digitalWrite(LAZE_PIN,HIGH );
      delay(1000);
    }
    digitalWrite(LAZE_PIN, LOW);
  }
  void pause(){
    digitalWrite(LAZE_PIN, LOW);
  }
};

PBChieuLaze pBChieuLaze;


/*=========================================================== Start code do quảng đường ============================================================*/
int sensorValue = 0;      // biến lưu giá trị đọc được
class BPDoDuong {
  private:
    unsigned int quanDuongDaDi;   // Số xung hoặc tổng quãng đường đã đi (mm/cm)
    int sensorPin;                // Chân cảm biến
    int lastState;                // Lưu trạng thái trước đó của cảm biến
    float chuViBanhXe;            // Chu vi bánh xe (đơn vị cm)
    unsigned int soVach;          // Số vạch trắng trên bánh xe (mỗi vòng)
    const int mucNhanBiet = 200;
    int getTinHieu(){
      int currentState = HIGH;
      if(analogRead(sensorPin) > mucNhanBiet){
        currentState = LOW;
      }
      return currentState;
    }
  public:
    // Hàm khởi tạo
    BPDoDuong(int pinCamBien, float chuVi, unsigned int vach) {
      quanDuongDaDi = 0;
      sensorPin = pinCamBien;
      chuViBanhXe = chuVi;
      soVach = vach;
      // pinMode(sensorPin, INPUT);
      lastState = getTinHieu();
    }
    // Cập nhật quãng đường (gọi thường xuyên trong loop)
    void updateQuanDuong() {
      int currentState = getTinHieu();
      // Phát hiện cạnh lên (đen -> trắng)
      if (lastState == LOW && currentState == HIGH) {
        quanDuongDaDi++;
        Serial.println("count:" + String(quanDuongDaDi));
      }
      lastState = currentState;
    }

    // Lấy quãng đường đã đi (đơn vị cm)
    float getQuangDuong() {
      // Mỗi xung tương ứng với 1/soVach vòng
      return (quanDuongDaDi * (chuViBanhXe / soVach));
    }

    // Reset quãng đường
    void reset() {
      quanDuongDaDi = 0;
    }
};

/*============================================================ Di chuyển xe =====================================================================*/

class TaiXe{
  private:
  int pinDongCo;
  BPDoDuong doQD;
  float qDDungXe;
  bool isSeThongBao = false;
  public:
  TaiXe(int pin) : doQD(A0, 34, 5) {
    pinDongCo = pin;
    qDDungXe = doQD.getQuangDuong();
  }
  void setup(){
    pinMode(pinDongCo,OUTPUT);
  }
  void themMucTieu(int kcTienToi){
    digitalWrite(pinDongCo,HIGH);
    qDDungXe = doQD.getQuangDuong() + kcTienToi;
    Serial.println("quan duong them la" + String(qDDungXe));
    isSeThongBao = true;
  }

  void pause(){
    digitalWrite(pinDongCo,LOW);
  }

  void chayXe(){
    doQD.updateQuanDuong();
    if((doQD.getQuangDuong() > qDDungXe) && isSeThongBao){
      digitalWrite(pinDongCo,LOW);
      Serial.println("102:"+String(doQD.getQuangDuong()));
      isSeThongBao = false;
    }
  }
};
TaiXe taiXe = TaiXe(3);

/*============================================================= Chương trình =====================================================================*/
void setup() {
  Serial.begin(9600);
  taiXe.setup();
  pBChieuLaze.setup();
  nhiemVuContents.reserve(300);
}

void loop() {
  if (maLenh != "") {
    if (maLenh.equals("001")) {
      pBChieuLaze.thucThiChieu(nhiemVuContents);
    } 
    else if (maLenh.equals("002")) {
      taiXe.themMucTieu(nhiemVuContents.toInt());
    }
    else if(maLenh.equals("003")){
      taiXe.pause();
    }
    else if(maLenh.equals("004")){
      taiXe.pause();
      pBChieuLaze.pause();
    }
    nhiemVuContents = "";
    maLenh = "";
  }
  taiXe.chayXe();
}

