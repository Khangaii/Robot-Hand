// 실행된 mediapipe 통한 굽힘률을 이용하여 모터 제어하는 UNO 코드
// 제어 값 0일때 100일때 값 수정 필요할 수도 있음. (배열)
  
#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
// 11/24_mediapipe 연동 모터 제어 코드
// Arduino UNO + PCA9685(HW-170)
//
// UNO A4  -> SDA
// UNO A5  -> SCL
// UNO 5V  -> VCC
// UNO GND -> PCA9685 GND
// PCA9685 OE -> GND
//
// 모터는 외부 전원 사용, 전원 GND는 모두 공통 연결.

// 배열 순서: 엄지, 검지, 중지, 약지, 소지
const uint8_t FINGER_COUNT = 5;

const uint8_t SERVO_CHANNELS[FINGER_COUNT] = {
  2, 3, 4, 5, 6
};

const uint8_t PCA9685_ADDRESS = 0x46;

// 제어값 0%일 때의 풀림 위치
const int OPEN_PULSE_US[FINGER_COUNT] = {
  360, 460, 460, 460, 500
};

// 제어값 100%일 때의 감김 위치
const int CLOSED_PULSE_US[FINGER_COUNT] = {
  1200, 1350, 1350, 1350, 1300
};

// 별도의 공통 MIN/MAX 펄스 제한은 없습니다.
// 각 손가락은 위 OPEN/CLOSED 배열 사이에서 움직입니다.

const uint16_t PWM_FREQUENCY_HZ = 50;
const uint32_t PERIOD_US = 1000000UL / PWM_FREQUENCY_HZ;

// 통신이 끊기면 열린 위치로 복귀한 후 PWM을 끕니다.
const uint32_t TIMEOUT_MS = 500;
const uint32_t HOME_TRAVEL_MS = 800;
bool homing = false;
uint32_t homeStartedMs = 0;

Adafruit_PWMServoDriver pwm(PCA9685_ADDRESS);

bool driverReady = false;
bool outputsActive = false;

char line[40];
size_t length = 0;
bool lineTooLong = false;

int lastPulseUs[FINGER_COUNT] = {
  -1, -1, -1, -1, -1
};

uint32_t lastValidMs = 0;


// 모든 서보의 PWM 신호를 끕니다.
void stopAllServos() {
  for (uint8_t i = 0; i < FINGER_COUNT; ++i) {
    if (driverReady) {
      pwm.setPWM(SERVO_CHANNELS[i], 0, 4096);
    }

    lastPulseUs[i] = -1;
  }

  outputsActive = false;
  homing = false;
}


// 한 손가락의 펄스 폭을 설정합니다.
bool writeFingerPulse(uint8_t finger, int pulseUs) {
  if (!driverReady) return false;

  // 별도의 MIN/MAX 제한 없이 계산된 값을 사용합니다.

  // 같은 위치라면 기존 PWM 출력을 유지합니다.
  if (lastPulseUs[finger] == pulseUs) {
    return true;
  }

  const uint16_t ticks =
      (uint16_t)(((uint32_t)pulseUs * 4096UL + PERIOD_US / 2UL)
                 / PERIOD_US);

  if (pwm.setPWM(SERVO_CHANNELS[finger], 0, ticks) != 0) {
    stopAllServos();
    driverReady = false;

    Serial.println(
        "ERROR: PCA9685 I2C write failed. Check wiring, then reset.");

    return false;
  }

  lastPulseUs[finger] = pulseUs;
  return true;
}


void returnHome() {
  if (!driverReady || homing) return;
  for (uint8_t i = 0; i < FINGER_COUNT; ++i) {
    if (!writeFingerPulse(i, OPEN_PULSE_US[i])) return;
  }
  outputsActive = true;
  homing = true;
  homeStartedMs = millis();
}


// PC 또는 D3-G에서 받은 한 줄을 처리합니다.
// 수신 순서: 엄지, 검지, 중지, 약지, 소지
// 예: 0,50,100,50,0
void handleLine(const char *message) {
  if (strcmp(message, "HOME") == 0) {
    returnHome();
    return;
  }
  if (strcmp(message, "STOP") == 0) {
    stopAllServos();
    return;
  }

  int bend[FINGER_COUNT];
  const char *cursor = message;

  // 다섯 값이 모두 0~100의 정상 데이터인지 검사합니다.
  for (uint8_t i = 0; i < FINGER_COUNT; ++i) {
    char *end;
    errno = 0;

    const long value = strtol(cursor, &end, 10);
    const char expected =
        (i == FINGER_COUNT - 1) ? '\0' : ',';

    if (errno == ERANGE ||
        end == cursor ||
        *end != expected ||
        value < 0 ||
        value > 100) {
      return;
    }

    bend[i] = (int)value;

    if (i < FINGER_COUNT - 1) {
      cursor = end + 1;
    }
  }

  // 손가락마다 자신의 OPEN/CLOSED 값으로 선형 변환합니다.
  // 0%   -> 해당 손가락 OPEN
  // 50%  -> 해당 손가락 OPEN/CLOSED의 중앙
  // 100% -> 해당 손가락 CLOSED
  for (uint8_t i = 0; i < FINGER_COUNT; ++i) {
    const long span =
        (long)CLOSED_PULSE_US[i] - OPEN_PULSE_US[i];

    const long scaled = span * bend[i];

    const int pulseUs = OPEN_PULSE_US[i] +
        (int)((scaled >= 0 ? scaled + 50L : scaled - 50L)
              / 100L);

    if (!writeFingerPulse(i, pulseUs)) {
      return;
    }
  }

  outputsActive = true;

  homing = false;
  // 같은 값이 반복 수신되어도 통신 유지 시각은 갱신합니다.
  lastValidMs = (uint32_t)millis();
}


void setup() {
  Serial.begin(115200);
  Wire.begin();

  driverReady = pwm.begin();

  if (!driverReady) {
    Serial.println(
        "ERROR: PCA9685 not found at 0x46. Check VCC/GND/SDA/SCL.");
    return;
  }

  pwm.setPWMFreq(PWM_FREQUENCY_HZ);
  delay(10);

  // 정상 데이터를 받기 전에는 신호를 출력하지 않습니다.
  stopAllServos();

  Serial.println(
      "READY: PCA9685 0x46; channels 2-6; baud 115200.");
}


void loop() {
  if (!driverReady) return;

  for (uint8_t count = 0;
       count < 64 && Serial.available() > 0;
       ++count) {

    const int raw = Serial.read();
    if (raw < 0) break;

    const char c = (char)raw;

    if (c == '\n') {
      if (!lineTooLong) {
        line[length] = '\0';
        handleLine(line);
      }

      length = 0;
      lineTooLong = false;

    } else if (c != '\r' && !lineTooLong) {
      if (length < sizeof(line) - 1) {
        line[length++] = c;
      } else {
        lineTooLong = true;
      }
    }
  }

  // 이동 시간을 확보한 뒤 PWM을 해제합니다.
  const uint32_t elapsed =
      (uint32_t)((uint32_t)millis() - lastValidMs);

  if (homing) {
    if ((uint32_t)(millis() - homeStartedMs) >= HOME_TRAVEL_MS) stopAllServos();
  } else if (outputsActive && elapsed >= TIMEOUT_MS) {
    returnHome();
  }
}
