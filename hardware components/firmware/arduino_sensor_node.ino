/**
 * ============================================================================
 * AI WASTE DETECTION SYSTEM - ARDUINO SENSOR NODE (UNO / NANO)
 * ============================================================================
 * Purpose:
 *  Reads HC-SR04 ultrasonic fill level, TCRT5000 IR touchless hand sensor,
 *  MQ-135 gas sensor, and controls SG90 servo lid + buzzer alarm.
 *  Transmits structured JSON telemetry packets over USB Serial (9600 baud)
 *  to the PC / Raspberry Pi host running `app.py`.
 * ============================================================================
 */

#include <Servo.h>

// PIN ASSIGNMENTS (ARDUINO UNO / NANO)
const int PIN_TRIG      = 9;   // Ultrasonic Trigger
const int PIN_ECHO      = 8;   // Ultrasonic Echo
const int PIN_IR_SENSOR = 7;   // TCRT5000 IR Sensor Digital Out
const int PIN_SERVO     = 6;   // Servo PWM Pin
const int PIN_BUZZER    = 5;   // Active Buzzer
const int PIN_MQ135     = A0;  // MQ-135 Gas Sensor Analog In

// DUSTBIN PHYSICAL MEASUREMENTS
const float BIN_HEIGHT_CM = 35.0; // Distance to empty bin floor

Servo binServo;
unsigned long lastSerialSend = 0;
unsigned long lidOpenTime = 0;
bool isLidOpen = false;

// READ ULTRASONIC DISTANCE
float getDistanceCm() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  long duration = pulseIn(PIN_ECHO, HIGH, 25000);
  if (duration <= 0) return BIN_HEIGHT_CM;
  float d = (duration * 0.0343) / 2.0;
  if (d > BIN_HEIGHT_CM) d = BIN_HEIGHT_CM;
  return d;
}

void setup() {
  Serial.begin(9600);

  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_IR_SENSOR, INPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_MQ135, INPUT);

  digitalWrite(PIN_BUZZER, LOW);

  binServo.attach(PIN_SERVO);
  binServo.write(0); // Lid closed

  // Boot beep
  digitalWrite(PIN_BUZZER, HIGH);
  delay(100);
  digitalWrite(PIN_BUZZER, LOW);
}

void loop() {
  // 1. Touchless Lid Auto Open/Close
  int irState = digitalRead(PIN_IR_SENSOR);
  if (irState == LOW) { // Hand detected
    if (!isLidOpen) {
      binServo.write(90); // Open
      isLidOpen = true;
    }
    lidOpenTime = millis();
  }

  if (isLidOpen && (millis() - lidOpenTime >= 3500)) {
    binServo.write(0); // Close
    isLidOpen = false;
  }

  // 2. Check for incoming serial commands from Python (e.g., ALARM)
  if (Serial.available() > 0) {
    char cmd = Serial.read();
    if (cmd == 'A' || cmd == '1') { // Alarm Trigger from app.py
      digitalWrite(PIN_BUZZER, HIGH);
      delay(300);
      digitalWrite(PIN_BUZZER, LOW);
    }
  }

  // 3. Send JSON Telemetry every 2 seconds
  if (millis() - lastSerialSend >= 2000) {
    lastSerialSend = millis();

    float dist = getDistanceCm();
    float fillPct = ((BIN_HEIGHT_CM - dist) / (BIN_HEIGHT_CM - 4.0)) * 100.0;
    if (fillPct < 0) fillPct = 0;
    if (fillPct > 100) fillPct = 100;

    int gasVal = analogRead(PIN_MQ135);

    // Output JSON to Serial
    Serial.print("{\"source\":\"arduino_node\",\"fill_percent\":");
    Serial.print((int)fillPct);
    Serial.print(",\"distance_cm\":");
    Serial.print(dist, 1);
    Serial.print(",\"gas_raw\":");
    Serial.print(gasVal);
    Serial.print(",\"lid_open\":");
    Serial.print(isLidOpen ? "true" : "false");
    Serial.println("}");
  }

  delay(20);
}

