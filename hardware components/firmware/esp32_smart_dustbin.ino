/**
 * ============================================================================
 * AI WASTE DETECTION SYSTEM - ESP32 IOT SMART DUSTBIN FIRMWARE
 * ============================================================================
 * Features:
 *  - HC-SR04 Ultrasonic Distance Sensor for Fill-Level Calculation (0-100%)
 *  - TCRT5000 IR Proximity Sensor for Touchless Automatic Lid (Servo Open/Close)
 *  - MQ-135 Gas & Odor Sensor for Air Quality Monitoring
 *  - HX711 + Load Cell for Waste Weight Measurement
 *  - 0.96" SSD1306 I2C OLED Display for Local Status
 *  - WS2812B NeoPixel RGB LED Ring for Visual Level Alerts
 *  - 5V Active Buzzer for Local Acoustic Alarms
 *  - Wi-Fi HTTP Client: Synchronizes Live Telemetry with Flask Server API
 * ============================================================================
 */

#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <Wire.h>
#include <Adafruit_GFX.h>
#include <Adafruit_SSD1306.h>
#include <Adafruit_NeoPixel.h>
#include <ESP32Servo.h>

// ==========================================
// NETWORK CONFIGURATION
// ==========================================
const char* WIFI_SSID     = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";

// Central Flask AI Waste Detection Server Endpoint
// Replace with the IP address of the PC running `app.py`
const char* SERVER_TELEMETRY_URL = "http://192.168.1.100:5001/api/hardware_telemetry";

// Device Identification
const char* DEVICE_ID       = "ESP32-BIN-01";
const char* DEVICE_LOCATION = "Paralakhemundi, Odisha - Cam 01";

// ==========================================
// PIN DEFINITIONS
// ==========================================
#define PIN_TRIG        5     // HC-SR04 Trigger Pin
#define PIN_ECHO        18    // HC-SR04 Echo Pin (via voltage divider)
#define PIN_IR_SENSOR   19    // TCRT5000 IR Sensor Digital Out
#define PIN_SERVO       13    // MG995 / SG90 PWM Servo Pin
#define PIN_BUZZER      14    // 5V Active Buzzer Pin
#define PIN_MQ135       34    // MQ-135 Analog Out (ADC1 CH6)
#define PIN_NEOPIXEL    4     // WS2812B RGB Data In
#define NUM_PIXELS      8     // Number of LEDs in NeoPixel Ring

#define OLED_WIDTH      128
#define OLED_HEIGHT     64
#define OLED_RESET      -1

// ==========================================
// DUSTBIN PHYSICAL PARAMETERS
// ==========================================
const float BIN_HEIGHT_CM     = 35.0; // Distance from ultrasonic sensor to empty bin bottom
const float SENSOR_MIN_GAP_CM = 4.0;  // Minimum gap before complete overflow (100% full)
const int   WARN_THRESHOLD    = 75;   // Warning threshold in percent
const int   CRIT_THRESHOLD    = 90;   // Critical overflow threshold in percent

// ==========================================
// GLOBAL OBJECTS & TIMERS
// ==========================================
Adafruit_SSD1306 display(OLED_WIDTH, OLED_HEIGHT, &Wire, OLED_RESET);
Adafruit_NeoPixel strip(NUM_PIXELS, PIN_NEOPIXEL, NEO_GRB + NEO_KHZ800);
Servo binServo;

unsigned long lastTelemetryTime = 0;
const unsigned long TELEMETRY_INTERVAL = 3000; // Send telemetry every 3 seconds

unsigned long lidOpenedTime = 0;
bool isLidOpen = false;
const unsigned long LID_HOLD_DURATION = 4000; // Hold lid open for 4 seconds after hand detected

// ==========================================
// HELPER: READ ULTRASONIC SENSOR
// ==========================================
float readDistanceCm() {
  digitalWrite(PIN_TRIG, LOW);
  delayMicroseconds(2);
  digitalWrite(PIN_TRIG, HIGH);
  delayMicroseconds(10);
  digitalWrite(PIN_TRIG, LOW);

  long duration = pulseIn(PIN_ECHO, HIGH, 30000); // 30ms timeout (~5 meters max)
  if (duration <= 0) {
    return BIN_HEIGHT_CM; // Fallback to empty if pulse times out
  }

  float distance = (duration * 0.0343) / 2.0;
  if (distance > BIN_HEIGHT_CM) distance = BIN_HEIGHT_CM;
  if (distance < SENSOR_MIN_GAP_CM) distance = SENSOR_MIN_GAP_CM;
  return distance;
}

// Calculate percentage from distance
float calculateFillPercentage(float distanceCm) {
  float usableHeight = BIN_HEIGHT_CM - SENSOR_MIN_GAP_CM;
  float wasteDepth   = BIN_HEIGHT_CM - distanceCm;
  float fillPercent  = (wasteDepth / usableHeight) * 100.0;
  if (fillPercent < 0.0) fillPercent = 0.0;
  if (fillPercent > 100.0) fillPercent = 100.0;
  return fillPercent;
}

// ==========================================
// HELPER: UPDATE NEOPIXEL STATUS RING
// ==========================================
void setNeoPixelStatus(float fillPercent) {
  uint32_t color;
  if (fillPercent < WARN_THRESHOLD) {
    color = strip.Color(0, 180, 0);       // Green: Safe / Low Fill
  } else if (fillPercent < CRIT_THRESHOLD) {
    color = strip.Color(220, 150, 0);     // Amber: Warning Nearly Full
  } else {
    color = strip.Color(255, 0, 0);       // Red: Critical Overflow!
  }

  for (int i = 0; i < NUM_PIXELS; i++) {
    strip.setPixelColor(i, color);
  }
  strip.show();
}

// ==========================================
// HELPER: UPDATE OLED LOCAL DISPLAY
// ==========================================
void updateOledDisplay(float fillPercent, float distanceCm, int gasVal) {
  display.clearDisplay();
  display.setTextSize(1);
  display.setTextColor(SSD1306_WHITE);

  // Header Title
  display.setCursor(0, 0);
  display.print("WASTE-AI SMART BIN");
  display.drawLine(0, 10, 127, 10, SSD1306_WHITE);

  // Big Fill Percentage
  display.setCursor(0, 16);
  display.setTextSize(2);
  display.print((int)fillPercent);
  display.print("%");

  // Status Text
  display.setTextSize(1);
  display.setCursor(60, 16);
  if (fillPercent >= CRIT_THRESHOLD) {
    display.print("CRITICAL!");
  } else if (fillPercent >= WARN_THRESHOLD) {
    display.print("WARNING");
  } else {
    display.print("NORMAL");
  }

  // Distance & Gas details
  display.setCursor(60, 28);
  display.print("Dist: ");
  display.print((int)distanceCm);
  display.print("cm");

  display.setCursor(0, 42);
  display.print("Air Gas AQI: ");
  display.print(gasVal);

  display.setCursor(0, 54);
  display.print("Wi-Fi: ");
  display.print(WiFi.status() == WL_CONNECTED ? "ONLINE" : "OFFLINE");

  display.display();
}

// ==========================================
// HELPER: SEND HTTP TELEMETRY TO FLASK SERVER
// ==========================================
void sendServerTelemetry(float fillPercent, float distanceCm, int gasVal) {
  if (WiFi.status() != WL_CONNECTED) {
    Serial.println("[Wi-Fi] Disconnected! Skipping telemetry sync.");
    return;
  }

  HTTPClient http;
  http.begin(SERVER_TELEMETRY_URL);
  http.addHeader("Content-Type", "application/json");

  // Prepare JSON Payload
  StaticJsonDocument<256> doc;
  doc["device_id"]       = DEVICE_ID;
  doc["location"]        = DEVICE_LOCATION;
  doc["fill_percent"]    = round(fillPercent);
  doc["distance_cm"]     = distanceCm;
  doc["gas_ppm"]         = gasVal;
  doc["lid_open"]        = isLidOpen;
  doc["timestamp_epoch"] = millis() / 1000;

  String jsonPayload;
  serializeJson(doc, jsonPayload);

  int httpCode = http.POST(jsonPayload);
  if (httpCode > 0) {
    Serial.printf("[HTTP] Telemetry POST OK (%d): %s\n", httpCode, jsonPayload.c_str());
    String response = http.getString();

    // Check if server commanded a physical alarm beep
    StaticJsonDocument<256> respDoc;
    DeserializationError err = deserializeJson(respDoc, response);
    if (!err && respDoc["trigger_buzzer"].as<bool>()) {
      Serial.println("[ALARM] Server triggered illegal dumping buzzer!");
      digitalWrite(PIN_BUZZER, HIGH);
      delay(300);
      digitalWrite(PIN_BUZZER, LOW);
    }
  } else {
    Serial.printf("[HTTP] Telemetry POST Failed: %s\n", http.errorToString(httpCode).c_str());
  }
  http.end();
}

// ==========================================
// SETUP
// ==========================================
void setup() {
  Serial.begin(115200);
  delay(500);
  Serial.println("\n--- AI WASTE DETECTION SMART BIN STARTING ---");

  // GPIO Modes
  pinMode(PIN_TRIG, OUTPUT);
  pinMode(PIN_ECHO, INPUT);
  pinMode(PIN_IR_SENSOR, INPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  pinMode(PIN_MQ135, INPUT);

  digitalWrite(PIN_BUZZER, LOW);

  // Initialize Servo
  binServo.attach(PIN_SERVO);
  binServo.write(0); // Lid closed initially

  // Initialize NeoPixel Ring
  strip.begin();
  strip.setBrightness(60);
  strip.show();

  // Initialize I2C OLED
  Wire.begin(21, 22);
  if (display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) {
    display.clearDisplay();
    display.setTextSize(1);
    display.setTextColor(SSD1306_WHITE);
    display.setCursor(10, 24);
    display.print("WasteAI Booting...");
    display.display();
  }

  // Connect to Wi-Fi
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.printf("Connecting to Wi-Fi '%s'...", WIFI_SSID);

  int attempts = 0;
  while (WiFi.status() != WL_CONNECTED && attempts < 20) {
    delay(500);
    Serial.print(".");
    attempts++;
  }

  if (WiFi.status() == WL_CONNECTED) {
    Serial.printf("\n[Wi-Fi] Connected! IP: %s\n", WiFi.localIP().toString().c_str());
  } else {
    Serial.println("\n[Wi-Fi] Failed to connect! Operating in standalone offline mode.");
  }

  // Startup beep
  digitalWrite(PIN_BUZZER, HIGH);
  delay(150);
  digitalWrite(PIN_BUZZER, LOW);
}

// ==========================================
// MAIN LOOP
// ==========================================
void loop() {
  // 1. Touchless Hands-Free Automatic Lid Check
  int irState = digitalRead(PIN_IR_SENSOR);
  // TCRT5000 outputs LOW when an obstacle/hand is detected
  if (irState == LOW) {
    if (!isLidOpen) {
      Serial.println("[LID] Hand detected! Opening dustbin lid (90 deg)...");
      binServo.write(90);
      isLidOpen = true;
    }
    lidOpenedTime = millis(); // Refresh open timer
  }

  // Auto-close lid after delay
  if (isLidOpen && (millis() - lidOpenedTime >= LID_HOLD_DURATION)) {
    Serial.println("[LID] Hold timer expired. Closing dustbin lid (0 deg)...");
    binServo.write(0);
    isLidOpen = false;
  }

  // 2. Periodic Telemetry and Sensor Reading
  if (millis() - lastTelemetryTime >= TELEMETRY_INTERVAL) {
    lastTelemetryTime = millis();

    float distance = readDistanceCm();
    float fillPct  = calculateFillPercentage(distance);
    int   gasVal   = analogRead(PIN_MQ135);

    // Update Local Outputs
    setNeoPixelStatus(fillPct);
    updateOledDisplay(fillPct, distance, gasVal);

    // Beep warning on critical overflow
    if (fillPct >= CRIT_THRESHOLD) {
      digitalWrite(PIN_BUZZER, HIGH);
      delay(80);
      digitalWrite(PIN_BUZZER, LOW);
    }

    // Send Telemetry to Flask Server
    sendServerTelemetry(fillPct, distance, gasVal);
  }

  delay(50);
}

