# Circuit Diagrams & Wiring Schematics
## (सर्किट कनेक्शन और वायरिंग आरेख)

इस दस्तावेज में **AI Waste Detection & Smart Dustbin System** के सभी हार्डवेयर घटकों के पिन-टू-पिन कनेक्शन, ऑपरेटिंग वोल्टेज और सर्किट स्कीमेटिक्स दिए गए हैं।

---

## 🔌 1. ESP32 मास्टर पिन कनेक्शन टेबल (ESP32 Pinout Mapping)

नीचे दी गई तालिका दर्शाती है कि प्रत्येक सेंसर और मॉड्यूल ESP32 डेवलपमेंट बोर्ड (30-pin) के किस पिन से जुड़ेगा:

| कंपोनेंट (Component) | कंपोनेंट पिन (Pin) | ESP32 GPIO पिन | सिग्नल का प्रकार (Signal Type) | विवरण (Description) |
|---|---|---|---|---|
| **HC-SR04 Ultrasonic** | VCC | 5V (VIN) | Power | 5V DC इनपुट |
| | GND | GND | Ground | कॉमन ग्राउंड |
| | TRIG | **GPIO 5** | Digital Output | अल्ट्रासोनिक पल्स ट्रिगर (10µs) |
| | ECHO | **GPIO 18** | Digital Input | पल्स वापसी माप (3.3V Voltage Divider अनुशंसित) |
| **MG995 / SG90 Servo** | Red (VCC) | 5V (Ext/Buck) | Power (5V 1A+) | अलग 5V बक कन्वर्टर से पावर दें |
| | Brown (GND) | GND | Ground | कॉमन ग्राउंड |
| | Orange (PWM) | **GPIO 13** | PWM Output (50Hz) | लिड खोलने (90°) और बंद करने (0°) का सिग्नल |
| **Active Buzzer (5V)** | VCC (+) | **GPIO 14** | Digital Output | HIGH पर बीप अलार्म ऑन |
| | GND (-) | GND | Ground | कॉमन ग्राउंड |
| **TCRT5000 IR Sensor** | VCC | 3.3V / 5V | Power | पावर सप्लाई |
| | GND | GND | Ground | कॉमन ग्राउंड |
| | DO (Digital Out) | **GPIO 19** | Digital Input | हाथ पास आने पर LOW सिग्नल |
| **MQ-135 Gas Sensor** | VCC | 5V | Power | हीटर और सेंसर पावर |
| | GND | GND | Ground | कॉमन ग्राउंड |
| | AO (Analog Out) | **GPIO 34 (ADC1)**| Analog Input | हवा में गैस सघनता (0 - 4095 ADC) |
| **HX711 (Load Cell)** | VCC | 5V | Power | ADC पावर |
| | GND | GND | Ground | कॉमन ग्राउंड |
| | DT (Data) | **GPIO 16** | Digital Input | 24-bit वजन डेटा |
| | SCK (Clock) | **GPIO 17** | Digital Output | क्लॉक पल्स |
| **0.96" I2C OLED** | VDD / VCC | 3.3V | Power | 3.3V पावर |
| | GND | GND | Ground | कॉमन ग्राउंड |
| | SCL | **GPIO 22** | I2C Clock | स्टैंडर्ड ESP32 I2C SCL |
| | SDA | **GPIO 21** | I2C Data | स्टैंडर्ड ESP32 I2C SDA |
| **DFPlayer Mini MP3** | VCC | 5V | Power | 5V पावर |
| | GND | GND | Ground | कॉमन ग्राउंड |
| | RX | **GPIO 26** | UART TX | 1KΩ रेसिस्टर के साथ ESP32 TX $\rightarrow$ DFPlayer RX |
| | TX | **GPIO 27** | UART RX | ESP32 RX $\leftarrow$ DFPlayer TX |
| | SPK1 / SPK2 | Speaker Pins | Audio Out | 3W 8Ω स्पीकर से कनेक्टेड |
| **WS2812B NeoPixel** | 5V | 5V (Ext) | Power | 5V पावर |
| | GND | GND | Ground | कॉमन ग्राउंड |
| | DIN | **GPIO 4** | Digital Data | NeoPixel रंग डेटा स्ट्रीम |

---

## 📐 2. सर्किट स्कीमेटिक डायग्राम (Mermaid Circuit Schematic)

```mermaid
graph TD
    subgraph "POWER SUBSYSTEM"
        Mains["12V 2A DC SMPS Adapter / 18650 Battery"] --> Buck["LM2596 DC-DC Buck (Outputs 5.0V 3A)"]
        Buck -->|5V High Current| Servo_Pwr["MG995 Servo Motor (VCC)"]
        Buck -->|5V DC| ESP_VIN["ESP32 VIN Pin"]
        Buck -->|GND| Common_GND["Common Ground Rail"]
    end

    subgraph "SENSORS MATRIX"
        US["HC-SR04 Ultrasonic (Lid Top)"] -->|GPIO 5 (Trig) / GPIO 18 (Echo)| ESP32
        MQ["MQ-135 Air Quality Sensor"] -->|GPIO 34 (Analog AO)| ESP32
        IR["TCRT5000 Hand Detector"] -->|GPIO 19 (Digital DO)| ESP32
        HX["HX711 + Load Cell (Base Plate)"] -->|GPIO 16 (DT) / GPIO 17 (SCK)| ESP32
    end

    subgraph "ACTUATORS & OUTPUTS"
        ESP32 -->|GPIO 13 (PWM 50Hz)| Servo_Pwr
        ESP32 -->|GPIO 14 (Digital Out)| Buzzer["5V Active Buzzer (85dB)"]
        ESP32 -->|GPIO 22 (SCL) / GPIO 21 (SDA)| OLED["0.96 inch I2C OLED (128x64)"]
        ESP32 -->|GPIO 26 (TX) / GPIO 27 (RX)| DFP["DFPlayer Mini Audio -> 3W Speaker"]
        ESP32 -->|GPIO 4 (NeoPixel Data)| RGB["WS2812B RGB Status Ring"]
    end

    subgraph "EDGE AI & SURVEILLANCE MATRIX"
        CAM["1080p HD Wide-Angle USB / RTSP IP Cam"] -->|Video Feed| FlaskServer["Central Flask Server (app.py)"]
        ESP32 <-->|Wi-Fi 2.4GHz HTTP REST (/api/hardware_telemetry)| FlaskServer
    end
```

---

## ⚡ 3. महत्वपूर्ण वायरिंग सावधानियां (Critical Wiring Notes)

1. **सर्वो मोटर को कभी भी सीधे ESP32 के 5V/VIN पिन से न चलाएं:**
   - MG995 सर्वो लोड पड़ने पर 1A से 1.5A तक करंट खींचता है। यदि आप इसे सीधे ESP32 से जोड़ेंगे, तो बोर्ड बार-बार **Brownout Reset** होकर रीस्टार्ट हो जाएगा।
   - **समाधान:** 12V मेन सप्लाई से **LM2596 बक कन्वर्टर** लगाएं, उसके आउटपुट को 5.0V पर सेट करें और सर्वो की लाल तार (VCC) को सीधे बक कन्वर्टर से जोड़ें। सर्वो की काली तार (GND) को ESP32 और बक कन्वर्टर के कॉमन ग्राउंड से जोड़ें।

2. **HC-SR04 Echo पिन वोल्टेज डिवाइडर (Voltage Divider):**
   - HC-SR04 का Echo पिन 5V आउटपुट देता है, जबकि ESP32 GPIO 3.3V सहने योग्य है।
   - HC-SR04 Echo पिन और ESP32 GPIO 18 के बीच 1kΩ और 2kΩ का वोल्टेज डिवाइडर लगाएं, या फिर 3.3V सपोर्ट करने वाला **HC-SR04P** या **RCWL-1601** मॉडल इस्तेमाल करें।

3. **DFPlayer Mini RX पिन सुरक्षा:**
   - ESP32 TX (GPIO 26) से DFPlayer RX के बीच **1kΩ का रेसिस्टर** लगाना आवश्यक है ताकि ऑडियो प्लेबैक में हमिंग शोर (humming noise) न आए।

4. **MQ-135 प्री-हीटिंग (Preheating):**
   - MQ-135 में आंतरिक हीटर कॉइल होती है। पहली बार सटीक रीडिंग लेने से पहले इसे कम से कम 24-48 घंटे तक पावर देकर 'Burn-in / Pre-heat' करना चाहिए। सामान्य उपयोग में पावर ऑन के 3 मिनट बाद यह सटीक रीडिंग देने लगता है।

