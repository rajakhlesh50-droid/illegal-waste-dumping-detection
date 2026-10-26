# Specifications: Microcontrollers & Edge Compute
## (माइक्रोकंट्रोलर्स और एज कंप्यूटिंग यूनिट्स)

---

## ⚡ 1. ESP32-WROOM-32 DevKit V1 (Primary IoT Node)
* **माइक्रोकंट्रोलर कोर:** Espressif ESP32 32-bit Xtensa Dual-Core LX6 @ 240 MHz
* **वायरलेस कनेक्टिविटी:** 
  - Wi-Fi 802.11 b/g/n (2.4 GHz, 150 Mbps तक)
  - Bluetooth v4.2 BR/EDR & BLE (Bluetooth Low Energy)
* **मेमोरी:** 520 KB SRAM, 4 MB SPI Flash Storage
* **ऑपरेटिंग वोल्टेज:** 3.3V Logic (ऑनबोर्ड AMS1117 रेगुलेटर के साथ 5V VIN इनपुट)
* **GPIO पिन्स:** 30 पिन्स (3x UART, 2x I2C, 3x SPI, 18x 12-bit ADC चैनल्स, 16x PWM)
* **पावर कंज़म्प्शन:** एक्टिव Wi-Fi ट्रांसमिशन पर ~160-240mA; डीप स्लीप में <10µA
* **सिस्टम में भूमिका:** डस्टबिन के सभी सेंसर्स (अल्ट्रासोनिक, गैस, लोड सेल) का डेटा इकट्ठा करना, सर्वो लिड और बजर को नियंत्रित करना तथा वाई-फाई के माध्यम से Flask API को टेलीमेट्री पोस्ट करना।

---

## 🎛️ 2. Arduino Uno R3 / Nano (ATmega328P)
* **माइक्रोकंट्रोलर:** Microchip ATmega328P 8-bit AVR RISC @ 16 MHz
* **ऑपरेटिंग वोल्टेज:** 5V DC
* **मेमोरी:** 32 KB Flash, 2 KB SRAM, 1 KB EEPROM
* **I/O पिन्स:** 14 डिजिटल I/O (6 PWM आउटपुट), 6 एनालॉग इनपुट (10-bit ADC)
* **सिस्टम में भूमिका:** सटीक 5V एनालॉग सेंसिंग (MQ-135, HX711) और USB केबल द्वारा स्थानीय कंप्यूटर पर JSON डेटा भेजने के लिए एक मजबूत और सुरक्षित विकल्प।

---

## 🍓 3. Raspberry Pi 4 Model B (Standalone Edge Hub)
* **प्रोसेसर:** Broadcom BCM2711, Quad-core Cortex-A72 (ARM v8) 64-bit SoC @ 1.8 GHz
* **रैम:** 4 GB / 8 GB LPDDR4-3200 SDRAM
* **कनेक्टिविटी:** Gigabit Ethernet, 2.4/5.0 GHz Dual-Band Wi-Fi, Bluetooth 5.0, 2x USB 3.0, 2x USB 2.0
* **GPIO:** स्टैंडर्ड 40-पिन GPIO हेडर (I2C, SPI, UART, PWM)
* **सिस्टम में भूमिका:** संपूर्ण Flask वेब सर्वर, SQLite डेटाबेस, और कैमरा AI डिटेक्शन को सीधे डस्टबिन पोल पर चलाने के लिए संपूर्ण कंप्यूटर।

