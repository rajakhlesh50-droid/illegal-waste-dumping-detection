# Bill of Materials (BOM) - Waste Detection & Smart Dustbin System
## (हार्डवेयर कंपोनेंट्स की संपूर्ण लागत और सामग्री सूची)

यह दस्तावेज सिस्टम के निर्माण में लगने वाले प्रत्येक हार्डवेयर उपकरण का मॉडल, तकनीकी विनिर्देश, कार्य, अनुशंसित मात्रा और भारतीय रुपयों (INR ₹) में अनुमानित बाजार मूल्य सूचीबद्ध करता है।

---

## 📊 हार्डवेयर कंपोनेंट्स तालिका (Hardware BOM Summary Table)

| क्र.सं. | कंपोनेंट का नाम (Component Name) | मॉडल / पार्ट नंबर (Model) | मात्रा (Qty) | मुख्य कार्य (Primary Function) | कार्यशील वोल्टेज (Voltage) | अनुमानित लागत (Est. Price INR) |
|---|---|---|:---:|---|:---:|:---:|
| **1** | **ESP32 Wi-Fi + Bluetooth IoT Board** | ESP32-WROOM-32 DevKit V1 (30-pin) | 1 | मुख्य IoT माइक्रोकंट्रोलर, सेंसर प्रोसेसिंग और फ्लास्क सर्वर HTTP डेटा सिंक | 3.3V - 5V DC | ₹380 - ₹450 |
| **2** | **Ultrasonic Fill-Level Distance Sensor** | HC-SR04 (या JSN-SR04T Waterproof) | 1 | डस्टबिन के अंदर कचरे की ऊंचाई और भराव प्रतिशत (0-100%) मापना | 5V DC | ₹120 - ₹280 |
| **3** | **Digital Weight Sensor (Load Cell + ADC)** | 10kg/20kg Straight Bar Load Cell + HX711 | 1 | डस्टबिन में जमा कचरे का वास्तविक वजन (Weight in kg) मापना | 5V DC | ₹220 - ₹290 |
| **4** | **Air Quality & Hazardous Gas Sensor** | MQ-135 Gas Sensor Module | 1 | अमोनिया, सल्फाइड और सड़ते कचरे की बदबू / हानिकारक गैस स्तर मापना | 5V DC | ₹130 - ₹180 |
| **5** | **Infrared Obstacle & Approach Sensor** | TCRT5000 IR Sensor Module | 1 | बिना छुए हाथ का पास आना डिटेक्ट करके टचलेस ढक्कन खोलना | 3.3V - 5V DC | ₹60 - ₹90 |
| **6** | **Metal Gear Servo Motor (Auto-Lid)** | TowerPro MG995 / MG90S (180°) | 1 | डस्टबिन के ढक्कन को 90 डिग्री खोलना और कचरा डालने के बाद बंद करना | 4.8V - 6V DC | ₹180 - ₹320 |
| **7** | **High-Decibel Active Alarm Buzzer** | 5V Continuous Active Buzzer (85dB) | 1 | अवैध कचरा फेंकने पर तुरंत स्थानीय तेज बीप अलार्म बजाना | 3.3V - 5V DC | ₹25 - ₹40 |
| **8** | **Voice Alert MP3 Module & Speaker** | DFPlayer Mini + 3W 8Ω Micro Speaker | 1 | हिंदी/अंग्रेजी में स्थानीय आवाज चेतावनी प्रसारित करना | 3.3V - 5V DC | ₹140 - ₹190 |
| **9** | **Graphic OLED Display Module** | 0.96 inch I2C OLED SSD1306 (128x64) | 1 | डस्टबिन पर लाइव भराव प्रतिशत, गैस स्तर और वाई-फाई स्टेटस दिखाना | 3.3V - 5V DC | ₹180 - ₹240 |
| **10** | **Addressable Status Indicator Ring** | WS2812B NeoPixel RGB 8-bit Ring | 1 | रंगीन रोशनी से डस्टबिन का स्टेटस दिखाना (हरा: खाली, लाल: फुल/अलर्ट) | 5V DC | ₹90 - ₹150 |
| **11** | **Wide-Angle Surveillance Web Camera** | 1080p Full HD USB Camera (120° FOV) | 1 | प्राथमिक AI विजन, कचरा पहचान और अवैध डंपिंग ट्रैकिंग | 5V USB | ₹850 - ₹1,400 |
| **12** | **DC-DC Step-Down Buck Converter** | LM2596 3A Adjustable Buck Module | 1 | 12V मेन सप्लाई को सर्वो और बोर्ड के लिए स्थिर 5V 3A में बदलना | 4V - 35V In | ₹70 - ₹110 |
| **13** | **Power Supply Adapter** | 12V 2A DC SMPS Adapter | 1 | संपूर्ण सिस्टम को दीवार से 24/7 बिजली आपूर्ति | 100-240V AC | ₹180 - ₹260 |
| **14** | **Rechargeable Battery Pack + BMS** | 2x 18650 Li-ion (3.7V 2600mAh) + 2S BMS | 1 Set | बिजली जाने पर 8-12 घंटे का अनइंटरप्टेड बैकअप पावर | 7.4V - 8.4V | ₹350 - ₹480 |
| **15** | **Weatherproof Electronic Enclosure** | IP65 ABS Plastic Waterproof Junction Box | 1 | सभी इलेक्ट्रॉनिक्स, सर्किट्स और तारों को धूप और पानी से सुरक्षित रखना | - | ₹220 - ₹350 |
| **16** | **Wires, Breadboard/PCB & Connectors** | Jumper Wires, 40-pin Male/Female, Standoffs | 1 Set | आंतरिक सर्किट कनेक्शन और सेंसर वायरिंग | - | ₹120 - ₹180 |

---

## 💰 कुल परियोजना लागत अनुमान (Total Budget Estimation)

### विकल्प A: बेसिक प्रोटोटाइप किट (Basic Prototype Kit - For Lab & College Demo)
* **शामिल कंपोनेंट्स:** ESP32 + HC-SR04 + SG90 Servo + Buzzer + OLED + USB Camera + 5V Adapter.
* **अनुमानित कुल लागत:** **₹2,100 - ₹2,800 INR**

### विकल्प B: पूर्ण स्मार्ट डस्टबिन किट (Standard Smart IoT Receptacle)
* **शामिल कंपोनेंट्स:** ESP32 + HC-SR04 + Load Cell + MQ-135 + MG995 Servo + Buzzer + DFPlayer Voice + OLED + IP65 Enclosure + 12V Buck Supply.
* **अनुमानित कुल लागत:** **₹3,400 - ₹4,500 INR**

### विकल्प C: सोलर पावर्ड आउटडोर स्टेशन (Solar Off-Grid Station - Complete Autonomous Unit)
* **शामिल कंपोनेंट्स:** Standard Kit + 20W Solar Panel + Solar Charge Controller + 18650 Battery Pack + Night Vision IR Cam.
* **अनुमानित कुल लागत:** **₹5,800 - ₹7,200 INR**

---

## 🛒 खरीद संदर्भ (Procurement & Availability)
सभी कंपोनेंट्स भारतीय इलेक्ट्रॉनिक्स स्टोर्स पर आसानी से उपलब्ध हैं:
1. **Robu.in** (Sensors, ESP32, Servos, Load Cells)
2. **ElectronicsComp.com** (Modules, Wires, Displays, Enclosures)
3. **Amazon.in** (Cameras, Power Adapters, Batteries)

