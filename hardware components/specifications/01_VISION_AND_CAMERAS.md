# Specifications: Vision & Camera Hardware
## (कैमरा और कंप्यूटर विजन हार्डवेयर विनिर्देश)

यह दस्तावेज़ एआई मॉडल (YOLOv8 + ByteTrack) द्वारा उपयोग किए जाने वाले कैमरों और ऑप्टिकल विजन हार्डवेयर के तकनीकी विनिर्देशों का विवरण देता है।

---

## 📷 1. 1080p Full HD Wide-Angle USB Camera (Primary Edge Cam)
* **सेंसर टाइप:** 1/2.7" CMOS Image Sensor (Sony IMX323 / OV2710)
* **रिज़ॉल्यूशन:** 1920 x 1080 (1080p Full HD @ 30 FPS), 1280 x 720 @ 60 FPS
* **फील्ड ऑफ व्यू (FOV):** 120° Diagonal Ultra-Wide Angle Lens (M12 mount)
* **नाईट विजन:** IR-Cut ऑटोमैटिक स्विच फ़िल्टर + 6x 850nm IR LEDs (कम रोशनी में 5-10 मीटर तक स्पष्ट ब्लैक-एंड-व्हाइट व्यू)
* **इंटरफ़ेस:** USB 2.0 UVC (Plug & Play, no proprietary drivers required on Windows/Linux)
* **फ़्रेम आउटपुट फॉर्मेट:** MJPEG / YUY2
* **बिजली की खपत:** 5V DC @ 250mA (USB पावर्ड)
* **प्रोजेक्ट में उपयोग:** यह कैमरा सर्वर से सीधे USB केबल द्वारा जुड़ा होता है और डस्टबिन के आस-पास कचरा फेंकने वाले लोगों और फर्श पर गिरे कचरे की रीयल-टाइम मॉनिटरिंग करता है।

---

## 🌐 2. Outdoor RTSP IP Bullet Camera (Hikvision / Tapo C310 / Dahua)
* **सेंसर टाइप:** 1/2.8" Progressive Scan CMOS (2MP / 3MP)
* **स्ट्रीम प्रोटोकॉल:** RTSP (Real-Time Streaming Protocol), ONVIF Profile S
* **स्ट्रीम URL फॉर्मेट:** `rtsp://username:password@192.168.1.104:554/live`
* **वेदरप्रूफ रेटिंग:** IP66 / IP67 (धूल, तेज बारिश और -20°C से +60°C तापमान में सक्षम)
* **IR रेंज:** 30 मीटर EXIR Night Vision
* **नेटवर्किंग:** 10/100M Ethernet RJ45 (PoE IEEE 802.3af) या 2.4GHz Wi-Fi (802.11 b/g/n)
* **प्रोजेक्ट में उपयोग:** कैंपस के गेट, बड़े डंपिंग हॉटस्पॉट या मुख्य सड़क पर लगे डस्टबिन की लंबी दूरी से निगरानी।

---

## 🔬 3. ESP32-CAM Micro-Optical Unit (OV2640 2MP)
* **प्रोसेसर:** ESP32-S SoC (Dual-Core 32-bit Tensilica LX6, 160/240MHz)
* **कैमरा सेंसर:** OmniVision OV2640 (1600 x 1200 UXGA @ 15 FPS, SVGA @ 30 FPS)
* **मेमोरी:** 520KB SRAM + 4MB External PSRAM
* **फ्लैश एलईडी:** 1x सुपर-ब्राइट वाइट एलईडी (डस्टबिन के अंदर अंधेरे में फोटो लेने के लिए)
* **प्रोजेक्ट में उपयोग:** डस्टबिन के अंदर ढक्कन पर माउंट होकर आंतरिक कचरे की तस्वीरें सीधे सर्वर पर भेजना।

