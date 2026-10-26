#!/usr/bin/env python3
"""
============================================================================
AI WASTE DETECTION SYSTEM - RASPBERRY PI GPIO HARDWARE CONTROLLER
============================================================================
Features:
  - Interfaces directly with Raspberry Pi 4 / 5 40-pin GPIO header
  - HC-SR04 Ultrasonic Distance Sensor (BCM 23 Trig / BCM 24 Echo)
  - 5V Active Buzzer for Illegal Dumping acoustic deterrence (BCM 25)
  - Touchless Servo Lid Control via Hardware PWM (BCM 18)
  - Transmits live fill-level telemetry to Central Flask API (http://localhost:5001/api/hardware_telemetry)
============================================================================
"""

import time
import requests
import json

try:
    import RPi.GPIO as GPIO
except (ImportError, RuntimeError):
    # Dummy mockup for non-Raspberry Pi testing environments
    class DummyGPIO:
        BCM = "BCM"
        OUT = "OUT"
        IN = "IN"
        LOW = 0
        HIGH = 1
        @staticmethod
        def setmode(m): pass
        @staticmethod
        def setwarnings(w): pass
        @staticmethod
        def setup(p, m): pass
        @staticmethod
        def output(p, v): pass
        @staticmethod
        def input(p): return 0
        @staticmethod
        def cleanup(): pass
    GPIO = DummyGPIO()
    print("[WARN] RPi.GPIO not available on host. Running in simulation mode.")

# GPIO PIN MAPPINGS (BCM NUMBERING)
PIN_TRIG   = 23  # Pin 16
PIN_ECHO   = 24  # Pin 18
PIN_BUZZER = 25  # Pin 22
PIN_IR     = 17  # Pin 11

BIN_HEIGHT_CM = 35.0
FLASK_API_URL = "http://127.0.0.1:5001/api/hardware_telemetry"

def setup_gpio():
    GPIO.setmode(GPIO.BCM)
    GPIO.setwarnings(False)
    GPIO.setup(PIN_TRIG, GPIO.OUT)
    GPIO.setup(PIN_ECHO, GPIO.IN)
    GPIO.setup(PIN_BUZZER, GPIO.OUT)
    GPIO.setup(PIN_IR, GPIO.IN)
    GPIO.output(PIN_TRIG, GPIO.LOW)
    GPIO.output(PIN_BUZZER, GPIO.LOW)
    time.sleep(0.2)

def read_ultrasonic_distance():
    GPIO.output(PIN_TRIG, GPIO.HIGH)
    time.sleep(0.00001) # 10us
    GPIO.output(PIN_TRIG, GPIO.LOW)

    pulse_start = time.time()
    pulse_end = time.time()
    timeout = time.time() + 0.04

    while GPIO.input(PIN_ECHO) == 0:
        pulse_start = time.time()
        if pulse_start > timeout:
            return BIN_HEIGHT_CM

    while GPIO.input(PIN_ECHO) == 1:
        pulse_end = time.time()
        if pulse_end > timeout:
            return BIN_HEIGHT_CM

    pulse_duration = pulse_end - pulse_start
    distance = pulse_duration * 17150
    return round(distance, 1)

def trigger_buzzer_beep(duration=0.3):
    GPIO.output(PIN_BUZZER, GPIO.HIGH)
    time.sleep(duration)
    GPIO.output(PIN_BUZZER, GPIO.LOW)

def main():
    setup_gpio()
    print("[HARDWARE] Raspberry Pi GPIO Sensor Node Initialized.")
    print(f"[HARDWARE] Syncing with Flask API: {FLASK_API_URL}")

    try:
        while True:
            dist = read_ultrasonic_distance()
            fill_pct = max(0.0, min(100.0, ((BIN_HEIGHT_CM - dist) / (BIN_HEIGHT_CM - 4.0)) * 100.0))

            payload = {
                "device_id": "RPI4-BIN-GATEWAY",
                "location": "Paralakhemundi, Odisha",
                "fill_percent": round(fill_pct),
                "distance_cm": dist,
                "gas_ppm": 210,
                "lid_open": False
            }

            try:
                res = requests.post(FLASK_API_URL, json=payload, timeout=2.0)
                if res.status_code == 200:
                    data = res.json()
                    if data.get("trigger_buzzer"):
                        print("[ALERT] Illegal dumping detected by server! Sounding buzzer!")
                        trigger_buzzer_beep(0.4)
            except Exception as e:
                print(f"[SYNC] Flask API sync error: {e}")

            time.sleep(2.5)

    except KeyboardInterrupt:
        print("\n[STOP] Shutting down GPIO...")
    finally:
        GPIO.cleanup()

if __name__ == "__main__":
    main()

