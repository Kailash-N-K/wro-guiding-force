from picamera2 import Picamera2
import cv2
import numpy as np
from libcamera import controls, Transform
import time
import pyttsx3
import threading
import random
from collections import defaultdict
import board
import digitalio
import busio
import adafruit_vl53l0x

motor = digitalio.DigitalInOut(board.D27)
motor.direction = digitalio.Direction.OUTPUT

button = digitalio.DigitalInOut(board.D24)
button.direction = digitalio.Direction.INPUT
button.pull = digitalio.Pull.UP

main_led = digitalio.DigitalInOut(board.D23)
main_led.direction = digitalio.Direction.OUTPUT

led = digitalio.DigitalInOut(board.D18)
led.direction = digitalio.Direction.OUTPUT

# === VL53L0X TOF Sensor Setup ===
i2c = busio.I2C(board.SCL, board.SDA)
tof = adafruit_vl53l0x.VL53L0X(i2c)
tof.measurement_timing_budget = 200000  # microseconds

main_led.value = 1

def get_distance():
    distance_mm = tof.range
    if distance_mm is None:
        return None
    distance_m = distance_mm / 1000.0
    if distance_m <= 0 or distance_m > 5:
        return None
    return distance_m

# === Load MobileNet-SSD ===
net = cv2.dnn.readNetFromCaffe('deploy.prototxt', 'mobilenet_iter_73000.caffemodel')

RELEVANT_CLASS_IDS = {
    15: "person",
    2: "bicycle",
    14: "motorbike",
    7: "car",
    6: "bus",
    5: "bottle",
    8: "cat",
    12: "dog",
    9: "chair",
    19: "sofa",
    11: "diningtable",
    20: "tvmonitor"
}

CLASS_GROUPS = {
    "person": "person",
    "bicycle": "vehicle",
    "motorbike": "vehicle",
    "car": "vehicle",
    "bus": "vehicle",
    "cat": "animal",
    "dog": "animal",
    "cow": "animal"
}

OBJECT_PRIORITY = {
    "bottle": 1,
    "tvmonitor": 2,
    "chair": 3,
    "diningtable": 4,
    "person": 5,
    "animal": 6,
    "vehicle": 7
}

# === TTS Setup ===
tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 150)
tts_lock = threading.Lock()
is_speaking = False
SPEAK_INTERVAL = 5

def motor_off():
    motor.value = 0

def speak(text):
    global is_speaking
    def _speak():
        global is_speaking
        with tts_lock:
            is_speaking = True
            tts_engine.say(text)
            tts_engine.runAndWait()
            is_speaking = False
    threading.Thread(target=_speak, daemon=True).start()

# === Camera Setup ===
picam2 = Picamera2()
video_config = picam2.create_video_configuration(main={"size": (640, 480)})
video_config["transform"] = Transform(hflip=1, vflip=1)
picam2.configure(video_config)
picam2.set_controls({"FrameDurationLimits": (33333, 33333)})
picam2.start()

frame_skip = 2
frame_count = 0
last_seen = {}
persistent_detections = defaultdict(lambda: {"box": None, "frames_left": 0})
PERSISTENCE_FRAMES = 5

# === Vibration Timing Setup ===
last_vibration_time = 0
VIBRATION_DURATION = 1.0        # seconds
VIBRATION_COOLDOWN = 2.0        # seconds

# === Main Loop ===
try:
    while True:
        if not button.value:
            led.value = 1
            frame = picam2.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            h, w = frame_bgr.shape[:2]
            display_frame = frame_bgr.copy()

            if frame_count % frame_skip == 0:
                blob = cv2.dnn.blobFromImage(frame_bgr, 0.007843, (300, 300), 20)
                net.setInput(blob)
                detections = net.forward()

                current_labels = set()
                detection_data = {}
                closest_object = None
                largest_area = 0

                for i in range(detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > 0.5:
                        class_id = int(detections[0, 0, i, 1])
                        if class_id not in RELEVANT_CLASS_IDS:
                            continue

                        label_name = RELEVANT_CLASS_IDS[class_id]
                        grouped_label = CLASS_GROUPS.get(label_name, label_name)

                        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                        (startX, startY, endX, endY) = box.astype("int")

                        if endY < h // 3:
                            continue

                        area = (endX - startX) * (endY - startY)
                        if area > largest_area:
                            5.1416
                            largest_area = area
                            closest_object = (grouped_label, (startX, startY, endX, endY))

                if closest_object:
                    grouped_label, box = closest_object
                    now = time.time()
                    cooldown_passed = (
                        grouped_label not in last_seen or
                        now - last_seen[grouped_label]["last_spoken_time"] > SPEAK_INTERVAL
                    )

                    if cooldown_passed and not is_speaking:
                        try:
                            distance = get_distance()
                            if distance is None or distance > 2.0:
                                continue

                            current_time = time.time()

                            if distance <= 0.75 and current_time - last_vibration_time > VIBRATION_COOLDOWN:
                                motor.value = 1
                                last_vibration_time = current_time
                                threading.Timer(VIBRATION_DURATION, motor_off).start()

                            print(f"Detected {grouped_label}, {distance:.2f} meters")
                            time.sleep(random.uniform(0.5, 1.0))
                            speak(f"Detected {grouped_label}, {distance:.2f} meters")

                            last_seen[grouped_label] = {
                                "last_spoken_time": now,
                                "in_frame_last_time": True
                            }

                        except Exception as e:
                            print(f"Distance error: {e}")

                    else:
                        if grouped_label in last_seen:
                            last_seen[grouped_label]["in_frame_last_time"] = True

                    persistent_detections[grouped_label]["box"] = box
                    persistent_detections[grouped_label]["frames_left"] = PERSISTENCE_FRAMES

                for label in last_seen:
                    last_seen[label]["in_frame_last_time"] = label == closest_object[0] if closest_object else False

            for label in list(persistent_detections.keys()):
                if persistent_detections[label]["frames_left"] > 0:
                    box = persistent_detections[label]["box"]
                    startX, startY, endX, endY = box
                    label_text = f"{label}"
                    cv2.rectangle(display_frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                    cv2.putText(display_frame, label_text, (startX, startY - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    persistent_detections[label]["frames_left"] -= 1
                else:
                    del persistent_detections[label]

#              cv2.imshow("Object Detection", display_frame)
#              frame_count += 1
# 
#              if cv2.waitKey(1) & 0xFF == ord('q'):
#                  break
        else:
            led.value=0

finally:
    picam2.stop()
    cv2.destroyAllWindows()
    motor.direction = digitalio.Direction.INPUT
    motor.pull = digitalio.Pull.UP
    button.direction = digitalio.Direction.INPUT
    button.pull = digitalio.Pull.UP
    main_led.direction = digitalio.Direction.INPUT
    main_led.pull = digitalio.Pull.UP
    led.direction = digitalio.Direction.INPUT
    led.direction.pull = digitalio.Pull.UP
