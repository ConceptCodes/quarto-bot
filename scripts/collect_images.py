import cv2
import os
import time
from pathlib import Path

SAVE_DIR = Path("data/collected_images") 
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 0 

print("Starting camera for data collection...")
cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
    print(f"Failed to open camera at index {CAMERA_INDEX}")
    exit(1)

print("Press SPACE to capture and save an image.")
print("Press 'q' to quit.")

img_count = len(list(SAVE_DIR.glob("*.jpg")))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to read frame from camera.")
        break
    cv2.imshow("Data Collection - Press SPACE to capture", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord(" "):
        timestamp = int(time.time() * 1000)
        filename = f"img_{img_count:04d}_{timestamp}.jpg"
        filepath = SAVE_DIR / filename
        cv2.imwrite(str(filepath), frame)
        print(f"Saved: {filepath}")
        img_count += 1
    elif key == ord("q"):
        print("Exiting data collection.")
        break

cap.release()
cv2.destroyAllWindows()
