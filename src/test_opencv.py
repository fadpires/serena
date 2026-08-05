import cv2

print("OpenCV version:", cv2.__version__)

# Try opening your Mac's webcam (device 0)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open webcam")
else:
    print("Webcam opened successfully")
    ret, frame = cap.read()
    if ret:
        print("Captured a frame! Shape:", frame.shape)
    cap.release()