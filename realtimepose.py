from ultralytics import YOLO
import cv2

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi kamera
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Kamera tidak dapat dibuka.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Deteksi pose
    results = model(frame)

    # Tampilkan hasil
    for result in results:
        annotated_frame = result.plot()  # Tambahkan anotasi pada frame
        cv2.imshow("YOLOv8 Pose Estimation", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()
