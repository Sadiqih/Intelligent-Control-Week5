from ultralytics import YOLO
import cv2
import numpy as np
import mediapipe as mp

# Load model YOLOv8 Pose
model = YOLO("yolov8n-pose.pt")

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Inisialisasi kamera
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Kamera tidak dapat dibuka.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal membaca frame dari kamera.")
        break

    # Konversi ke RGB untuk MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Deteksi pose tubuh dengan YOLOv8
    results = model(frame)

    for result in results:
        annotated_frame = frame.copy()  # Gunakan frame asli untuk anotasi

        for person in result.keypoints.xy.numpy():
            if person.shape[0] >= 17:  # Pastikan jumlah keypoints cukup
                keypoints = person

                # Indeks keypoints untuk tangan dan kaki
                LEFT_ELBOW, RIGHT_ELBOW = 7, 8
                LEFT_WRIST, RIGHT_WRIST = 9, 10
                LEFT_KNEE, RIGHT_KNEE = 13, 14
                LEFT_ANKLE, RIGHT_ANKLE = 15, 16

                # Ganti titik (0,0) yang salah dengan NaN
                keypoints = np.where(keypoints > 0, keypoints, np.nan)

                # Fungsi menggambar garis hanya jika kedua titik valid
                def draw_line(p1, p2, color=(0, 255, 0), thickness=2):
                    if p1 is not None and p2 is not None and not np.isnan(p1).any() and not np.isnan(p2).any():
                        cv2.line(annotated_frame, tuple(map(int, p1)), tuple(map(int, p2)), color, thickness)

                # Gambar hanya tangan dan kaki
                draw_line(keypoints[LEFT_ELBOW], keypoints[LEFT_WRIST])
                draw_line(keypoints[RIGHT_ELBOW], keypoints[RIGHT_WRIST])
                draw_line(keypoints[LEFT_KNEE], keypoints[LEFT_ANKLE])
                draw_line(keypoints[RIGHT_KNEE], keypoints[RIGHT_ANKLE])

    # Deteksi tangan dengan MediaPipe
    hand_results = hands.process(rgb_frame)

    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            for idx, landmark in enumerate(hand_landmarks.landmark):
                h, w, _ = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # Gambar titik pada jari-jari tangan
                cv2.circle(annotated_frame, (cx, cy), 5, (0, 0, 255), -1)

            # Gambar garis tangan menggunakan MediaPipe
            mp_draw.draw_landmarks(annotated_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=3),
                                   mp_draw.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2))

    # Tampilkan hasil
    cv2.imshow("YOLOv8 Hand & Feet Detection", annotated_frame)

    # Tekan 'q' untuk keluar
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan resource
cap.release()
cv2.destroyAllWindows()