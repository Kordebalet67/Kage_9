import cv2
import time
import torch

# Загрузка предварительно обученной модели YOLOv5
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Функция для обработки одного кадра
def process_frame(frame):
    results = model(frame)
    return results

# Функция для отрисовки результатов на кадре
def draw_results(frame, results):
    for *box, conf, cls in results.xyxy[0]:  # xyxy format
        x1, y1, x2, y2 = map(int, box)
        label = f'{model.names[int(cls)]} {conf:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Открытие видео файла или веб-камеры
cap = cv2.VideoCapture('kaggleton_9_longer.mp4')  # или 0 для веб-камеры - флэшбэк с диплома

# Инициализация для расчета FPS
prev_time = time.time()
frame_count = 0
fps = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Обработка кадра
    results = process_frame(frame)

    # Отрисовка результатов
    frame = draw_results(frame, results)

    # Рассчет и отображение FPS
    frame_count += 1
    curr_time = time.time()
    elapsed_time = curr_time - prev_time
    if elapsed_time > 1:  # Обновление FPS каждую секунду
        fps = frame_count / elapsed_time
        prev_time = curr_time
        frame_count = 0
    
    fps_text = f'FPS: {fps:.2f}'
    cv2.putText(frame, fps_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Отображение кадра
    cv2.imshow('Frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
