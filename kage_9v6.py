import cv2
import torch
import time
import threading
import queue
import numpy as np
import YOLO


# Установка оптимизированного режима вычислений для ускорения на GPU
torch.backends.cudnn.benchmark = True

# Загрузка предварительно обученной модели YOLOv5s
# model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True).to('cuda')
model = YOLO("yolov8n.pt")

# Установка оптимальных параметров модели
model.conf = 0.4  # Порог доверия для обнаружения объектов
model.iou = 0.5  # Порог IoU для подавления дубликатов

# Функция для обработки одного кадра
def process_frame(frame):
    results = model(frame)
    return results

# Функция для отрисовки результатов на кадре
def draw_results(frame, results):
    # Извлечение предсказаний из результатов
    predictions = results.xyxy[0].cpu().numpy()
    for pred in predictions:
        x1, y1, x2, y2, conf, cls = pred
        if int(cls) == 0:  # 0 соответствует классу 'person' в COCO dataset
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            label = f'Person {conf:.2f}'
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

# Функция для чтения каждого второго кадра из видеопотока
def read_frames():
    frame_counter = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_counter % 2 == 0:  # Обрабатываем каждый второй кадр
            frame_queue.put(frame)
        frame_counter += 1
        time.sleep(0.01)  # Для уменьшения нагрузки на процессор

# Функция для обработки кадров из очереди
def process_frames():
    while True:
        frame = frame_queue.get()
        # Изменение размера кадра
        frame_resized = cv2.resize(frame, (640, 640))
        # Конвертируем кадр в тензор и нормализуем его
        frame_resized = torch.from_numpy(frame_resized).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to('cuda')
        # Обработка кадра с помощью модели
        with torch.no_grad():
            results = process_frame(frame_resized)
        # Переводим результаты обратно в формат OpenCV и отображаем их
        frame = draw_results(frame, results)
        output_queue.put(frame)
        frame_queue.task_done()

# Открытие видео файла или веб-камеры
cap = cv2.VideoCapture('kaggleton_9_longer.mp4')  # или 0 для веб-камеры

# Получение размеров окна
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Очередь для кадров
frame_queue = queue.Queue()

# Очередь для обработанных кадров
output_queue = queue.Queue()

# Запуск потока для чтения кадров
reader_thread = threading.Thread(target=read_frames, daemon=True)
reader_thread.start()

# Запуск потоков для обработки кадров
for _ in range(4):  # Запуск 4 потоков для обработки кадров
    processing_thread = threading.Thread(target=process_frames, daemon=True)
    processing_thread.start()

# Инициализация для расчета FPS
prev_time = time.time()
frame_count = 0
fps = 0

while cap.isOpened():
    # Получение обработанного кадра из очереди
    frame = output_queue.get()

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
