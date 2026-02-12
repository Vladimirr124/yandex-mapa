import cv2
import time
from datetime import datetime

def smart_camera_capture(rtsp_url, camera_name, timeout=10, retries=3):
    """
    Улучшенный захват кадра с обработкой ошибок
    """
    for attempt in range(retries):
        try:
            print(f"Попытка {attempt + 1} подключения к {camera_name}...")
            
            cap = cv2.VideoCapture(rtsp_url)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            start_time = time.time()
            
            while (time.time() - start_time) < timeout:
                ret, frame = cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Генерируем имя файла с timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"{camera_name}_{timestamp}.jpg"
                    
                    # Сохраняем с максимальным качеством
                    cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    height, width = frame.shape[:2]
                    print(f"✓ Снимок сохранен: {filename}")
                    print(f"  Разрешение: {width}x{height}")
                    
                    cap.release()
                    return True
                
                time.sleep(0.5)
            
            cap.release()
            print(f"✗ Таймаут подключения к {camera_name}")
            
        except Exception as e:
            print(f"✗ Ошибка при подключении к {camera_name}: {e}")
    
    return False

# Захват с обеих камер
cameras = [
    {"name": "apr5", "url": "rtsp://admin:cppk2020@172.21.123.213"}
]

print("Начинаем захват снимков с камер...")
for camera in cameras:
    smart_camera_capture(camera["url"], camera["name"])
print("Завершено!")