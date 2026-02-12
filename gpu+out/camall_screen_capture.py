"""
Распознавание по захвату окна (VLC/RDP).
Поиск окна -> захват -> распознавание номера -> вывод в терминал.
При подтверждении номера — событие на карту (прямоугольник появляется).
При выходе поезда — событие на карту (прямоугольник удаляется).
"""
import os
import sys
import json
import asyncio
import time
import re
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)
sys.path.insert(0, SCRIPT_DIR)

import cv2
import numpy as np

try:
    import mss
except ImportError:
    print("[!] Установите mss: pip install mss")
    sys.exit(1)

try:
    import pygetwindow as gw
except ImportError:
    print("[!] Установите pygetwindow: pip install pygetwindow")
    sys.exit(1)

try:
    import websockets
except ImportError:
    print("[!] Установите websockets: pip install websockets")
    sys.exit(1)

from camall import ImageProcessor

CONFIG_SCREEN = "config_screen_capture.json"
SCREENSHOTS_DIR = "recognized_screenshots"

# Клиенты WebSocket для карты (появление/удаление прямоугольника)
_map_clients = set()
_map_rect = None
_map_ws_port = 8765


async def _map_ws_handler(websocket):
    _map_clients.add(websocket)
    try:
        await websocket.wait_closed()
    finally:
        _map_clients.discard(websocket)


async def _map_broadcast(msg):
    if not _map_clients:
        return
    payload = json.dumps(msg, ensure_ascii=False)
    dead = set()
    for ws in _map_clients:
        try:
            await ws.send(payload)
        except Exception:
            dead.add(ws)
    for ws in dead:
        _map_clients.discard(ws)


def save_screenshot_with_number(frame, number):
    """Сохраняет скриншот в папку с номером в имени файла."""
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    safe_number = re.sub(r'[<>:"/\\|?*]', "_", str(number))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(SCREENSHOTS_DIR, f"{safe_number}_{timestamp}.jpg")
    cv2.imwrite(filename, frame)
    return filename


def find_and_prepare_window(window_title):
    """
    Находит окно по названию (частичное совпадение), выводит на передний план
    и разворачивает (maximize). Возвращает (left, top, width, height) или None.
    """
    if not window_title or not window_title.strip():
        print("[!] В конфиге не указан window_title для поиска окна")
        return None

    title_lower = window_title.strip().lower()
    windows = gw.getWindowsWithTitle(window_title)
    if not windows:
        for w in gw.getAllWindows():
            if w.visible and title_lower in (w.title or "").lower():
                windows = [w]
                break

    if not windows:
        print(f"[!] Окно с названием содержащим '{window_title}' не найдено.")
        print("[~] Доступные окна (первые 20):")
        for i, w in enumerate(gw.getAllWindows()):
            if not w.visible or not w.title:
                continue
            print(f"    {i+1}. {w.title[:70]}")
            if i >= 19:
                break
        return None

    win = windows[0]
    print(f"[~] Найдено окно: \"{win.title}\"")

    try:
        win.activate()
        time.sleep(0.3)
    except Exception as e:
        print(f"[!] Не удалось активировать окно: {e}")

    try:
        win.maximize()
        print("[~] Окно развёрнуто (maximize).")
        time.sleep(1.2)
    except Exception as e:
        print(f"[!] Не удалось развернуть окно: {e}")

    try:
        time.sleep(0.2)
        left, top = win.left, win.top
        width, height = win.width, win.height
        if width <= 0 or height <= 0:
            print("[!] Некорректный размер окна после развёртывания")
            return None
        if left < 0 or top < 0 or width < 200 or height < 200:
            print("[!] Окно вернуло нереальные координаты (часто бывает в RDP).")
            print("[~] Задайте область вручную в config_screen_capture.json:")
            print('    "capture": { "type": "region", "region": [0, 0, 1920, 1080] }')
            print("    — замените на свои left, top, width, height (пиксели).")
            return None
        print(f"[~] Область захвата: left={left}, top={top}, width={width}, height={height}")
        return (left, top, width, height)
    except Exception as e:
        print(f"[!] Ошибка при получении размеров окна: {e}")
        return None


def get_primary_screen_region():
    """Возвращает (left, top, width, height) основного (первого) монитора."""
    with mss.mss() as sct:
        # monitors[0] — все экраны, monitors[1] — первый монитор
        mon = sct.monitors[1]
        return (mon["left"], mon["top"], mon["width"], mon["height"])


def capture_region(left, top, width, height):
    """Захват области экрана в BGR (numpy, формат для cv2)."""
    with mss.mss() as sct:
        region = {"left": left, "top": top, "width": width, "height": height}
        screenshot = sct.grab(region)
        img = np.array(screenshot)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)


async def process_frame(frame, processor, executor):
    """
    Весь кадр -> распознавание. Номер подтверждается только если он есть в БД (trains.name).
    Возвращает (confirmed_number или None, saw_any_number: bool).
    saw_any_number = True, если OCR увидел номер и он есть в БД (нужно для логики «вышел»).
    """
    loop = asyncio.get_event_loop()
    num = await loop.run_in_executor(executor, processor.recognize_text_enhanced, frame)

    if num:
        # Для БД срезаем последние 2 цифры (номер вагона), сверяем с trains.name
        train_for_db, wagon = processor.split_train_and_wagon(num)
        if processor.db_config:
            train_id = await loop.run_in_executor(
                executor, processor.find_train_id_by_name, train_for_db
            )
            if train_id is None:
                print(f"[~] Номер «{num}» (поезд {train_for_db}) не найден в БД — продолжаем распознавание")
                processor.recognition_tracker.add_recognition(None)
                return (None, False)
        is_confirmed = processor.recognition_tracker.add_recognition(num)
        return (num if is_confirmed else None, True)
    else:
        processor.recognition_tracker.add_recognition(None)
        return (None, False)


async def main():
    global _map_rect, _map_ws_port
    if not os.path.exists(CONFIG_SCREEN):
        print(f"[!] Файл {CONFIG_SCREEN} не найден.")
        return
    with open(CONFIG_SCREEN, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    capture_cfg = cfg.get("capture", {})
    interval = cfg.get("capture_interval", 4)
    exit_frames = cfg.get("exit_frames_threshold", 10)

    map_cfg = cfg.get("map", {})
    _map_ws_port = map_cfg.get("ws_port", 8765)
    _map_rect = map_cfg.get("rect") or {}

    # Поиск окна или фиксированная область
    capture_type = capture_cfg.get("type")
    window_title = capture_cfg.get("window_title", "VLC") if capture_type == "window" else None
    region = None
    left = top = width = height = 0

    use_primary_fallback = False  # True = окно не нашли, захватываем экран (кадр потом масштабируем для OCR)
    if capture_type == "window":
        region = find_and_prepare_window(window_title)
        if region is not None:
            left, top, width, height = region
        else:
            # Окно не найдено: можно задать в конфиге fallback_region [left, top, width, height],
            # иначе захват всего основного экрана (для OCR кадр уменьшим до разумного размера)
            fallback = capture_cfg.get("fallback_region")
            if isinstance(fallback, list) and len(fallback) == 4:
                left, top, width, height = fallback
                print("[~] Окно не найдено — захват по fallback_region из конфига.")
            else:
                left, top, width, height = get_primary_screen_region()
                use_primary_fallback = True
                print("[~] Окно не найдено — захват основного экрана (кадр будет уменьшен для распознавания).")
            region = (left, top, width, height)
            print(f"[~] Область: left={left}, top={top}, width={width}, height={height}")
    elif capture_type == "region":
        r = capture_cfg.get("region", [])
        if len(r) != 4:
            print("[!] Для type=region укажите region: [left, top, width, height]")
            return
        left, top, width, height = r
        region = (left, top, width, height)
        print(f"[~] Захват по области: {left}, {top}, {width}, {height}")
    else:
        print("[!] В capture укажите type: 'window' или 'region' и window_title или region.")
        return

    db_config = cfg.get("db")
    # Процессор: при наличии db номера подтверждаются только если есть в БД (trains)
    processor = ImageProcessor({
        "name": "screen",
        "depot_id": 0,
        "track_id": 0,
        "clean": None,
        "db": db_config,
        "debug": False,
    })
    executor = ThreadPoolExecutor(max_workers=2)
    loop = asyncio.get_event_loop()

    print(f"[~] Распознавание по всему кадру (номер ищется в любом месте). Интервал: {interval} с.")
    if db_config:
        print(f"[~] Подключение к БД: номера подтверждаются только если найдены в таблице trains.")
    print(f"[~] При подтверждении: поезд (номер) зашел. При {exit_frames} кадрах подряд без номера: поезд (номер) вышел.")
    if _map_rect:
        print(f"[~] Карта: WebSocket порт {_map_ws_port}, прямоугольник по конфигу — при входе/выходе поезда карта обновляется.")
    print()

    # WebSocket-сервер для карты (прямоугольник при входе/выходе поезда)
    async def run_ws_server():
        async with websockets.serve(_map_ws_handler, "0.0.0.0", _map_ws_port, ping_interval=20, ping_timeout=10):
            await asyncio.Future()

    ws_task = asyncio.create_task(run_ws_server())

    last_confirmed_number = None
    consecutive_no_recognition = 0

    # При захвате всего экрана (fallback) уменьшаем кадр для OCR — иначе распознавание хуже
    max_fallback_width = 1920

    while True:
        try:
            frame = await loop.run_in_executor(
                executor, capture_region, left, top, width, height
            )
            if use_primary_fallback and frame is not None and (frame.shape[1] > max_fallback_width or frame.shape[0] > 1080):
                def _resize_for_ocr(img):
                    h, w = img.shape[:2]
                    scale = min(max_fallback_width / w, 1080 / h, 1.0)
                    if scale >= 1.0:
                        return img
                    new_w, new_h = int(w * scale), int(h * scale)
                    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                frame = await loop.run_in_executor(executor, _resize_for_ocr, frame)
            confirmed, saw_number = await process_frame(frame, processor, executor)
            if confirmed is not None:
                train_part, wagon_part = processor.split_train_and_wagon(confirmed)
                if wagon_part is not None:
                    display_msg = f"поезд {train_part}, вагон - {wagon_part}"
                else:
                    display_msg = f"поезд {confirmed}"
                print("распознан номер:", confirmed)
                print(display_msg + " зашел")
                path = await loop.run_in_executor(
                    executor, save_screenshot_with_number, frame, confirmed
                )
                print(f"[~] Скриншот сохранён: {path}")
                last_confirmed_number = confirmed
                consecutive_no_recognition = 0
                if _map_rect:
                    await _map_broadcast({"event": "train_entered", "number": confirmed, "rect": _map_rect})
            elif saw_number:
                consecutive_no_recognition = 0
            else:
                consecutive_no_recognition += 1
                if consecutive_no_recognition >= exit_frames and last_confirmed_number is not None:
                    exited_number = last_confirmed_number
                    train_part, wagon_part = processor.split_train_and_wagon(exited_number)
                    if wagon_part is not None:
                        display_msg = f"поезд {train_part}, вагон - {wagon_part}"
                    else:
                        display_msg = f"поезд {exited_number}"
                    print(display_msg + " вышел")
                    last_confirmed_number = None
                    consecutive_no_recognition = 0
                    processor.recognition_tracker.reset()
                    if _map_rect:
                        await _map_broadcast({"event": "train_exited", "number": exited_number})

            await asyncio.sleep(interval)
        except KeyboardInterrupt:
            print("\n[~] Остановка по Ctrl+C")
            break
        except Exception as e:
            print(f"[!] Ошибка: {e}")
            await asyncio.sleep(interval)

    executor.shutdown(wait=True)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[~] Завершение")
