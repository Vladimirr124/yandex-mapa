import os
import cv2
import json
import re
import asyncio
import psycopg2
import numpy as np
from glob import glob
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor
from difflib import get_close_matches
import easyocr
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from collections import deque
import time

os.makedirs("debug_cams", exist_ok=True)
os.makedirs("roi_data", exist_ok=True)  # папка для сохранения ROI
os.makedirs("clean", exist_ok=True)  # папка для эталонных изображений

class ROISelector:
    def __init__(self, image):
        self.image = image
        self.fig, self.ax = plt.subplots(figsize=(12, 8))
        self.ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        self.ax.set_title("Выберите область с номером поезда\nНажмите и перетащите мышью, затем Enter")
        self.rect = None
        self.roi = None
        self.start_x = None
        self.start_y = None
        self.end_x = None
        self.end_y = None
        self.fig.canvas.mpl_connect('button_press_event', self.on_press)
        self.fig.canvas.mpl_connect('button_release_event', self.on_release)
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)

    def on_press(self, event):
        if event.inaxes == self.ax:
            self.start_x = event.xdata
            self.start_y = event.ydata

    def on_release(self, event):
        if event.inaxes == self.ax and self.start_x is not None:
            self.end_x = event.xdata
            self.end_y = event.ydata
            if self.rect:
                self.rect.remove()
            x1, y1 = min(self.start_x, self.end_x), min(self.start_y, self.end_y)
            x2, y2 = max(self.start_x, self.end_x), max(self.start_y, self.end_y)
            self.rect = Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor='red', linewidth=2)
            self.ax.add_patch(self.rect)
            self.fig.canvas.draw()

    def on_key(self, event):
        if event.key == 'enter' and self.start_x is not None and self.end_x is not None:
            x1, y1 = int(min(self.start_x, self.end_x)), int(min(self.start_y, self.end_y))
            x2, y2 = int(max(self.start_x, self.end_x)), int(max(self.start_y, self.end_y))
            self.roi = (x1, y1, x2, y2)
            plt.close(self.fig)
        elif event.key == 'escape':
            self.roi = None
            plt.close(self.fig)

    def select_roi(self):
        plt.tight_layout()
        plt.show()
        return self.roi


class RecognitionTracker:
    """Класс для отслеживания повторяющихся распознаваний"""
    def __init__(self, required_consecutive=3):
        self.required_consecutive = required_consecutive
        self.last_recognition = None
        self.consecutive_count = 0
        self.confirmed_number = None
        
    def add_recognition(self, train_number):
        """Добавляет новое распознавание и возвращает статус подтверждения"""
        if train_number is None:
            # Сбрасываем счетчик при отсутствии распознавания
            self.last_recognition = None
            self.consecutive_count = 0
            return False
            
        if train_number == self.last_recognition:
            self.consecutive_count += 1
        else:
            # Новый номер, сбрасываем счетчик
            self.last_recognition = train_number
            self.consecutive_count = 1
            
        # Проверяем, достигли ли нужного количества повторений
        if self.consecutive_count >= self.required_consecutive:
            if self.confirmed_number != train_number:
                self.confirmed_number = train_number
                return True  # Новый подтвержденный номер
                
        return False  # Номер еще не подтвержден или тот же самый
        
    def get_current_status(self):
        """Возвращает текущий статус распознавания"""
        return {
            'last_recognition': self.last_recognition,
            'consecutive_count': self.consecutive_count,
            'confirmed_number': self.confirmed_number,
            'is_confirmed': self.consecutive_count >= self.required_consecutive
        }
        
    def reset(self):
        """Сбрасывает трекер"""
        self.last_recognition = None
        self.consecutive_count = 0
        self.confirmed_number = None


class ImageProcessor:
    def __init__(self, config):
        self.config = config
        self.reader = easyocr.Reader(['ru','en'], gpu=False)
        self.VALID_PREFIXES = {"ЭП2Д","ЭП2ДМ","ЭГ2Тв","ЭГЭ2Тв","ЭД4М","ЭД4МК","ЭД4Мку","ЭР2Р","ЭД2Т"}
        self.SPECIAL_CASES = {"РА-3","ДП-М"}
        self.THREE_DIGIT_PREFIXES = {"ЭГ2Тв","ЭГЭ2Тв","РА-3","ДП-М"}
        self.FOUR_DIGIT_PREFIXES = {"ЭП2Д"}
        self.OTHER_PREFIXES = {"ЭП2ДМ","ЭД4М","ЭД4МК","ЭД4Мку","ЭР2Р","ЭД2Т"}
        self.image_path = config.get("image_path")
        self.depot_id = config.get("depot_id")
        self.track_id = config.get("track_id")
        self.clean_image_path = config.get("clean")
        self.debug = config.get("debug", False)
        self.save_results = config.get("save_results", True)
        self.interactive_roi = config.get("interactive_roi", True)
        self.db_config = config.get("db")
        
        # Настраиваемый порог сравнения с эталоном (по умолчанию 50%)
        self.clean_match_threshold = config.get("clean_match_threshold", 50.0)
        
        # Трекер распознаваний
        self.recognition_tracker = RecognitionTracker(required_consecutive=3)
        
        # Инициализация БД
        self.init_database()

    def init_database(self):
        """Создает таблицы если они отсутствуют"""
        if not self.db_config:
            return
            
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Создание таблицы train_cam если отсутствует
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS train_cam (
                    id SERIAL PRIMARY KEY,
                    train_number VARCHAR(50),
                    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
                    depot_id INTEGER,
                    track_id INTEGER,
                    depo_event_id INTEGER,
                    recognition_count INTEGER DEFAULT 1,
                    train_id INTEGER
                )
            """)
            
            conn.commit()
            cursor.close()
            conn.close()
            if self.debug:
                print("[DEBUG] Таблицы инициализированы")
        except Exception as e:
            print(f"[!] Ошибка при инициализации БД: {e}")

    # ---------------- Вспомогательные методы ---------------- #
    def get_utc3_time(self):
        return datetime.now(timezone(timedelta(hours=3)))

    def normalize_for_prefix(self, text):
        rules = {'4':'Д','O':'0','1':'И','Z':'2','S':'5','7':'Г','9':'Г','И':'М','Н':'М','К':'И','З':'Э','А':'Д',' ':'-'}
        return ''.join(rules.get(c,c) for c in text).upper()

    def normalize_for_digits(self, text):
        rules = {'O':'0','О':'0','З':'3','Ч':'4','S':'5','Б':'6','Т':'7','Г':'7','В':'8','Д':'4','Z':'2','I':'1','А':'4','У':'4','B':'8','P':'9'}
        return re.sub(r'[^\d]','', ''.join(rules.get(c,c) for c in text))

    def find_best_prefix_match(self, text):
        candidates = list(self.VALID_PREFIXES)+list(self.SPECIAL_CASES)
        left = text.split('-',1)[0]
        norm_left = self.normalize_for_prefix(left)
        if norm_left in candidates: return norm_left
        from difflib import get_close_matches
        matches = get_close_matches(norm_left, candidates, n=1, cutoff=0.6)
        if matches: return matches[0]
        for c in candidates:
            if c in norm_left: return c
        for c in candidates:
            if c in self.normalize_for_prefix(text): return c
        return None

    def extract_digits(self, text):
        right = text.split('-',1)[1] if '-' in text else text
        digits = self.normalize_for_digits(right)
        return digits[-6:] if len(digits)>6 else digits

    def is_valid_train_number(self, digits):
        return len(digits) in {3,4,5,6}

    def extract_valid_combination(self, ocr_blocks):
        PREFIX_DIGIT_RULES = {"ЭП2Д":6,"ЭП2ДМ":6,"ЭД4М":6,"ЭД4МК":6,"ЭД4Мку":6,"ЭР2Р":6,"ЭД2Т":6,
                              "ЭГ2Тв":6,"ЭГЭ2Тв":6,"РА-3":6,"ДП-М":6}
        valid_prefixes = [(self.find_best_prefix_match(t), conf) for _, t, conf in ocr_blocks if self.find_best_prefix_match(t)]
        valid_numbers = [(self.extract_digits(t), conf) for _, t, conf in ocr_blocks if self.is_valid_train_number(self.extract_digits(t))]
        best_comb, best_conf = None, 0
        for p, pc in valid_prefixes:
            for n, nc in valid_numbers:
                expected_len = PREFIX_DIGIT_RULES.get(p,len(n))
                n_trim = n[:min(expected_len, len(n))]
                conf_avg = (pc+nc)/2
                if conf_avg>best_conf:
                    best_comb = f"{p}-{n_trim}"
                    best_conf = conf_avg
        return best_comb

    def _norm_digits(self, s, max_len=6):
        """Убирает пробелы из цифровой части (напр. «057 09» -> «05709») и обрезает до max_len."""
        digits = re.sub(r'[^\d]', '', str(s))
        return digits[:max_len] if digits else s

    def try_direct_patterns(self, ocr_blocks):
        combined = ' '.join([t for _,t,_ in ocr_blocks])
        combined = self.normalize_for_prefix(combined)
        patterns = [r'(ЭГЭ2Тв)-?\s*([\d\s]{3,12})',r'(ЭГ2Тв)-?\s*([\d\s]{3,12})',r'(ЭП2Д)-?(\d{4,6})',
                    r'(ЭП2ДМ)-?(\d{4,6})',r'(ЭД4М)-?(\d{3,6})',r'(ЭД4МК)-?(\d{4,6})',
                    r'(ЭД4Мку)-?(\d{4,6})',r'(ЭР2Р)-?(\d{3,6})',r'(ЭД2Т)-?(\d{3,6})',
                    r'([А-Я]{2,6}\d[А-Я]?[А-Я]?)-?(\d{3,6})']
        for pat in patterns:
            m = re.search(pat, combined, re.IGNORECASE)
            if m:
                if self.find_best_prefix_match(m.group(1)):
                    prefix = m.group(1)
                    digits = self._norm_digits(m.group(2), 6)
                    if len(digits) >= 3:
                        return f"{prefix}-{digits}"
        return None

    def find_train_number_in_any_block(self, ocr_results):
        """
        Ищет номер поезда в любом отдельном блоке OCR (для динамичного видео, где номер может быть в разных местах).
        Игнорирует даты (dd/mm/yyyy), время и прочий текст — приоритет блоку, целиком похожему на номер.
        """
        full_patterns = [
            (r'(ЭГЭ2Тв)-?\s*([\d\s]{3,12})', 6), (r'(ЭГ2Тв)-?\s*([\d\s]{3,12})', 6),
            (r'(ЭП2Д)-?(\d{4,6})', 6), (r'(ЭП2ДМ)-?(\d{4,6})', 6),
            (r'(ЭД4М)-?(\d{3,6})', 6), (r'(ЭД4МК)-?(\d{4,6})', 6),
            (r'(ЭД4Мку)-?(\d{4,6})', 6), (r'(ЭР2Р)-?(\d{3,6})', 6), (r'(ЭД2Т)-?(\d{3,6})', 6),
        ]
        date_like = re.compile(r'\d{1,2}[./]\d{1,2}[./]\d{2,4}|\d{1,2}:\d{2}(:\d{2})?')
        best_num, best_conf = None, 0.0
        for bbox, text, conf in ocr_results:
            if not text or len(text) < 5:
                continue
            if date_like.search(text):
                continue
            norm = self.normalize_for_prefix(text)
            for pat_item in full_patterns:
                pat, max_d = (pat_item[0], pat_item[1]) if isinstance(pat_item, tuple) else (pat_item, 6)
                m = re.search(pat, norm, re.IGNORECASE)
                if m and self.find_best_prefix_match(m.group(1)):
                    prefix = m.group(1)
                    digits = self._norm_digits(m.group(2), max_d) if len(m.groups()) >= 2 else m.group(2)
                    if len(digits) >= 3:
                        num = f"{prefix}-{digits}"
                        if conf > best_conf:
                            best_num, best_conf = num, conf
        return best_num

    def preprocess_enhanced(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)
        blurred = cv2.medianBlur(enhanced,3)
        _, thresh = cv2.threshold(blurred,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        return thresh

    def recognize_text_enhanced(self, image):
        processed = self.preprocess_enhanced(image)
        ocr_results = self.reader.readtext(processed)
        num = self.find_train_number_in_any_block(ocr_results)
        if not num:
            num = self.extract_valid_combination(ocr_results)
        if not num:
            num = self.try_direct_patterns(ocr_results)
        return num

    # ---------------- Управление файлами debug_cams ---------------- #
    def cleanup_old_debug_files(self, cam_name, max_files=5):
        """Удаляет старые файлы в debug_cams, оставляя только max_files самых новых для каждой камеры"""
        try:
            # Ищем файлы для данной камеры
            pattern = f"debug_cams/{cam_name}_*.jpg"
            files = glob(pattern)
            
            # Если файлов больше максимального количества, удаляем самые старые
            if len(files) > max_files:
                # Сортируем файлы по времени создания (сначала самые старые)
                files.sort(key=os.path.getctime)
                
                # Удаляем самые старые файлы
                for file_to_remove in files[:-max_files]:
                    os.remove(file_to_remove)
                    if self.debug:
                        print(f"[DEBUG] Удален старый файл: {file_to_remove}")
                
                if self.debug:
                    print(f"[DEBUG] Для камеры {cam_name} оставлено {max_files} файлов из {len(files)}")
                    
        except Exception as e:
            print(f"[!] Ошибка при очистке старых файлов для камеры {cam_name}: {e}")

    # ---------------- Проверка вместимости и позиционирование ---------------- #
    def check_parking_capacity(self, train_id, parking_id):
        """Проверяет, достаточно ли места на пути для поезда"""
        if not self.db_config:
            return False
            
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Получаем вместимость пути
            cursor.execute(
                "SELECT capacity FROM parkings WHERE id = %s",
                (parking_id,)
            )
            capacity_result = cursor.fetchone()
            if not capacity_result:
                print(f"[!] Путь {parking_id} не найден в таблице parkings")
                return False
            
            capacity = capacity_result[0]
            if self.debug:
                print(f"[DEBUG] Вместимость пути {parking_id}: {capacity}")
            
            # Получаем количество вагонов нового поезда
            cursor.execute(
                "SELECT COUNT(*) FROM carriages WHERE train_id = %s",
                (train_id,)
            )
            new_train_carriages = cursor.fetchone()[0]
            if self.debug:
                print(f"[DEBUG] Вагоны нового поезда {train_id}: {new_train_carriages}")
            
            # Получаем количество занятых вагонов на пути
            cursor.execute(
                """
                SELECT de.train_id, COUNT(DISTINCT c.id) as carriage_count
                FROM depo_events de
                JOIN carriages c ON c.train_id = de.train_id
                WHERE de.parking_id = %s AND de.out_time IS NULL 
                AND de.depo_event_action = 2 AND de.train_id IS NOT NULL
                GROUP BY de.train_id
                """,
                (parking_id,)
            )
            occupied_results = cursor.fetchall()
            occupied_carriages = sum(row[1] for row in occupied_results)
            
            if self.debug:
                print(f"[DEBUG] Занятые вагоны на пути {parking_id}: {occupied_carriages}")
                print(f"[DEBUG] Детали занятых поездов: {occupied_results}")
            
            free_space = capacity - occupied_carriages
            
            if free_space >= new_train_carriages:
                if self.debug:
                    print(f"[DEBUG] Достаточно места: свободно {free_space}, нужно {new_train_carriages}")
                return True
            else:
                print(f"[!] Недостаточно места: свободно {free_space}, нужно {new_train_carriages}")
                return False
                
        except Exception as e:
            print(f"[!] Ошибка проверки вместимости: {e}")
            return False
        finally:
            if 'cursor' in locals():
                cursor.close()
            if 'conn' in locals():
                conn.close()

    def get_train_carriages_count(self, train_id):
        """Получить количество вагонов поезда"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT COUNT(*) FROM carriages WHERE train_id = %s",
                (train_id,)
            )
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return result[0] if result else 0
            
        except Exception as e:
            print(f"[!] Ошибка получения количества вагонов: {e}")
            return 0

    def get_next_train_position(self, parking_id):
        """Получить позицию для следующего поезда на пути (с учетом всех поездов)"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Находим ВСЕ поезда на пути через depo_events
            cursor.execute(
                """
                SELECT de.train_id, de.in_time
                FROM depo_events de
                WHERE de.parking_id = %s AND de.out_time IS NULL 
                AND de.depo_event_action = 2 AND de.train_id IS NOT NULL
                ORDER BY de.in_time ASC
                """,
                (parking_id,)
            )
            all_trains = cursor.fetchall()
            
            if not all_trains:
                if self.debug:
                    print(f"[DEBUG] На пути {parking_id} нет поездов, позиционирование не требуется")
                return None
            
            if self.debug:
                print(f"[DEBUG] Найдено поездов на пути {parking_id}: {len(all_trains)}")
            
            # Находим минимальную позицию и считаем общую длину всех поездов
            min_position = None
            total_length = 0
            
            for train_id, in_time in all_trains:
                # Получаем позицию поезда
                cursor.execute(
                    "SELECT point_x FROM sheme_positions WHERE train_id = %s ORDER BY id DESC LIMIT 1",
                    (train_id,)
                )
                position_result = cursor.fetchone()
                
                if position_result:
                    position_x = position_result[0]
                    # Получаем количество вагонов поезда
                    carriages_count = self.get_train_carriages_count(train_id)
                    train_length = carriages_count * 65  # 65 метров на вагон
                    
                    if self.debug:
                        print(f"[DEBUG] Поезд {train_id}: позиция {position_x}, вагонов {carriages_count}, длина {train_length}")
                    
                    # Находим минимальную позицию
                    if min_position is None or position_x < min_position:
                        min_position = position_x
                    
                    # Суммируем длину всех поездов
                    total_length += train_length
                else:
                    if self.debug:
                        print(f"[DEBUG] Позиция поезда {train_id} не найдена в sheme_positions")
            
            if min_position is None:
                if self.debug:
                    print(f"[DEBUG] Не найдено ни одной позиции поездов")
                return None
            
            # Следующий поезд ставим в позицию: минимальная + общая длина
            next_position = min_position + total_length
            if self.debug:
                print(f"[DEBUG] Минимальная позиция: {min_position}, общая длина: {total_length}")
                print(f"[DEBUG] Следующий поезд будет в позиции: {next_position}")
            
            cursor.close()
            conn.close()
            
            return next_position
            
        except Exception as e:
            print(f"[!] Ошибка получения позиции для следующего поезда: {e}")
            return None

    def set_train_position(self, train_id, parking_id, position_x):
        """Установить позицию поезда в sheme_positions"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Проверяем, есть ли запись для этого поезда
            cursor.execute(
                "SELECT id FROM sheme_positions WHERE train_id = %s",
                (train_id,)
            )
            existing = cursor.fetchone()
            
            if existing:
                # Обновляем существующую запись
                cursor.execute(
                    "UPDATE sheme_positions SET point_x = %s WHERE train_id = %s",
                    (position_x, train_id)
                )
                if self.debug:
                    print(f"[DEBUG] Обновлена позиция поезда {train_id}: {position_x}")
            else:
                # Создаем новую запись с обязательными полями
                cursor.execute(
                    "INSERT INTO sheme_positions (stantion_id, train_id, point_x) VALUES (%s, %s, %s)",
                    (self.depot_id, train_id, position_x)
                )
                if self.debug:
                    print(f"[DEBUG] Создана позиция поезда {train_id}: {position_x}")
            
            conn.commit()
            cursor.close()
            conn.close()
            
            return True
            
        except Exception as e:
            print(f"[!] Ошибка установки позиции поезда: {e}")
            return False

    # ---------------- Поиск train_id в таблице trains ---------------- #
    def split_train_and_wagon(self, train_number):
        """Последние 2 цифры — номер вагона; для сопоставления с БД срезаем их.
        Возвращает (номер_поезда_для_БД, последние_2_цифры или None)."""
        if not train_number or "-" not in str(train_number):
            return (train_number, None)
        parts = train_number.split("-", 1)
        prefix = parts[0]
        right = parts[1]
        digits = re.sub(r"[^\d]", "", right)
        if len(digits) >= 2:
            train_part = prefix + "-" + digits[:-2]
            wagon = digits[-2:]
            return (train_part, wagon)
        return (train_number, None)

    def find_train_id_by_name(self, train_number):
        """Ищет train_id в таблице trains по полю name"""
        if not self.db_config:
            return None
            
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            # Ищем точное совпадение
            cursor.execute(
                "SELECT id FROM trains WHERE name = %s",
                (train_number,)
            )
            result = cursor.fetchone()
            
            if result:
                train_id = result[0]
                cursor.close()
                conn.close()
                return train_id
                
            # Если точное совпадение не найдено, ищем похожие варианты
            cursor.execute(
                "SELECT id, name FROM trains WHERE name LIKE %s",
                (f"%{train_number}%",)
            )
            similar_results = cursor.fetchall()
            
            cursor.close()
            conn.close()
            
            if similar_results:
                # Логируем найденные похожие варианты
                print(f"[~] Точное совпадение для '{train_number}' не найдено. Похожие варианты: {[r[1] for r in similar_results]}")
            else:
                print(f"[!] Состав '{train_number}' не найден в таблице trains")
                
            return None
            
        except Exception as e:
            print(f"[!] Ошибка при поиске train_id: {e}")
            return None

    def check_active_depo_event(self, train_id, parking_id):
        """Проверяет, есть ли активное событие для поезда на этом пути"""
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT id FROM depo_events WHERE train_id = %s AND parking_id = %s AND out_time IS NULL LIMIT 1",
                (train_id, parking_id)
            )
            result = cursor.fetchone()
            cursor.close()
            conn.close()
            
            return result
            
        except Exception as e:
            print(f"[!] Ошибка при проверке активных событий: {e}")
            return None

    # ---------------- Сравнение с эталоном ---------------- #
    def compare_with_clean(self, frame):
        """Сравнивает текущий кадр с эталоном пустого депо"""
        if not self.clean_image_path or not os.path.exists(self.clean_image_path):
            if self.debug:
                print(f"[DEBUG] Эталонное изображение не найдено: {self.clean_image_path}")
            return False
            
        try:
            clean_img = cv2.imread(self.clean_image_path)
            if clean_img is None:
                print(f"[!] Не удалось загрузить эталонное изображение: {self.clean_image_path}")
                return False
                
            # Приводим к одинаковому размеру
            h, w = frame.shape[:2]
            clean_img = cv2.resize(clean_img, (w, h))
            
            # Конвертируем в grayscale
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            clean_gray = cv2.cvtColor(clean_img, cv2.COLOR_BGR2GRAY)
            
            # Вычисляем разницу
            diff = cv2.absdiff(frame_gray, clean_gray)
            _, diff_thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
            
            # Вычисляем процент различий
            diff_percent = np.sum(diff_thresh) / (255.0 * diff_thresh.size) * 100
            
            if self.debug:
                print(f"[DEBUG] Различие с эталоном: {diff_percent:.2f}%")
                
            # ИНВЕРТИРОВАННАЯ ЛОГИКА: 
            # Если различия МЕНЬШЕ порога - депо пустое (мало изменений)
            # Если различия БОЛЬШЕ порога - есть поезд (много изменений)
            return diff_percent < self.clean_match_threshold
            
        except Exception as e:
            print(f"[!] Ошибка при сравнении с эталоном: {e}")
            return False

    # ---------------- Работа с базой ---------------- #
    def save_to_database(self, train_number):
        """Сохраняет номер поезда и создает запись в depo_events с проверкой вместимости"""
        if not self.db_config: 
            return None
            
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            timestamp = self.get_utc3_time()
            
            # Ищем train_id в таблице trains
            train_id = self.find_train_id_by_name(train_number)
            
            if train_id is None:
                print(f"[!] Не удалось найти состав '{train_number}' в таблице trains. Запись не будет создана.")
                return None
            
            # Проверяем, не стоит ли уже поезд на этом пути
            active_event = self.check_active_depo_event(train_id, self.track_id)
            if active_event:
                print(f"[~] Поезд {train_number} (ID {train_id}) уже на пути {self.track_id}")
                return active_event[0]
            
            # Проверяем вместимость пути
            if not self.check_parking_capacity(train_id, self.track_id):
                print(f"[!] Поезд {train_number} не может быть размещен - недостаточно места на пути {self.track_id}")
                return None
            
            # Вставляем запись в train_cam
            cursor.execute(
                """INSERT INTO train_cam(depot_id, track_id, train_number, detected_at, 
                recognition_count, train_id) VALUES (%s,%s,%s,%s,%s,%s) RETURNING id""",
                (self.depot_id, self.track_id, train_number, timestamp, 
                 self.recognition_tracker.consecutive_count, train_id)
            )
            train_cam_id = cursor.fetchone()[0]
            
            # Закрываем предыдущие события для этого поезда на других путях
            cursor.execute(
                "UPDATE depo_events SET out_time = %s WHERE train_id = %s AND parking_id != %s AND out_time IS NULL",
                (timestamp, train_id, self.track_id)
            )
            
            # Создаем новую запись в depo_events
            cursor.execute("""
                INSERT INTO depo_events (parking_id, train_id, is_active, depo_event_action, 
                                       date, in_time, user_id, user_change_id, update_date)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
            """, (self.track_id, train_id, True, 2, timestamp, timestamp, 777, 777, timestamp))
            
            depo_event_id = cursor.fetchone()[0]
            
            # Обновляем train_cam с ссылкой на depo_event_id
            cursor.execute(
                "UPDATE train_cam SET depo_event_id = %s WHERE id = %s",
                (depo_event_id, train_cam_id)
            )
            
            conn.commit()
            cursor.close()
            conn.close()
            
            print(f"[✓] Создана новая запись в depo_events для состава {train_number} (train_id: {train_id})")
            
            # Устанавливаем позицию поезда с задержкой
            print(f"[DEBUG] Ждем 2 секунды перед установкой позиции...")
            time.sleep(2)
            
            position_x = self.get_next_train_position(self.track_id)
            if position_x is not None:
                print(f"[DEBUG] Устанавливаем позицию для поезда {train_number}: {position_x}")
                if self.set_train_position(train_id, self.track_id, position_x):
                    print(f"[✓] Позиция поезда {train_number} установлена: {position_x}")
                else:
                    print(f"[!] Не удалось установить позицию для поезда {train_number}")
            else:
                print(f"[DEBUG] Позиционирование для поезда {train_number} не требуется (первый поезд в депо)")
            
            if self.debug:
                print(f"[DEBUG] Сохранено в БД: {train_number}, train_id: {train_id}, depo_event_id: {depo_event_id}")
                
            return depo_event_id
            
        except Exception as e:
            print(f"[!] Ошибка при записи в БД: {e}")
            return None

    def update_depo_event_out_time(self, depo_event_id):
        """Устанавливает out_time для записи в depo_events"""
        if not self.db_config:
            return
            
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            timestamp = self.get_utc3_time()
            
            # Проверяем, не установлен ли уже out_time
            cursor.execute(
                "SELECT out_time FROM depo_events WHERE id = %s",
                (depo_event_id,)
            )
            result = cursor.fetchone()
            
            if result and result[0] is None:
                # Устанавливаем out_time
                cursor.execute(
                    "UPDATE depo_events SET out_time = %s, is_active = false WHERE id = %s",
                    (timestamp, depo_event_id)
                )
                conn.commit()
                if self.debug:
                    print(f"[DEBUG] Установлен out_time для depo_event_id: {depo_event_id}")
            else:
                if self.debug:
                    print(f"[DEBUG] out_time уже установлен для depo_event_id: {depo_event_id}")
                    
            cursor.close()
            conn.close()
            
        except Exception as e:
            print(f"[!] Ошибка при обновлении depo_events: {e}")

    def get_active_depo_events(self):
        """Возвращает активные depo_event_id для текущего depot_id и track_id"""
        if not self.db_config:
            return []
            
        try:
            conn = psycopg2.connect(**self.db_config)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT de.id, t.name 
                FROM depo_events de
                JOIN trains t ON de.train_id = t.id
                WHERE de.out_time IS NULL 
                AND de.parking_id = %s
            """, (self.track_id,))
            
            active_events = cursor.fetchall()
            cursor.close()
            conn.close()
            
            return active_events
            
        except Exception as e:
            print(f"[!] Ошибка при получении активных событий: {e}")
            return []

    # ---------------- ROI ---------------- #
    def select_roi_interactive(self, frame, cam_name):
        roi_file = f"roi_data/roi_{cam_name}.json"
        if os.path.exists(roi_file):
            with open(roi_file,"r",encoding="utf-8") as f:
                roi = json.load(f)
            return roi
        selector = ROISelector(frame)
        roi = selector.select_roi()
        if roi:
            x1, y1, x2, y2 = roi
            roi_dict = {"x1": x1,"y1": y1,"x2": x2,"y2": y2}
            with open(roi_file,"w",encoding="utf-8") as f:
                json.dump(roi_dict,f,ensure_ascii=False, indent=2)
            return roi_dict
        return None

# ---------------- Асинхронная обработка камер ---------------- #
def ensure_roi(cam_conf, processor):
    if "roi" not in cam_conf or not cam_conf["roi"]:
        cap = cv2.VideoCapture(cam_conf["rtsp_url"])
        ret, frame = cap.read()
        cap.release()
        if not ret: return None
        
        # Выводим разрешение снимка
        height, width = frame.shape[:2]
        print(f"[~] Разрешение снимка с камеры {cam_conf['name']}: {width}x{height}")
        
        print(f"[~] Выберите ROI для камеры {cam_conf['name']}")
        roi = processor.select_roi_interactive(frame, cam_conf["name"])
        if roi:
            cam_conf["roi"] = roi
        else: return None
    return cam_conf["roi"]

async def process_camera_frame(frame, cam_conf, processor, executor):
    roi = cam_conf.get("roi")
    if not roi: return None
    x1, y1, x2, y2 = roi["x1"], roi["y1"], roi["x2"], roi["y2"]
    cropped_frame = frame[y1:y2, x1:x2]
    
    # Используем переданный executor для блокирующих операций OCR
    loop = asyncio.get_event_loop()
    num = await loop.run_in_executor(executor, processor.recognize_text_enhanced, cropped_frame)
    
    if num:
        # Добавляем распознавание в трекер
        is_confirmed = processor.recognition_tracker.add_recognition(num)
        status = processor.recognition_tracker.get_current_status()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"debug_cams/{cam_conf['name']}_{num}_{timestamp}_count{status['consecutive_count']}.jpg"
        
        # Сохраняем изображение асинхронно
        await loop.run_in_executor(executor, cv2.imwrite, filename, cropped_frame)
        
        # Очищаем старые файлы для этой камеры
        await loop.run_in_executor(executor, processor.cleanup_old_debug_files, cam_conf['name'], 5)
        
        if is_confirmed:
            # Номер подтвержден 3 раза подряд - сохраняем в БД
            depo_event_id = await loop.run_in_executor(executor, processor.save_to_database, num)
            if depo_event_id:
                print(f"[✓] Камера {cam_conf['name']} -> {num} (подтверждено {status['consecutive_count']} раз, запись создана)")
            else:
                print(f"[!] Камера {cam_conf['name']} -> {num} (подтверждено {status['consecutive_count']} раз, но запись НЕ создана)")
        else:
            # Номер распознан, но еще не подтвержден
            print(f"[~] Камера {cam_conf['name']} -> {num} (распознано {status['consecutive_count']}/3 раз)")
    else:
        # Сбрасываем трекер при отсутствии распознавания
        processor.recognition_tracker.add_recognition(None)
        if processor.debug:
            status = processor.recognition_tracker.get_current_status()
            print(f"[~] Камера {cam_conf['name']} -> не распознано (сброс счетчика)")
    
    return num

async def check_depo_empty_periodically(processor, cam_conf, interval=300, executor=None):
    """Периодически проверяет, не опустело ли депо"""
    while True:
        try:
            cap = cv2.VideoCapture(cam_conf["rtsp_url"])
            if not cap.isOpened(): 
                print(f"[!] Не удалось подключиться к камере {cam_conf['name']} для проверки депо")
                await asyncio.sleep(60)  # Ждем 1 минуту перед повторной попыткой
                continue
                
            print(f"[✓] Подключение для проверки депо: {cam_conf['name']}")
            loop = asyncio.get_event_loop()
            
            while True:
                await asyncio.sleep(interval)  # Ждем указанный интервал
                
                # Читаем кадр асинхронно
                ret, frame = await loop.run_in_executor(executor, cap.read)
                
                if not ret:
                    print(f"[!] Не удалось получить кадр от камеры {cam_conf['name']} для проверки депо")
                    break
                
                # Выводим разрешение снимка при проверке депо
                height, width = frame.shape[:2]
                print(f"[DEBUG] Разрешение снимка с камеры {cam_conf['name']} при проверке депо: {width}x{height}")
                    
                # Проверяем, пустое ли депо асинхронно
                is_empty = await loop.run_in_executor(executor, processor.compare_with_clean, frame)
                
                if is_empty:
                    # Получаем активные события асинхронно
                    active_events = await loop.run_in_executor(executor, processor.get_active_depo_events)
                    
                    for event_id, train_name in active_events:
                        print(f"[✓] Поезд {train_name} выехал из депо (depot_id: {processor.depot_id}, track_id: {processor.track_id})")
                        await loop.run_in_executor(executor, processor.update_depo_event_out_time, event_id)
            
            cap.release()
            
        except Exception as e:
            print(f"[!] Ошибка в check_depo_empty_periodically для {cam_conf['name']}: {e}")
            await asyncio.sleep(60)  # Ждем 1 минуту перед повторной попыткой

async def camera_loop(cam_conf, processor, interval, executor):
    """Основной цикл обработки кадров с камеры с переподключением"""
    reconnect_attempts = 0
    max_reconnect_attempts = 5
    reconnect_delay = 10  # секунд
    
    while reconnect_attempts < max_reconnect_attempts:
        try:
            cap = cv2.VideoCapture(cam_conf["rtsp_url"])
            if not cap.isOpened(): 
                print(f"[!] Не удалось подключиться к камере {cam_conf['name']} (попытка {reconnect_attempts + 1}/{max_reconnect_attempts})")
                reconnect_attempts += 1
                await asyncio.sleep(reconnect_delay)
                continue
                
            print(f"[✓] Успешное подключение к камере {cam_conf['name']}")
            reconnect_attempts = 0  # Сбрасываем счетчик при успешном подключении
            
            loop = asyncio.get_event_loop()
            
            while True:
                # Читаем кадр асинхронно
                ret, frame = await loop.run_in_executor(executor, cap.read)
                
                if not ret:
                    print(f"[!] Потеряно соединение с камерой {cam_conf['name']}. Попытка переподключения...")
                    break
                
                # Выводим разрешение снимка
                height, width = frame.shape[:2]
                print(f"[DEBUG] Разрешение снимка с камеры {cam_conf['name']}: {width}x{height}")
                    
                # Обрабатываем кадр асинхронно
                await process_camera_frame(frame, cam_conf, processor, executor)
                await asyncio.sleep(interval)
                
        except Exception as e:
            print(f"[!] Ошибка в camera_loop для {cam_conf['name']}: {e}")
            reconnect_attempts += 1
            if reconnect_attempts < max_reconnect_attempts:
                print(f"[~] Повторная попытка подключения к {cam_conf['name']} через {reconnect_delay} сек...")
                await asyncio.sleep(reconnect_delay)
            else:
                print(f"[!] Превышено максимальное количество попыток подключения к {cam_conf['name']}")
                break
        finally:
            if 'cap' in locals():
                cap.release()
    
    print(f"[!] Камера {cam_conf['name']} отключена. Превышено максимальное количество попыток подключения.")

async def main():
    with open("config_cams.json","r",encoding="utf-8") as f:
        cams_config = json.load(f)
    if not isinstance(cams_config,list):
        cams_config = [cams_config]
    
    # Создаем общий ThreadPoolExecutor для всех задач
    executor = ThreadPoolExecutor(max_workers=4)
    
    tasks = []
    for cam_conf in cams_config:
        processor = ImageProcessor(cam_conf)
        roi = ensure_roi(cam_conf, processor)
        if not roi: 
            print(f"[!] Не удалось настроить ROI для камеры {cam_conf['name']}. Пропускаем.")
            continue
            
        interval = cam_conf.get("capture_interval", 4)
        check_interval = cam_conf.get("depo_check_interval", 300)
        
        # Задача для распознавания номеров (непрерывная с переподключением)
        recognition_task = asyncio.create_task(camera_loop(cam_conf, processor, interval, executor))
        tasks.append(recognition_task)
        
        # Задача для периодической проверки пустого депо
        check_task = asyncio.create_task(check_depo_empty_periodically(processor, cam_conf, check_interval, executor))
        tasks.append(check_task)
        
        print(f"[✓] Запущены задачи для камеры {cam_conf['name']}: распознавание каждые {interval}с, проверка депо каждые {check_interval}с")
        
    if tasks:
        print(f"[✓] Запущено всего задач: {len(tasks)}")
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            print(f"[!] Ошибка в одной из задач: {e}")
    else:
        print("[!] Нет камер для обработки")
    
    # Корректное завершение executor
    executor.shutdown(wait=True)

if __name__=="__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n[~] Завершение работы скрипта")
    except Exception as e:
        print(f"[!] Критическая ошибка: {e}")