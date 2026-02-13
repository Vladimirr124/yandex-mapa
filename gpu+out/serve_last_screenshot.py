"""
Простой HTTP-сервер: отдаёт последний по времени .jpg из recognized_screenshots по адресу /last.jpg
Порт 8766. Запуск: python serve_last_screenshot.py
"""
import os
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCREENSHOTS_DIR = os.path.join(SCRIPT_DIR, "recognized_screenshots")
PORT = 8766


def get_latest_jpg():
    if not os.path.isdir(SCREENSHOTS_DIR):
        return None
    latest = None
    latest_mtime = 0
    for name in os.listdir(SCREENSHOTS_DIR):
        if name.lower().endswith(".jpg"):
            path = os.path.join(SCREENSHOTS_DIR, name)
            try:
                mtime = os.path.getmtime(path)
                if mtime > latest_mtime:
                    latest_mtime = mtime
                    latest = path
            except OSError:
                pass
    return latest


class Handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        """CORS preflight: нужен, когда карта с GitHub Pages делает fetch() с заголовком ngrok-skip-browser-warning."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "ngrok-skip-browser-warning")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_GET(self):
        if self.path.split("?")[0] == "/last.jpg":
            path = get_latest_jpg()
            if path is None:
                self.send_response(404)
                self.end_headers()
                return
            try:
                with open(path, "rb") as f:
                    data = f.read()
            except OSError:
                self.send_response(500)
                self.end_headers()
                return
            self.send_response(200)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


def main():
    os.chdir(SCRIPT_DIR)
    latest = get_latest_jpg()
    if latest:
        print(f"[~] Папка скриншотов: {SCREENSHOTS_DIR}")
        print(f"[~] Последний файл: {os.path.basename(latest)}")
    else:
        print(f"[!] В папке нет .jpg или папка не найдена: {SCREENSHOTS_DIR}")
    server = HTTPServer(("0.0.0.0", PORT), Handler)
    print(f"[~] Сервер: http://localhost:{PORT}/last.jpg")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[~] Остановка")
        server.shutdown()


if __name__ == "__main__":
    main()
    sys.exit(0)
