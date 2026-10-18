"""天气 Agent Python 后端。

本文件只使用 Python 标准库，提供：
* 静态文件服务（index.html、city_coords.json 等）
* Open-Meteo 地理编码与天气代理
* 基于 sqlite3 的注册、登录、退出和聊天记录持久化

数据库文件：web/data/weather_agent.db（首次启动自动创建）。
密码使用 PBKDF2-HMAC-SHA256 + 随机 salt 哈希后保存，数据库不保存明文密码。
"""

from __future__ import annotations

import hashlib
import hmac
import json
import mimetypes
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timezone
from http.cookies import SimpleCookie
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, unquote, urlencode, urlparse
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
DB_PATH = DATA_DIR / "weather_agent.db"
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "3000"))

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
UPSTREAM_TIMEOUT = float(os.environ.get("WEATHER_UPSTREAM_TIMEOUT", "12"))
USER_AGENT = "ai-weather-agent-demo/2.0 (Python stdlib)"

# PBKDF2 迭代次数可通过环境变量调高，默认值兼顾普通电脑的登录体验与安全性。
PBKDF2_ITERATIONS = int(os.environ.get("PASSWORD_PBKDF2_ITERATIONS", "240000"))
SESSION_TTL = int(os.environ.get("SESSION_TTL_SECONDS", str(7 * 24 * 3600)))
MAX_BODY_BYTES = 1024 * 1024
MAX_CHAT_CONTENT = 100_000
DEFAULT_ALLOWED_ORIGINS = {
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:8502",
    "http://127.0.0.1:8502",
}
ALLOWED_ORIGINS = DEFAULT_ALLOWED_ORIGINS | {
    origin.strip()
    for origin in os.environ.get("ALLOWED_ORIGINS", "").split(",")
    if origin.strip()
}

_session_lock = threading.Lock()
_sessions: Dict[str, Dict[str, Any]] = {}


def utc_now() -> str:
    """返回 SQLite 和前端统一使用的 UTC ISO 时间。"""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def get_db() -> sqlite3.Connection:
    """打开数据库并设置行对象，调用方负责 close。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH, timeout=10)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db() -> None:
    """创建用户表和聊天记录表；重复启动不会覆盖已有数据。"""
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password_hash TEXT NOT NULL,
                password_salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                role TEXT NOT NULL CHECK (role IN ('user', 'assistant', 'system')),
                content TEXT NOT NULL,
                created_at TEXT NOT NULL,
                FOREIGN KEY (username) REFERENCES users(username) ON DELETE CASCADE
            );

            CREATE INDEX IF NOT EXISTS idx_chat_messages_user_time
                ON chat_messages(username, id);
            """
        )


def hash_password(password: str, salt: Optional[bytes] = None) -> Tuple[str, str]:
    """生成密码哈希和 salt 的十六进制字符串。"""
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac(
        "sha256", password.encode("utf-8"), salt, PBKDF2_ITERATIONS
    )
    return digest.hex(), salt.hex()


def verify_password(password: str, stored_hash: str, stored_salt: str) -> bool:
    """以恒定时间比较密码哈希，降低时序攻击风险。"""
    try:
        digest, _ = hash_password(password, bytes.fromhex(stored_salt))
    except (TypeError, ValueError):
        return False
    return hmac.compare_digest(digest, stored_hash)


def create_session(username: str) -> str:
    token = secrets.token_urlsafe(32)
    with _session_lock:
        _sessions[token] = {"username": username, "expires": time.time() + SESSION_TTL}
    return token


def remove_session(token: Optional[str]) -> None:
    if token:
        with _session_lock:
            _sessions.pop(token, None)


def clean_sessions() -> None:
    now = time.time()
    with _session_lock:
        expired = [token for token, value in _sessions.items() if value["expires"] <= now]
        for token in expired:
            _sessions.pop(token, None)


def username_from_cookie(cookie_header: str) -> Optional[str]:
    """解析 HttpOnly session cookie，并在有效时续期。"""
    if not cookie_header:
        return None
    cookie = SimpleCookie()
    try:
        cookie.load(cookie_header)
    except Exception:
        return None
    morsel = cookie.get("weather_session")
    if not morsel:
        return None
    token = morsel.value
    now = time.time()
    with _session_lock:
        session = _sessions.get(token)
        if not session or session["expires"] <= now:
            _sessions.pop(token, None)
            return None
        session["expires"] = now + SESSION_TTL
        return str(session["username"])


def validate_username(username: Any) -> Optional[str]:
    """限制用户名长度和字符，避免空用户名、控制字符和极端超长输入。"""
    if not isinstance(username, str):
        return None
    value = username.strip()
    if not 3 <= len(value) <= 32:
        return None
    if not re.fullmatch(r"[A-Za-z0-9_]+", value):
        return None
    return value


def validate_password(password: Any) -> Optional[str]:
    if not isinstance(password, str) or not 6 <= len(password) <= 128:
        return None
    if any(char in password for char in ("\x00", "\r", "\n")):
        return None
    return password


def json_response(ok: bool, **payload: Any) -> Dict[str, Any]:
    return {"ok": ok, **payload}


def upstream_json(url: str) -> Tuple[int, Dict[str, Any]]:
    """请求天气上游并解析 JSON；返回 HTTP 状态和数据。"""
    request = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
    try:
        with urlopen(request, timeout=UPSTREAM_TIMEOUT) as response:
            raw = response.read(MAX_BODY_BYTES)
            status = int(getattr(response, "status", 200))
    except HTTPError as exc:
        detail = "天气上游接口错误"
        try:
            detail = json.loads(exc.read(MAX_BODY_BYTES).decode("utf-8", "replace")).get(
                "reason", detail
            )
        except Exception:
            pass
        return int(exc.code), {"error": detail}
    except (TimeoutError, URLError, OSError) as exc:
        return 502, {"error": f"天气数据源请求失败：{exc}"
                      if str(exc) else "天气数据源请求失败"}
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return 502, {"error": "天气数据源返回的不是合法 JSON"}
    if not isinstance(data, dict):
        return 502, {"error": "天气数据源返回格式异常"}
    return status, data


def build_forecast_url(params: Dict[str, str]) -> Optional[str]:
    try:
        latitude = float(params.get("lat", ""))
        longitude = float(params.get("lon", ""))
    except (TypeError, ValueError):
        return None
    if not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
        return None
    try:
        days = int(params.get("forecast_days", "7"))
    except ValueError:
        days = 7
    days = max(1, min(16, days))
    query = {
        "latitude": str(latitude),
        "longitude": str(longitude),
        "daily": (
            "weather_code,temperature_2m_max,temperature_2m_min,"
            "precipitation_probability_max,relative_humidity_2m_mean,"
            "wind_speed_10m_max,wind_direction_10m_dominant,uv_index_max"
        ),
        "current": (
            "temperature_2m,apparent_temperature,relative_humidity_2m,"
            "weather_code,wind_speed_10m,wind_direction_10m"
        ),
        "timezone": "Asia/Shanghai",
        "forecast_days": str(days),
    }
    return f"{FORECAST_URL}?{urlencode(query)}"


def build_geocode_url(params: Dict[str, str]) -> Optional[str]:
    name = (params.get("name") or "").strip()
    if not name or len(name) > 80:
        return None
    query = {
        "name": name,
        "count": "10",
        "language": params.get("language") or "zh",
        "format": "json",
        "countryCode": params.get("country_code") or "CN",
    }
    return f"{GEOCODE_URL}?{urlencode(query)}"


class WeatherHandler(BaseHTTPRequestHandler):
    """处理 API 和静态文件请求。"""

    server_version = "WeatherAgentPython/1.0"

    def log_message(self, format: str, *args: Any) -> None:
        # 保留简洁访问日志，便于本地调试。
        print(f"[{self.log_date_time_string()}] {self.address_string()} - {format % args}")

    def _cors_headers(self) -> Dict[str, str]:
        origin = self.headers.get("Origin")
        headers = {
            "Access-Control-Allow-Methods": "GET, POST, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type",
            "Access-Control-Max-Age": "600",
            "Vary": "Origin",
        }
        if origin in ALLOWED_ORIGINS:
            headers["Access-Control-Allow-Origin"] = origin
            headers["Access-Control-Allow-Credentials"] = "true"
        elif not origin:
            # 无 Origin 通常是同源请求或 curl 调试请求；不携带浏览器凭据时允许查看。
            headers["Access-Control-Allow-Origin"] = "*"
        return headers

    def _send_bytes(
        self,
        status: int,
        body: bytes,
        content_type: str,
        extra: Optional[Dict[str, str]] = None,
    ) -> None:
        headers = {"Content-Type": content_type, "Content-Length": str(len(body)), **self._cors_headers()}
        if extra:
            headers.update(extra)
        self.send_response(status)
        for key, value in headers.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: Dict[str, Any], extra: Optional[Dict[str, str]] = None) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        headers = {"Cache-Control": "no-store"}
        if extra:
            headers.update(extra)
        self._send_bytes(status, body, "application/json; charset=utf-8", headers)

    def _error(self, status: int, message: str, code: Optional[str] = None) -> None:
        data: Dict[str, Any] = json_response(False, error=message)
        if code:
            data["code"] = code
        self._send_json(status, data)

    def _read_json(self) -> Optional[Dict[str, Any]]:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            self._error(400, "请求体长度非法", "INVALID_BODY")
            return None
        if length <= 0 or length > MAX_BODY_BYTES:
            self._error(413 if length > MAX_BODY_BYTES else 400, "请求体为空或过大", "INVALID_BODY")
            return None
        try:
            raw = self.rfile.read(length)
            value = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._error(400, "请求体必须是合法 JSON", "INVALID_JSON")
            return None
        if not isinstance(value, dict):
            self._error(400, "请求体必须是 JSON 对象", "INVALID_JSON")
            return None
        return value

    def _current_user(self) -> Optional[str]:
        return username_from_cookie(self.headers.get("Cookie", ""))

    def _require_user(self) -> Optional[str]:
        username = self._current_user()
        if not username:
            self._error(401, "请先登录后再使用天气 Agent", "AUTH_REQUIRED")
            return None
        return username

    def do_OPTIONS(self) -> None:  # noqa: N802
        self._send_bytes(204, b"", "text/plain; charset=utf-8")

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path.startswith("/api/"):
            self.handle_api("GET", parsed.path, parse_qs(parsed.query, keep_blank_values=True))
        else:
            self.serve_static(parsed.path)

    def do_POST(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self._error(404, "未知接口")
            return
        self.handle_api("POST", parsed.path, parse_qs(parsed.query, keep_blank_values=True))

    def do_DELETE(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if not parsed.path.startswith("/api/"):
            self._error(404, "未知接口")
            return
        self.handle_api("DELETE", parsed.path, parse_qs(parsed.query, keep_blank_values=True))

    @staticmethod
    def first(params: Dict[str, Iterable[str]], key: str) -> str:
        values = params.get(key, [])
        return str(next(iter(values), ""))

    def handle_api(self, method: str, path: str, params: Dict[str, Iterable[str]]) -> None:
        """统一 API 路由。"""
        try:
            if path == "/api/health" and method == "GET":
                self._send_json(200, json_response(True, time=utc_now()))
                return
            if path == "/api/auth/register" and method == "POST":
                self.register()
                return
            if path == "/api/auth/login" and method == "POST":
                self.login()
                return
            if path == "/api/auth/logout" and method == "POST":
                self.logout()
                return
            if path == "/api/auth/me" and method == "GET":
                self.auth_me()
                return
            if path == "/api/chat/history" and method == "GET":
                self.chat_history(params)
                return
            if path == "/api/chat/history" and method == "DELETE":
                self.clear_chat_history()
                return
            if path == "/api/chat" and method == "POST":
                self.save_chat()
                return
            if path == "/api/chat/messages" and method == "POST":
                # 前端也可使用语义更明确的 /api/chat/messages 路径。
                self.save_chat()
                return
            if path == "/api/geocode" and method == "GET":
                self.geocode(params)
                return
            if path == "/api/forecast" and method == "GET":
                self.forecast(params)
                return
            self._error(404, f"未知接口：{path}", "NOT_FOUND")
        except Exception as exc:  # 防止单个请求异常导致服务进程退出
            print(f"API error on {method} {path}: {exc}")
            self._error(500, "服务器内部错误，请稍后重试", "INTERNAL_ERROR")

    def register(self) -> None:
        data = self._read_json()
        if data is None:
            return
        username = validate_username(data.get("username"))
        password = validate_password(data.get("password"))
        if not username:
            self._error(400, "用户名需为 3-32 位字母、数字或下划线", "INVALID_USERNAME")
            return
        if not password:
            self._error(400, "密码需为 6-128 位", "INVALID_PASSWORD")
            return
        password_hash, salt = hash_password(password)
        try:
            with get_db() as conn:
                conn.execute(
                    "INSERT INTO users(username, password_hash, password_salt, created_at) VALUES (?, ?, ?, ?)",
                    (username, password_hash, salt, utc_now()),
                )
        except sqlite3.IntegrityError:
            self._error(409, "用户名已存在，请换一个用户名", "USERNAME_EXISTS")
            return
        self._send_json(201, json_response(True, username=username, message="注册成功，请登录"))

    def login(self) -> None:
        data = self._read_json()
        if data is None:
            return
        username = validate_username(data.get("username"))
        password = validate_password(data.get("password"))
        if not username or not password:
            self._error(401, "账号或密码错误", "INVALID_CREDENTIALS")
            return
        with get_db() as conn:
            row = conn.execute(
                "SELECT username, password_hash, password_salt FROM users WHERE username = ?", (username,)
            ).fetchone()
        if not row or not verify_password(password, row["password_hash"], row["password_salt"]):
            self._error(401, "账号或密码错误", "INVALID_CREDENTIALS")
            return
        token = create_session(username)
        cookie = f"weather_session={token}; Path=/; Max-Age={SESSION_TTL}; HttpOnly; SameSite=Lax"
        self._send_json(
            200,
            json_response(True, username=username, user={"username": username}, message="登录成功"),
            {"Set-Cookie": cookie},
        )

    def logout(self) -> None:
        cookie_header = self.headers.get("Cookie", "")
        cookie = SimpleCookie()
        try:
            cookie.load(cookie_header)
        except Exception:
            pass
        token = cookie.get("weather_session").value if cookie.get("weather_session") else None
        remove_session(token)
        self._send_json(
            200,
            json_response(True, message="已退出登录"),
            {"Set-Cookie": "weather_session=; Path=/; Max-Age=0; HttpOnly; SameSite=Lax"},
        )

    def auth_me(self) -> None:
        username = self._current_user()
        if not username:
            self._send_json(200, json_response(True, authenticated=False))
            return
        self._send_json(200, json_response(True, authenticated=True, user={"username": username}))

    def chat_history(self, params: Dict[str, Iterable[str]]) -> None:
        username = self._require_user()
        if not username:
            return
        try:
            limit = int(self.first(params, "limit") or "200")
        except ValueError:
            limit = 200
        limit = max(1, min(500, limit))
        with get_db() as conn:
            rows = conn.execute(
                "SELECT id, role, content, created_at FROM chat_messages "
                "WHERE username = ? ORDER BY id DESC LIMIT ?",
                (username, limit),
            ).fetchall()
        # 查询时倒序取最近 N 条，再恢复成对话发生顺序。
        messages = [dict(row) for row in reversed(rows)]
        self._send_json(200, json_response(True, username=username, messages=messages))

    def clear_chat_history(self) -> None:
        username = self._require_user()
        if not username:
            return
        with get_db() as conn:
            conn.execute("DELETE FROM chat_messages WHERE username = ?", (username,))
        self._send_json(200, json_response(True, message="聊天记录已清空"))

    def save_chat(self) -> None:
        username = self._require_user()
        if not username:
            return
        data = self._read_json()
        if data is None:
            return

        # 支持单条 message，也支持一次提交 messages 数组，方便前端恢复上下文。
        raw_messages: Any = data.get("messages")
        if raw_messages is None:
            raw_messages = [data]
        if not isinstance(raw_messages, list) or not raw_messages or len(raw_messages) > 100:
            self._error(400, "messages 必须是 1-100 条消息的数组", "INVALID_MESSAGES")
            return

        normalized = []
        for message in raw_messages:
            if not isinstance(message, dict):
                self._error(400, "聊天消息格式错误", "INVALID_MESSAGE")
                return
            role = message.get("role")
            content = message.get("content")
            if role not in ("user", "assistant", "system"):
                self._error(400, "消息 role 只能是 user、assistant 或 system", "INVALID_ROLE")
                return
            if not isinstance(content, str) or not content.strip() or len(content) > MAX_CHAT_CONTENT:
                self._error(400, "消息内容为空或过长", "INVALID_CONTENT")
                return
            normalized.append((username, role, content.strip(), utc_now()))

        with get_db() as conn:
            conn.executemany(
                "INSERT INTO chat_messages(username, role, content, created_at) VALUES (?, ?, ?, ?)",
                normalized,
            )
            last_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        first_id = last_id - len(normalized) + 1
        self._send_json(
            201,
            json_response(True, saved=len(normalized), first_id=first_id, last_id=last_id, messages=[
                {"role": row[1], "content": row[2], "created_at": row[3]} for row in normalized
            ]),
        )

    def geocode(self, params: Dict[str, Iterable[str]]) -> None:
        if not self._require_user():
            return
        query = {key: self.first(params, key) for key in ("name", "language", "country_code", "parent")}
        url = build_geocode_url(query)
        if not url:
            self._error(400, "缺少或非法的 name 参数", "INVALID_LOCATION")
            return
        status, data = upstream_json(url)
        if status >= 400 or data.get("error"):
            self._error(502, f"地理编码请求失败：{data.get('error', '上游接口错误')}", "GEOCODE_ERROR")
            return
        self._send_json(200, data)

    def forecast(self, params: Dict[str, Iterable[str]]) -> None:
        if not self._require_user():
            return
        query = {key: self.first(params, key) for key in ("lat", "lon", "forecast_days")}
        url = build_forecast_url(query)
        if not url:
            self._error(400, "缺少或非法的 lat / lon 参数", "INVALID_COORDINATES")
            return
        status, data = upstream_json(url)
        if status >= 400 or data.get("error"):
            self._error(502, f"天气数据请求失败：{data.get('error', '上游接口错误')}", "WEATHER_ERROR")
            return
        self._send_json(200, data)

    def serve_static(self, pathname: str) -> None:
        """安全地托管 web 目录，阻止 .. 路径穿越。"""
        decoded = unquote(pathname)
        relative = decoded.lstrip("/") or "index.html"
        file_path = (ROOT / relative).resolve()
        try:
            file_path.relative_to(ROOT)
        except ValueError:
            self._error(403, "禁止访问该路径")
            return
        if not file_path.is_file():
            self._error(404, "文件不存在")
            return
        try:
            body = file_path.read_bytes()
        except OSError:
            self._error(500, "读取页面失败")
            return
        content_type = mimetypes.guess_type(str(file_path))[0] or "application/octet-stream"
        if content_type.startswith("text/") or content_type in ("application/javascript", "application/json"):
            content_type += "; charset=utf-8"
        self._send_bytes(200, body, content_type, {"Cache-Control": "no-cache" if file_path.name == "index.html" else "public, max-age=300"})


def main() -> None:
    init_db()
    clean_sessions()
    server = ThreadingHTTPServer((HOST, PORT), WeatherHandler)
    print(f"Weather Agent Python server listening on http://127.0.0.1:{PORT}")
    print(f"SQLite database: {DB_PATH}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n正在停止服务器…")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
