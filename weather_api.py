"""真实天气数据层。

数据来源：Open-Meteo 免费天气 API（无需 API Key）。
- 预报接口：https://api.open-meteo.com/v1/forecast
- 地名检索：https://geocoding-api.open-meteo.com/v1/search

城市解析策略（可靠性优先）：
1. 先查本地精确坐标表 city_coords.json（收录 100+ 个中国城市，坐标经过人工校对）；
2. 再查别名表（"广州市"/"羊城" → 广州，"鹏城" → 深圳 等）；
3. 最后才走在线地名检索，并要求检索结果名称与输入严格匹配，避免同音异地错配。

不再使用按哈希生成的模拟数据，返回的都是真实预报值。
"""

from __future__ import annotations

import json
import os
import re
import ssl
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta

_HERE = os.path.dirname(os.path.abspath(__file__))
_CITY_TABLE_PATH = os.path.join(_HERE, "city_coords.json")

FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
HTTP_TIMEOUT = 12
CACHE_TTL = 600  # 秒；真实数据不需要频繁刷新
MAX_FORECAST_DAYS = 16  # Open-Meteo 免费版最多 16 天

# 别名 / 俗称 / 常见简称 -> 标准城市名
CITY_ALIASES = {
    "羊城": "广州", "花城": "广州",
    "鹏城": "深圳",
    "帝都": "北京", "首都": "北京",
    "魔都": "上海",
    "山城": "重庆",
    "蓉城": "成都", "蓉": "成都",
    "春城": "昆明",
    "泉城": "济南",
    "冰城": "哈尔滨",
    "星城": "长沙",
    "榕城": "福州",
    "金城": "兰州",
    "夏都": "西宁",
    "申城": "上海",
    "杭城": "杭州",
    "津门": "天津",
    "东方之珠": "香港",
    "香港特别行政区": "香港",
    "澳门特别行政区": "澳门",
    "香港岛": "香港",
    "氹仔": "澳门",
    "台北市": "台北",
    "长安": "西安",
    "鹏": "深圳",
    "穗": "广州",
}

# 常见行政区后缀，用于把“广州市 / 朝阳区 / 肇庆市”等还原成标准名
_ADMIN_SUFFIXES = (
    "特别行政区", "自治州", "自治县", "自治区", "地区",
    "市辖区", "新区", "市", "县", "区", "镇", "乡",
)

WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# WMO 天气代码 -> (中文描述, emoji)
WMO_CODE_MAP = {
    0: ("晴", "☀️"),
    1: ("晴间多云", "🌤️"),
    2: ("多云", "⛅"),
    3: ("阴", "☁️"),
    45: ("有雾", "🌫️"),
    48: ("雾凇", "🌫️"),
    51: ("小毛毛雨", "🌦️"),
    53: ("毛毛雨", "🌦️"),
    55: ("较强毛毛雨", "🌧️"),
    56: ("冻毛毛雨", "🌧️"),
    57: ("强冻毛毛雨", "🌧️"),
    61: ("小雨", "🌦️"),
    63: ("中雨", "🌧️"),
    65: ("大雨", "🌧️"),
    66: ("冻雨", "🌧️"),
    67: ("强冻雨", "🌧️"),
    71: ("小雪", "🌨️"),
    73: ("中雪", "🌨️"),
    75: ("大雪", "❄️"),
    77: ("米雪", "🌨️"),
    80: ("阵雨", "🌦️"),
    81: ("强阵雨", "🌧️"),
    82: ("暴雨", "⛈️"),
    85: ("阵雪", "🌨️"),
    86: ("强阵雪", "❄️"),
    95: ("雷阵雨", "⛈️"),
    96: ("雷阵雨伴冰雹", "⛈️"),
    99: ("强雷阵雨伴冰雹", "⛈️"),
}

# 蒲福风级：风速上限（km/h）
_BEAUFORT_LIMITS = [1, 6, 12, 20, 29, 39, 50, 62, 75, 89, 103, 118]

_cache: dict[str, tuple[float, object]] = {}


def _cache_get(key: str):
    hit = _cache.get(key)
    if not hit:
        return None
    ts, value = hit
    if time.time() - ts > CACHE_TTL:
        _cache.pop(key, None)
        return None
    return value


def _cache_set(key: str, value: object) -> None:
    _cache[key] = (time.time(), value)


def _http_get_json(url: str, params: dict) -> dict:
    query = urllib.parse.urlencode(params)
    request = urllib.request.Request(
        f"{url}?{query}",
        headers={"User-Agent": "ai-weather-agent-demo/1.0", "Accept": "application/json"},
    )
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, timeout=HTTP_TIMEOUT, context=context) as response:
        return json.loads(response.read().decode("utf-8"))


def load_city_table() -> dict:
    """读取本地城市坐标表。"""
    cached = _cache_get("city_table")
    if cached is not None:
        return cached
    with open(_CITY_TABLE_PATH, "r", encoding="utf-8") as file:
        table = json.load(file)
    _cache_set("city_table", table)
    return table


def _normalize_city_name(raw: str) -> str:
    """去掉行政区后缀和空白，得到查询用的标准名。"""
    name = (raw or "").strip()
    name = re.sub(r"\s+", "", name)
    for suffix in _ADMIN_SUFFIXES:
        if name.endswith(suffix) and len(name) > len(suffix) + 1:
            name = name[: -len(suffix)]
            break
    return name


def _geocode_online(name: str) -> dict | None:
    """在线地名检索兜底。

    要求结果名与查询名严格一致，且必须是“城市级”地点，
    避免把“火星天气”这类无意义输入匹配成某个同名小村庄。
    """
    for language, query in (("zh", name), ("en", name)):
        try:
            payload = _http_get_json(
                GEOCODE_URL,
                {"name": query, "count": 10, "language": language, "format": "json"},
            )
        except Exception:
            continue

        candidates = []
        for item in payload.get("results") or []:
            result_name = _normalize_city_name(item.get("name") or "")
            if result_name != name and name not in result_name and result_name not in name:
                continue
            feature_code = str(item.get("feature_code") or "")
            population = item.get("population") or 0
            # PPLC/PPLA* 是首都、省会、地级市、区县驻地；其余必须是人口 5 万以上的城镇
            if not feature_code.startswith("PPLA") and feature_code != "PPLC" and population < 50000:
                continue
            candidates.append(item)

        if not candidates:
            continue

        candidates.sort(key=lambda item: item.get("population") or 0, reverse=True)
        top = candidates[0]
        admin_parts = [top.get("admin1"), top.get("admin2"), top.get("country")]
        admin = " · ".join([part for part in admin_parts if part]) or "未知地区"
        return {
            "name": _normalize_city_name(top.get("name") or name),
            "admin": admin,
            "lat": round(float(top["latitude"]), 4),
            "lon": round(float(top["longitude"]), 4),
            "source": "online",
        }

    return None


def resolve_city(raw: str) -> dict | None:
    """把用户输入的城市解析成带坐标的城市信息，解析失败返回 None。"""
    name = _normalize_city_name(raw)
    if not name:
        return None

    table = load_city_table()

    if name in table:
        info = table[name]
        return {"name": name, "admin": info["admin"], "lat": info["lat"], "lon": info["lon"], "source": "local"}

    alias = CITY_ALIASES.get(name)
    if alias and alias in table:
        info = table[alias]
        return {"name": alias, "admin": info["admin"], "lat": info["lat"], "lon": info["lon"], "source": "alias"}

    # 用户可能写了“广州市天河区”这类更长的串，尝试在其中找到已知城市
    for city in sorted(table, key=len, reverse=True):
        if city in name:
            info = table[city]
            return {"name": city, "admin": info["admin"], "lat": info["lat"], "lon": info["lon"], "source": "local"}

    return _geocode_online(name)


def list_supported_cities() -> list[str]:
    """返回本地已知的城市名列表（按拼音无关的固定顺序）。"""
    return list(load_city_table().keys())


def describe_wmo(code: int | None) -> tuple[str, str]:
    """把 WMO 天气代码翻译成中文描述和 emoji。"""
    if code is None:
        return "未知", "❓"
    return WMO_CODE_MAP.get(int(code), ("未知", "❓"))


def kmh_to_beaufort(kmh: float | None) -> str:
    """把 km/h 风速换算成蒲福风级的中文描述。"""
    if kmh is None:
        return "未知"
    if kmh < 1:
        return "0级"
    for level, limit in enumerate(_BEAUFORT_LIMITS, start=1):
        if kmh <= limit:
            return f"{level}级"
    return "12级以上"


def _daily_humidity_mean(times: list[str], values: list) -> dict[str, float]:
    """把逐小时湿度按日期求平均。"""
    buckets: dict[str, list[float]] = {}
    for stamp, value in zip(times or [], values or []):
        if value is None:
            continue
        day = str(stamp)[:10]
        buckets.setdefault(day, []).append(float(value))
    return {day: sum(items) / len(items) for day, items in buckets.items() if items}


def fetch_forecast(city: dict, start_offset: int = 0, days: int = 1) -> dict:
    """获取真实预报。

    返回 {"city": {...}, "items": [...], "current": {...}}，失败时抛出异常。
    """
    start_offset = max(0, int(start_offset))
    days = max(1, min(int(days), MAX_FORECAST_DAYS))
    forecast_days = min(start_offset + days, MAX_FORECAST_DAYS)

    cache_key = f"forecast:{city['lat']},{city['lon']}:{start_offset}:{days}"
    cached = _cache_get(cache_key)
    if cached is not None:
        return cached

    payload = _http_get_json(
        FORECAST_URL,
        {
            "latitude": city["lat"],
            "longitude": city["lon"],
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "precipitation_probability_max",
                    "relative_humidity_2m_mean",
                    "wind_speed_10m_max",
                    "uv_index_max",
                ]
            ),
            "current": "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m,is_day",
            "timezone": "Asia/Shanghai",
            "forecast_days": forecast_days,
        },
    )

    daily = payload.get("daily") or {}
    times = daily.get("time") or []

    # 部分字段老版本接口不返回，缺失时用逐小时数据补算
    humidity_series = daily.get("relative_humidity_2m_mean")
    if humidity_series is None:
        try:
            hourly = _http_get_json(
                FORECAST_URL,
                {
                    "latitude": city["lat"],
                    "longitude": city["lon"],
                    "hourly": "relative_humidity_2m",
                    "timezone": "Asia/Shanghai",
                    "forecast_days": forecast_days,
                },
            ).get("hourly") or {}
            humidity_by_day = _daily_humidity_mean(hourly.get("time"), hourly.get("relative_humidity_2m"))
        except Exception:
            humidity_by_day = {}
        humidity_series = [humidity_by_day.get(day) for day in times]

    items: list[dict] = []
    for index, day_text in enumerate(times):
        if index < start_offset:
            continue
        if len(items) >= days:
            break

        code = (daily.get("weather_code") or [None] * len(times))[index]
        condition, emoji = describe_wmo(code)
        high = (daily.get("temperature_2m_max") or [None] * len(times))[index]
        low = (daily.get("temperature_2m_min") or [None] * len(times))[index]
        pop = (daily.get("precipitation_probability_max") or [None] * len(times))[index]
        humidity = humidity_series[index] if humidity_series and index < len(humidity_series) else None
        wind_kmh = (daily.get("wind_speed_10m_max") or [None] * len(times))[index]
        uv = (daily.get("uv_index_max") or [None] * len(times))[index]

        try:
            weekday = WEEKDAYS[datetime.strptime(day_text, "%Y-%m-%d").weekday()]
        except (TypeError, ValueError):
            weekday = ""

        items.append(
            {
                "日期": day_text,
                "星期": weekday,
                "城市": city["name"],
                "地区": city["admin"],
                "天气": condition,
                "天气图标": emoji,
                "最高温": f"{round(high)}°C" if high is not None else "未知",
                "最低温": f"{round(low)}°C" if low is not None else "未知",
                "湿度": f"{round(humidity)}%" if humidity is not None else "未知",
                "风力": kmh_to_beaufort(wind_kmh),
                "降水概率": f"{round(pop)}%" if pop is not None else "未知",
                "紫外线": round(uv) if uv is not None else None,
                "数据源": "Open-Meteo 真实预报",
            }
        )

    if not items:
        raise RuntimeError(f"没有取到 {city['name']} 的预报数据。")

    current = None
    current_raw = payload.get("current") or {}
    if current_raw:
        current_condition, current_emoji = describe_wmo(current_raw.get("weather_code"))
        current = {
            "城市": city["name"],
            "天气": current_condition,
            "天气图标": current_emoji,
            "温度": (
                f"{round(current_raw['temperature_2m'])}°C"
                if current_raw.get("temperature_2m") is not None
                else "未知"
            ),
            "体感温度": (
                f"{round(current_raw['apparent_temperature'])}°C"
                if current_raw.get("apparent_temperature") is not None
                else "未知"
            ),
            "湿度": (
                f"{round(current_raw['relative_humidity_2m'])}%"
                if current_raw.get("relative_humidity_2m") is not None
                else "未知"
            ),
            "风速": kmh_to_beaufort(current_raw.get("wind_speed_10m")),
            "白天": bool(current_raw.get("is_day", 1)),
        }

    result = {"city": city, "items": items, "current": current}
    _cache_set(cache_key, result)
    return result


def time_range_from_text(text: str) -> tuple[int, int, str]:
    """从自然语言里解析查询的时间范围。

    返回 (起始天数偏移, 天数, 人话描述)。
    """
    text = text or ""

    chinese_numbers = {
        "一": 1, "两": 2, "二": 2, "三": 3, "四": 4, "五": 5,
        "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    }

    # 明确的第 N 天（大后天 / 后天 / 明天）
    if "大后天" in text:
        return 3, 1, "大后天"
    if "后天" in text:
        return 2, 1, "后天"
    if re.search(r"明天|明日|明儿", text):
        return 1, 1, "明天"

    # 未来 N 天 / N 天
    match = re.search(r"(\d+)\s*天", text)
    days = int(match.group(1)) if match else None
    if days is None:
        match = re.search(r"([一两二三四五六七八九十])\s*天", text)
        if match:
            days = chinese_numbers[match.group(1)]

    if days:
        return 0, max(1, min(days, MAX_FORECAST_DAYS)), f"未来 {days} 天"

    if re.search(r"一周|一星期|7\s*天|七天|这周|本周|整周", text):
        return 0, 7, "未来一周"
    if re.search(r"最近|这几天|近几|未来几天|接下来|往后", text):
        return 0, 3, "未来 3 天"

    return 0, 1, "今天"


def today_text() -> str:
    return date.today().strftime("%Y-%m-%d")


def tomorrow_text() -> str:
    return (date.today() + timedelta(days=1)).strftime("%Y-%m-%d")
