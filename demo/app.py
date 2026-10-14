"""AI 天气查询 Agent：DeepSeek 工具调用 + 和风天气实时数据。"""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import streamlit as st
from dotenv import dotenv_values
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


ENV_PATH = Path(__file__).resolve().parent / ".env"
DEEPSEEK_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_MODEL = "deepseek-flash"
CHINA_TIMEZONE = timezone(timedelta(hours=8))
MAX_DAYS = 10  # 和风天气 v1 每日预报最多 10 天
MAX_HISTORY_MESSAGES = 12
HTTP_TIMEOUT = (5, 12)  # 连接、读取超时（秒）
WEATHER_WORDS = re.compile(
    r"天气|气温|温度|降雨|下雨|有雨|雨吗|下雪|晴|阴|多云|带伞|穿什么|风力|风速|刮风|冷不冷|热不热|冷吗|热吗|防晒|湿度|预报|气象"
)
FOLLOW_UP_WORDS = re.compile(r"今天|明天|明日|后天|大后天|未来|接下来|这几天|\d+\s*天|[一二两三四五六七八九十]\s*天")
DAY_COUNT_WORDS = {"一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
                   "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
CITY_ALIASES = {
    "穗": "广州", "羊城": "广州", "广洲": "广州",
    "鹏城": "深圳", "深市": "深圳", "魔都": "上海",
    "帝都": "北京", "京城": "北京", "春城": "昆明",
}
ADMIN_SUFFIXES = ("特别行政区", "自治区", "自治州", "地区", "省", "市", "区", "县", "旗", "盟")
SUPPORTED_CITY = re.compile(r"^[\u4e00-\u9fffA-Za-z0-9·\-]{1,40}$")


def get_settings() -> tuple[str, str, str, str, str]:
    """每轮只读取本项目 .env；绝不把密钥放入页面或日志。"""
    values = dotenv_values(ENV_PATH)
    return (
        (values.get("LLM_API_KEY") or "").strip(),
        (values.get("LLM_BASE_URL") or DEEPSEEK_BASE_URL).strip(),
        (values.get("LLM_MODEL") or DEFAULT_MODEL).strip(),
        (values.get("QWEATHER_KEY") or "").strip(),
        (values.get("QWEATHER_API_HOST") or "").strip(),
    )


def is_deepseek_url(base_url: str) -> bool:
    parsed = urlparse(base_url)
    return parsed.scheme == "https" and parsed.netloc == "api.deepseek.com" and parsed.path.rstrip("/") in ("", "/v1")


def qweather_host(host: str) -> str:
    """限制天气请求只能发送到和风天气专属域名，避免泄漏密钥。"""
    parsed = urlparse(host if "://" in host else "https://" + host)
    if (parsed.scheme != "https" or not parsed.hostname or
            not parsed.hostname.endswith(".qweatherapi.com") or
            parsed.username or parsed.password or parsed.port or
            parsed.path not in ("", "/") or parsed.query or parsed.fragment):
        raise ValueError("QWEATHER_API_HOST 应填写控制台中的专属域名，例如 abcxyz.qweatherapi.com")
    return parsed.hostname


def weather_error(code: str, message: str) -> dict[str, Any]:
    return {"ok": False, "error_code": code, "error": message}


def request_qweather(host: str, key: str, path: str, params: dict[str, Any]) -> dict[str, Any]:
    """密钥只进入请求头，不进入 URL、异常文本或展示的 JSON。"""
    response = requests.get(
        f"https://{host}{path}",
        params=params,
        headers={"X-QW-Api-Key": key, "Accept": "application/json", "User-Agent": "weather-agent-demo/2.0"},
        timeout=HTTP_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("和风天气返回的数据格式不正确")
    if payload.get("code") is not None and str(payload["code"]) != "200":
        raise ValueError("和风天气接口返回业务错误")
    return payload


def normalize_city(city: str) -> str:
    name = re.sub(r"\s+", "", city or "").strip("，。？！,!? ")
    name = name.removesuffix("市")
    return CITY_ALIASES.get(name, name)


def normalize_location_name(name: str) -> str:
    """统一地点名比较形式，兼容“高要”和“高要区”等行政后缀。"""
    normalized = normalize_city(name)
    for suffix in ADMIN_SUFFIXES:
        if normalized.endswith(suffix):
            normalized = normalized[: -len(suffix)]
            break
    return CITY_ALIASES.get(normalized, normalized)


def validate_city_input(city: str) -> str:
    """清理并校验城市输入，避免空值、超长文本和控制字符进入 GeoAPI。"""
    normalized = normalize_city(city)
    if not normalized:
        raise LookupError("请先提供要查询的城市名称。")
    if len(normalized) > 40 or not SUPPORTED_CITY.fullmatch(normalized):
        raise LookupError("城市名称包含无法识别的字符，请输入城市名或省市组合。")
    return normalized


def location_search_variants(city: str) -> list[tuple[str, str, str]]:
    """生成 GeoAPI 查询候选：完整地点、显式市区组合、无后缀组合。"""
    query = validate_city_input(city)
    region_match = re.match(r"^(.+?(?:省|自治区|特别行政区))(.+)$", query)
    region = region_match.group(1) if region_match else ""
    core = region_match.group(2) if region_match else query
    variants: list[tuple[str, str, str]] = [(core, region, "direct")]

    # “肇庆市高要区”“北京市朝阳区”等带行政后缀的组合。
    parent_match = re.match(r"^(.+?(?:市|州|地区|盟))(.+)$", core)
    if parent_match:
        variants.insert(0, (parent_match.group(2), parent_match.group(1), "district"))
        variants.append((parent_match.group(1), region, "city_fallback"))

    # “肇庆高要”“广州天河”常省略“市/区”，按可能的上级城市拆分重试。
    if not parent_match and len(core) >= 4:
        max_parent_len = min(6, len(core) - 2)
        fallback_parents: list[str] = []
        for parent_len in range(2, max_parent_len + 1):
            parent, child = core[:parent_len], core[parent_len:]
            variants.append((child, parent, "district"))
            fallback_parents.append(parent)
        if fallback_parents:
            variants.extend((parent, region, "city_fallback") for parent in fallback_parents)

    deduped: list[tuple[str, str, str]] = []
    seen: set[tuple[str, str]] = set()
    for location, adm, level in variants:
        key = (normalize_location_name(location), normalize_location_name(adm))
        if key not in seen:
            seen.add(key)
            deduped.append((location, adm, level))
    return deduped


def location_matches_parent(location: dict[str, Any], parent: str) -> bool:
    """判断 GeoAPI 结果是否属于指定的市/省上级行政区。"""
    parent_name = normalize_location_name(parent)
    return any(
        normalize_location_name(str(location.get(field, ""))) == parent_name
        for field in ("adm2", "adm1")
    )


def find_location(city: str, host: str, key: str) -> tuple[dict[str, Any], dict[str, Any], str]:
    """定位市或区县；优先区县，找不到时回退到市级地点。"""
    original = validate_city_input(city)
    last_geo: dict[str, Any] = {}
    for location_query, adm, level in location_search_variants(original):
        params = {"location": location_query, "number": 20, "lang": "zh", "range": "cn"}
        if adm:
            params["adm"] = adm
        geo = request_qweather(host, key, "/geo/v2/city/lookup", params)
        last_geo = geo
        locations = geo.get("location") or []
        target = normalize_location_name(location_query)
        exact = [
            item for item in locations
            if normalize_location_name(str(item.get("name", ""))) == target
        ]
        if adm:
            scoped = [item for item in exact if location_matches_parent(item, adm)]
            if scoped:
                exact = scoped
        if not exact:
            continue
        unique = {(item.get("id"), item.get("adm1"), item.get("adm2")) for item in exact}
        if len(unique) > 1:
            raise LookupError(f"“{original}”对应多个地点，请补充省份或上级城市后再查询。")
        return exact[0], geo, level
    raise LookupError(f"未找到地点“{original}”，请检查市/区县名称或补充上级城市。")


def fetch_real_weather(
    city: str,
    days: int = 1,
    start_offset: int = 0,
    unit: str = "celsius",
) -> dict[str, Any]:
    """GeoAPI 定位后查询实况或每日预报；成功时携带 API 原始 JSON。"""
    _, _, _, key, raw_host = get_settings()
    if not key or not raw_host:
        return weather_error("CONFIG", "请在 demo/.env 配置 QWEATHER_KEY 和 QWEATHER_API_HOST。")
    try:
        host = qweather_host(raw_host)
        unit = str(unit or "celsius").lower()
        if unit not in {"celsius", "fahrenheit"}:
            return weather_error("UNIT", "温度单位只能是 celsius 或 fahrenheit。")
        days, start_offset = int(days), int(start_offset)
        if days < 1 or start_offset < 0 or days + start_offset > MAX_DAYS:
            return weather_error("RANGE", f"目前仅支持从今天起未来 {MAX_DAYS} 天内的查询。")
        location, geo_raw, match_level = find_location(city, host, key)
        lat, lon = float(location["lat"]), float(location["lon"])
        if not (-90 <= lat <= 90 and -180 <= lon <= 180):
            raise ValueError("城市坐标无效")
        is_current = start_offset == 0 and days == 1
        if is_current:
            path = f"/weather/v1/current/{lat:.2f}/{lon:.2f}"
            weather_raw = request_qweather(host, key, path, {"lang": "zh", "localTime": "true"})
            if not isinstance(weather_raw.get("temperature"), dict):
                raise ValueError("实况数据缺少温度字段")
        else:
            path = f"/weather/v1/daily/{lat:.2f}/{lon:.2f}"
            weather_raw = request_qweather(
                host, key, path,
                {"days": days + start_offset, "lang": "zh", "localTime": "true"},
            )
            if not isinstance(weather_raw.get("days"), list) or len(weather_raw["days"]) < days + start_offset:
                raise ValueError("预报数据不完整")
            selected = weather_raw["days"][start_offset:start_offset + days]
            if any(not day.get("forecastStartTime") or
                   not isinstance(day.get("temperatureMax"), dict) or
                   day["temperatureMax"].get("value") is None for day in selected):
                raise ValueError("预报数据缺少必要字段")
        return {
            "ok": True, "source": "和风天气 QWeather",
            "city": {"name": location.get("name"), "adm1": location.get("adm1"),
                     "adm2": location.get("adm2"), "id": location.get("id"),
                     "type": location.get("type"), "match_level": match_level,
                     "requested": validate_city_input(city)},
            "query": {"days": days, "start_offset": start_offset,
                      "kind": "current" if is_current else "forecast", "unit": unit},
            "raw": {"geo": geo_raw, "weather": weather_raw},
        }
    except LookupError as exc:
        return weather_error("CITY_NOT_FOUND", str(exc))
    except requests.Timeout:
        return weather_error("TIMEOUT", "和风天气请求超时，请稍后重试。")
    except requests.HTTPError as exc:
        status = exc.response.status_code if exc.response is not None else None
        return weather_error("HTTP_ERROR", f"和风天气接口报错（HTTP {status or '未知'}），请检查密钥、API Host 与服务权限。")
    except requests.RequestException:
        return weather_error("NETWORK", "无法连接和风天气，请检查网络后重试。")
    except (ValueError, KeyError, TypeError, json.JSONDecodeError):
        return weather_error("API_ERROR", "和风天气返回异常或配置无效，请检查 API Host 与接口权限。")
    except Exception:
        # 第三方字段形状变化等未知异常也不允许穿透到模型后编造天气。
        return weather_error("API_ERROR", "天气数据暂时无法解析，请稍后重试。")


@tool
def get_real_weather(
    city: str,
    days: int = 1,
    start_offset: int = 0,
    unit: str = "celsius",
) -> str:
    """查询和风天气真实数据。
    city 可填写省、市、区、县或组合地点，例如“肇庆高要”“肇庆市高要区”；
    days 为天数；start_offset 为从今天起的偏移（明天=1）；unit 为 celsius 或 fahrenheit。
    工具优先定位区县，区县无数据时返回上级市级天气并标记降级。失败时返回 ok=false，不允许编造天气。
    """
    return json.dumps(fetch_real_weather(city, days, start_offset, unit), ensure_ascii=False)


def create_weather_agent(api_key: str, base_url: str, model_name: str, default_unit: str = "celsius"):
    """创建 DeepSeek Agent；天气结果仍由本地严格校验。"""
    if not is_deepseek_url(base_url):
        raise ValueError("LLM_BASE_URL 必须是 DeepSeek 官方接口")
    model = ChatOpenAI(
        model=model_name, api_key=api_key, base_url=base_url,
        timeout=30, max_retries=0, use_responses_api=False,
        extra_body={"thinking": {"type": "disabled"}},
    )
    today = datetime.now(CHINA_TIMEZONE).date().isoformat()
    unit_label = "华氏度（fahrenheit）" if default_unit == "fahrenheit" else "摄氏度（celsius）"
    prompt = (
        f"你是中文天气助手。当前中国日期为 {today}。"
        "任何涉及天气、气温、降水、风力或穿衣建议的问题，必须先调用 get_real_weather，"
        "绝不能根据记忆、常识或历史工具结果编造当前天气。普通闲聊可直接回答。"
        "追问省略城市时从对话历史确定城市；无法确定就询问用户。"
        "城市参数必须保留用户输入的完整地点层级，例如‘肇庆高要’不能只传‘肇庆’；"
        "工具会优先匹配区县，没有区县数据时才按工具返回的降级标记说明市级结果。"
        f"用户没有指定单位时使用 {unit_label}，并且每次工具调用都要显式传入 unit；"
        "明天 start_offset=1；未来3天 days=3、start_offset=0。"
        "用户一次提出多个城市或多个日期问题时，分别调用工具后再合并回答。"
        "工具返回 ok=false 时只解释错误，不得生成天气信息。"
        "工具成功时只能基于本轮工具 JSON 回答，并区分实时实况与未来预报。"
        "不要展示内部思维链；回答简洁。"
    )
    return create_agent(model=model, tools=[get_real_weather], system_prompt=prompt)


def parse_range(question: str) -> tuple[int, int]:
    match = re.search(r"(?:未来|接下来)?\s*(\d+|[一二两三四五六七八九十])\s*天", question)
    if match:
        raw = match.group(1)
        return 0, int(raw) if raw.isdigit() else DAY_COUNT_WORDS[raw]
    if "大后天" in question:
        return 3, 1
    if "后天" in question:
        return 2, 1
    if "明天" in question or "明日" in question:
        return 1, 1
    if "这几天" in question or "接下来" in question:
        return 0, 3
    return 0, 1


def parse_unit(question: str, default: str = "celsius") -> str:
    """识别用户单位偏好；没有明确单位时沿用页面选择。"""
    text = question or ""
    if re.search(r"华氏|℉|fahrenheit", text, re.IGNORECASE):
        return "fahrenheit"
    if re.search(r"摄氏|℃|celsius", text, re.IGNORECASE):
        return "celsius"
    return default if default in {"celsius", "fahrenheit"} else "celsius"


def parse_date_ranges(question: str) -> list[tuple[int, int]]:
    """把一句话中的多个日期要求拆成偏移量和天数，支持今天/明天/未来 N 天。"""
    pattern = re.compile(
        r"未来\s*(?:\d+|[一二两三四五六七八九十])\s*天|"
        r"接下来\s*(?:\d+|[一二两三四五六七八九十])\s*天|这几天|"
        r"大后天|后天|明天|明日|今天"
    )
    ranges: list[tuple[int, int]] = []
    for match in pattern.finditer(question or ""):
        item = parse_range(match.group(0))
        if item not in ranges:
            ranges.append(item)
    return ranges or [parse_range(question)]


def extract_city_candidates(question: str) -> list[str]:
    """提取显式城市，支持“广州和深圳”等多个城市表达。"""
    text = re.sub(r"\s+", "", question or "")
    text = text.strip("，。？！,!?/\\")
    text = re.sub(r"^(请问|帮我查一下|帮我查|查询|查一下|查查|看看|我想知道|告诉我|那|呢)+", "", text)
    text = re.sub(r"^(今天|明天|明日|后天|大后天|未来\d+天|接下来\d+天)", "", text)
    text = re.split(
        r"今天|明天|明日|后天|大后天|未来|接下来|这几天|\d+天|[一二两三四五六七八九十]天|天气|气温|温度|降雨|下雨|有雨|雨吗|下雪|晴|阴|多云|预报|气象|风力|风速|湿度|冷不冷|热不热|穿什么|带伞",
        text, maxsplit=1,
    )[0]
    # 处理“北京现在的天气”“广州的温度”等口语修饰词。
    text = re.sub(r"(?:现在|当前|目前|今日|今天|明天|明日|后天|大后天|会不会|是否|有没有|有无|会|可能|的|不)+$", "", text)
    text = re.sub(r"(分别|各自)$", "", text)
    parts = re.split(r"(?:和|及|以及|与|、|，|,|/|\\)", text)
    cities: list[str] = []
    for part in parts:
        city = normalize_city(part)
        if city and city not in cities:
            cities.append(city)
    return cities


def extract_city(question: str) -> str:
    """提取第一个显式城市，供兼容旧调用方使用。"""
    cities = extract_city_candidates(question)
    return cities[0] if cities else ""


def previous_city(previous: list[dict]) -> str:
    for item in reversed(previous):
        if item.get("role") == "assistant" and item.get("city"):
            return str(item["city"]).split("、", 1)[0]
    return ""


def previous_cities(previous: list[dict]) -> list[str]:
    """读取最近一次回答中的城市集合，支持多城市追问。"""
    for item in reversed(previous):
        if item.get("role") != "assistant" or not item.get("city"):
            continue
        cities = [normalize_city(part) for part in re.split(r"、|,|，", str(item["city"]))]
        return [city for city in cities if city]
    return []


def is_weather_request(question: str, previous: list[dict]) -> bool:
    return bool(WEATHER_WORDS.search(question)) or bool(previous_city(previous) and FOLLOW_UP_WORDS.search(question))


def resolve_city(question: str, previous: list[dict]) -> str:
    current = extract_city(question)
    if current and current not in ("那", "呢", "那呢", "怎么样", "如何"):
        return current
    return previous_city(previous) if FOLLOW_UP_WORDS.search(question) else ""


def resolve_cities(question: str, previous: list[dict]) -> list[str]:
    """解析本轮城市；没有显式城市时沿用上一轮的全部城市。"""
    current = extract_city_candidates(question)
    if current:
        return current
    return previous_cities(previous) if FOLLOW_UP_WORDS.search(question) else []


def weather_query_specs(
    question: str,
    previous: list[dict],
    default_unit: str = "celsius",
) -> list[dict[str, Any]]:
    """生成本轮需要核对的天气子任务，多个日期会拆成多次工具查询。"""
    cities = resolve_cities(question, previous)
    if not cities:
        return []
    unit = parse_unit(question, default_unit)
    return [
        {"city": city, "start_offset": start, "days": days, "unit": unit}
        for city in cities
        for start, days in parse_date_ranges(question)
    ]


def temperature_celsius(obj: Any) -> float | None:
    """读取和风天气温度字段，并统一转换为摄氏度。"""
    if not isinstance(obj, dict) or obj.get("value") is None:
        return None
    try:
        value = float(obj["value"])
    except (TypeError, ValueError):
        return None
    unit = str(obj.get("unit", "celsius")).lower()
    if unit in {"f", "°f", "℉", "fahrenheit"}:
        return (value - 32) * 5 / 9
    return value


def temperature_text(obj: Any, unit: str = "celsius") -> str:
    value = temperature_celsius(obj)
    if value is None:
        return ""
    if unit == "fahrenheit":
        value = value * 9 / 5 + 32
        return f"{round(value, 1):g}℉"
    return f"{round(value, 1):g}℃"


def percent_text(value: Any) -> str:
    """兼容 0.65 和 65 两种湿度/概率表达。"""
    if isinstance(value, dict):
        value = value.get("value")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return ""
    if 0 <= number <= 1:
        number *= 100
    return f"{round(number):g}%"


def advice_text(condition: str, temperature: float | None, precipitation: str, wind_scale: Any) -> str:
    """只根据 API 字段生成简单出行建议，避免模型凭空发挥。"""
    advice: list[str] = []
    try:
        if precipitation and float(precipitation.rstrip("%")) >= 60:
            advice.append("建议携带雨具")
    except (TypeError, ValueError):
        pass
    if temperature is not None and temperature <= 12:
        advice.append("建议添加外套")
    elif temperature is not None and temperature >= 32:
        advice.append("注意防暑和补水")
    try:
        if wind_scale is not None and float(wind_scale) >= 6:
            advice.append("风力较大，减少户外活动")
    except (TypeError, ValueError):
        pass
    if condition and any(word in condition for word in ("晴", "少云")):
        advice.append("外出注意防晒")
    return "；".join(advice)


def format_weather_payload(payload: dict[str, Any]) -> str:
    """最终天气文本从真实 JSON 确定性生成，不采信模型编出的数值。"""
    if not payload.get("ok"):
        message = payload.get("error", "天气查询失败，请稍后重试。")
        suggestions = payload.get("suggestions") or []
        return message + (f" 可尝试：{'、'.join(suggestions)}。" if suggestions else "")
    city_info = payload["city"]
    city = city_info["name"]
    requested = city_info.get("requested") or city
    match_level = city_info.get("match_level", "direct")
    location_note = ""
    if match_level == "district":
        parent = city_info.get("adm2") or city_info.get("adm1")
        if parent:
            location_note = f"（区县定位：{parent}）"
    elif match_level == "city_fallback":
        location_note = f"（未找到“{requested}”的区县数据，已降级为市级天气）"
    weather = payload["raw"]["weather"]
    query = payload["query"]
    unit = query.get("unit", "celsius")
    if query["kind"] == "current":
        parts = [f"{city}实时天气{location_note}（和风天气）"]
        condition = (weather.get("condition") or {}).get("text")
        temp = temperature_text(weather.get("temperature"), unit)
        feels = temperature_text(weather.get("feelsLike"), unit)
        if condition:
            parts.append(f"天气：{condition}")
        if temp:
            parts.append(f"气温：{temp}")
        if feels:
            parts.append(f"体感：{feels}")
        humidity = percent_text(weather.get("humidity"))
        if humidity:
            parts.append(f"湿度：{humidity}")
        wind = weather.get("wind") or {}
        if wind.get("scale") is not None:
            parts.append(f"风力：{wind['scale']} 级")
        probability = percent_text(weather.get("precipitationProbability") or weather.get("precipitation"))
        if probability:
            parts.append(f"降雨概率：{probability}")
        advice = advice_text(condition or "", temperature_celsius(weather.get("temperature")), probability, wind.get("scale"))
        if advice:
            parts.append(f"建议：{advice}")
        return "，".join(parts) + "。"
    start = query["start_offset"]
    selected = weather["days"][start:start + query["days"]]
    lines = [f"{city}天气预报{location_note}（和风天气，非实时实况）："]
    for day in selected:
        stamp = str(day.get("forecastStartTime", ""))
        fields = [stamp[:10] if stamp else "日期未知"]
        condition = (day.get("daytime") or {}).get("condition") or {}
        condition_text = condition.get("text") or ""
        if condition_text:
            fields.append(condition_text)
        high = temperature_text(day.get("temperatureMax"), unit)
        low = temperature_text(day.get("temperatureMin"), unit)
        if low or high:
            fields.append(f"{low or '未知'}～{high or '未知'}")
        probability = percent_text(day.get("precipitationProbability"))
        if probability:
            fields.append(f"降雨概率 {probability}")
        advice = advice_text(condition_text, temperature_celsius(day.get("temperatureMax")), probability, (day.get("windDay") or {}).get("scale"))
        if advice:
            fields.append(f"建议：{advice}")
        lines.append("- " + "，".join(fields))
    return "\n".join(lines)


def message_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(block.get("text", "") for block in content if isinstance(block, dict)).strip()
    return str(content)


def call_weather(
    city: str,
    days: int,
    start_offset: int,
    trace: list[dict],
    unit: str = "celsius",
) -> dict[str, Any]:
    args = {"city": city, "days": days, "start_offset": start_offset, "unit": unit}
    call_id = f"forced-weather-{len(trace)}"
    trace.append({"kind": "call", "name": "get_real_weather", "arguments": args, "call_id": call_id})
    raw = get_real_weather.invoke(args)
    trace.append({"kind": "result", "name": "get_real_weather", "content": raw, "call_id": call_id})
    return json.loads(raw)


def payload_matches_query(payload: dict[str, Any], city: str) -> bool:
    """防止多工具调用时把别的地点结果配到当前查询。"""
    if not payload.get("ok"):
        return True
    info = payload.get("city") or {}
    requested = info.get("requested")
    returned = info.get("name")
    if requested:
        return normalize_city(str(requested)) == normalize_city(city)
    return not returned or normalize_location_name(str(returned)) == normalize_location_name(city)


def verified_weather_answer(
    question: str,
    previous: list[dict],
    trace: list[dict],
    default_unit: str = "celsius",
) -> tuple[str, str]:
    """确认本轮每个城市/日期/单位；模型漏调时逐项强制补查。"""
    queries = weather_query_specs(question, previous, default_unit)
    city_names = list(dict.fromkeys(query["city"] for query in queries))
    city = "、".join(city_names)
    if not queries:
        return "请告诉我要查询哪个城市的天气。", ""

    answers: list[str] = []
    for query in queries:
        start_offset = query["start_offset"]
        days = query["days"]
        unit = query["unit"]
        if days < 1 or start_offset + days > MAX_DAYS:
            answers.append(f"目前仅支持从今天起未来 {MAX_DAYS} 天内的查询。")
            continue

        matched_payload: dict[str, Any] | None = None
        for index, step in enumerate(trace):
            if step.get("kind") != "call" or step.get("name") != "get_real_weather":
                continue
            args = step.get("arguments") or {}
            try:
                matches = (
                    normalize_city(str(args.get("city", ""))) == normalize_city(query["city"])
                    and int(args.get("days", 1)) == days
                    and int(args.get("start_offset", 0)) == start_offset
                    and str(args.get("unit", "celsius")).lower() == unit
                )
            except (TypeError, ValueError):
                matches = False
            if not matches:
                continue
            call_id = step.get("call_id")
            for result in trace[index + 1:]:
                if (
                    result.get("kind") == "result"
                    and result.get("name") == "get_real_weather"
                    and (not call_id or result.get("call_id") == call_id)
                ):
                    try:
                        candidate = json.loads(result["content"])
                        if payload_matches_query(candidate, query["city"]):
                            matched_payload = candidate
                    except (TypeError, json.JSONDecodeError):
                        matched_payload = None
                    break
            if matched_payload is not None:
                break

        if matched_payload is None:
            trace.append({
                "kind": "validation",
                "content": f"模型未调用正确的天气工具（{query['city']}，偏移 {start_offset} 天）；已强制执行查询。",
            })
            matched_payload = call_weather(query["city"], days, start_offset, trace, unit)
        answers.append(format_weather_payload(matched_payload))

    return "\n\n".join(answers), city


def run_agent(
    agent: Any,
    question: str,
    previous: list[dict],
    default_unit: str = "celsius",
) -> tuple[str, list[dict], str]:
    """执行一轮 Agent；天气类回答必须通过工具结果校验。"""
    history = []
    for item in previous[-MAX_HISTORY_MESSAGES:]:
        cls = HumanMessage if item["role"] == "user" else AIMessage
        history.append(cls(content=item["content"]))
    input_messages = [*history, HumanMessage(content=question)]
    result = agent.invoke({"messages": input_messages}, config={"recursion_limit": 12})
    trace: list[dict] = []
    final_answer = ""
    for step in result["messages"][len(input_messages):]:
        if isinstance(step, AIMessage):
            for call in step.tool_calls:
                trace.append({
                    "kind": "call", "name": call["name"], "arguments": call["args"],
                    "call_id": call.get("id"),
                })
            if not step.tool_calls:
                final_answer = message_to_text(step.content)
        elif isinstance(step, ToolMessage):
            trace.append({
                "kind": "result", "name": step.name or "get_real_weather",
                "content": message_to_text(step.content), "call_id": step.tool_call_id,
            })
    if is_weather_request(question, previous):
        answer, city = verified_weather_answer(question, previous, trace, default_unit)
        return answer, trace, city
    if not final_answer:
        raise RuntimeError("模型未返回可显示的最终回答")
    if not trace:
        trace.append({"kind": "direct", "content": "普通提问：模型直接回答，未调用天气工具。"})
    return final_answer, trace, ""


def describe_model_error(exc: Exception) -> str:
    status = getattr(exc, "status_code", None)
    if status == 401:
        return "DeepSeek 密钥无效或未授权（401）"
    if status == 402:
        return "DeepSeek 账户余额不足（402）"
    if status == 429:
        return "DeepSeek 请求频率超限（429）"
    return f"DeepSeek 暂时不可用（{type(exc).__name__}）"


def inject_styles() -> None:
    """读取项目内 CSS，集中美化页面；样式文件不存在时保持默认界面。"""
    css_path = Path(__file__).resolve().parent / "styles.css"
    try:
        css = css_path.read_text(encoding="utf-8")
    except OSError:
        return
    # Streamlit 1.57+ 推荐使用 st.html 注入显式请求的自定义 CSS。
    st.html(f"<style>{css}</style>")


def show_trace(trace: list[dict]) -> None:
    """展示可观察的工具调用与 API 原始 JSON，不展示隐藏思维链。"""
    with st.expander("Agent 决策与工具调用日志"):
        for index, step in enumerate(trace, start=1):
            if step["kind"] == "call":
                st.markdown(f"**{index}. 调用工具：{step['name']}**")
                st.json(step["arguments"])
            elif step["kind"] == "result":
                st.markdown(f"**{index}. API 原始返回：{step['name']}**")
                try:
                    st.json(json.loads(step["content"]))
                except (TypeError, json.JSONDecodeError):
                    st.code(step["content"], language="text")
            else:
                st.write(step["content"])


def answer_chunks(answer: str):
    """工具校验完成后才流式展示可信答案，不提前流出模型幻觉。"""
    for part in re.findall(r"[^，。\n]+[，。\n]?", answer):
        yield part


def main() -> None:
    """保留原有侧栏 + 对话区布局、会话记忆和推理面板。"""
    st.set_page_config(page_title="AI天气查询Agent", page_icon="🌤️")
    inject_styles()
    st.markdown(
        """
        <section class="weather-hero">
          <span class="weather-hero__sun" aria-hidden="true"></span>
          <p class="weather-hero__eyebrow">REAL-TIME WEATHER AGENT</p>
          <h1>AI天气查询Agent</h1>
          <p class="weather-hero__caption">
            用自然语言查询城市天气。Agent 会先调用和风天气，再基于实时数据回答，拒绝编造天气信息。
          </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "llm_error" not in st.session_state:
        st.session_state.llm_error = None

    with st.sidebar:
        st.markdown(
            '<div class="weather-sidebar-brand">'
            '<span class="weather-sidebar-brand__icon">🌤️</span>'
            '<span>天气 Agent Demo</span>'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="weather-sidebar-label">演示说明</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="weather-sidebar-note">'
            '试试：广州今天天气怎么样？<br>肇庆高要今天温度<br>肇庆市高要区明天会下雨吗<br><br>'
            '市、区、县均通过和风天气 GeoAPI 实时定位；区县无数据时提示市级降级。'
            '</div>',
            unsafe_allow_html=True,
        )
        unit_choice = st.selectbox("默认温度单位", ["摄氏度", "华氏度"], key="weather_unit_choice")
        default_unit = "fahrenheit" if unit_choice == "华氏度" else "celsius"
        if st.button("清空对话"):
            st.session_state.messages = []
            st.rerun()
        if st.session_state.llm_error and st.button("重试大模型"):
            st.session_state.llm_error = None
            st.rerun()

    llm_key, base_url, model_name, weather_key, weather_host = get_settings()
    base_url_ok = is_deepseek_url(base_url)
    mode_notice = st.empty()
    if not weather_key or not weather_host:
        mode_notice.warning("请在 demo/.env 填写 QWEATHER_KEY 和 QWEATHER_API_HOST；未配置时不会生成天气数据。")
    elif not llm_key or not base_url_ok or st.session_state.llm_error:
        mode_notice.info("大模型暂不可用；天气问题仍将直接调用和风天气，不生成模拟数据。")
    else:
        mode_notice.success(f"当前为 DeepSeek Agent + 和风天气模式：{model_name}")

    if not st.session_state.messages:
        st.markdown(
            '<div class="weather-empty">'
            '<div class="weather-empty__title">从一个城市开始吧</div>'
            '<div>支持市级、区县级天气、未来预报和连续追问。</div>'
            '<div class="weather-empty__examples">例如：高要区今天会下雨吗？ · 那明天呢？</div>'
            '</div>',
            unsafe_allow_html=True,
        )

    for item in st.session_state.messages:
        with st.chat_message(item["role"]):
            st.markdown(item["content"])
            if item.get("trace"):
                show_trace(item["trace"])

    question = st.chat_input("输入天气问题或普通问题…")
    if not question:
        return
    previous = st.session_state.messages.copy()
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Agent 正在处理…"):
            try:
                if llm_key and base_url_ok and not st.session_state.llm_error and weather_key and weather_host:
                    agent = create_weather_agent(llm_key, base_url, model_name, default_unit)
                    answer, trace, city = run_agent(agent, question, previous, default_unit)
                elif is_weather_request(question, previous):
                    trace = [{"kind": "direct", "content": "大模型不可用；执行强制天气工具查询。"}]
                    answer, city = verified_weather_answer(question, previous, trace, default_unit)
                else:
                    answer, trace, city = "大模型当前不可用；请先完成配置或稍后重试。", [], ""
            except Exception as exc:
                st.session_state.llm_error = describe_model_error(exc)
                mode_notice.warning(st.session_state.llm_error + "；天气问题将继续尝试真实天气 API。")
                if is_weather_request(question, previous):
                    trace = [{"kind": "validation", "content": st.session_state.llm_error + "；直接调用天气工具。"}]
                    answer, city = verified_weather_answer(question, previous, trace, default_unit)
                else:
                    answer, trace, city = st.session_state.llm_error, [], ""
        st.write_stream(answer_chunks(answer))
        if trace:
            show_trace(trace)
    st.session_state.messages.append({
        "role": "assistant", "content": answer, "trace": trace,
        "city": city, "unit": default_unit,
    })


if __name__ == "__main__":
    main()
