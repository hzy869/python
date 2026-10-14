"""AI 天气查询 Agent（Streamlit 网页版）。

数据来源
--------
真实天气数据来自 Open-Meteo 免费预报接口（见 weather_api.py），
不再使用按哈希生成的模拟数据。

大模型（可选）
--------------
在 .env 中配置 LLM_API_KEY 后，走 LangChain Tool Calling Agent；
未配置时使用内置的本地 Agent（规则解析 + 真实天气工具），页面交互完全一致。
两种模式都一定会调用 query_weather 工具取真实数据，不允许编造天气。
"""

import json
import os
import re
from types import SimpleNamespace

import streamlit as st
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

import weather_api

# 项目根目录运行 Streamlit 时，.env 可以被正确加载
load_dotenv(override=True)


# ---------------------------------------------------------------------------
# 工具：真实天气查询
# ---------------------------------------------------------------------------

@tool
def query_weather(city: str, days: int = 1, start_offset: int = 0) -> str:
    """查询指定城市的真实天气预报。

    参数说明：
    - city: 城市名称，例如“广州”“肇庆”“杭州”
    - days: 需要查询的天数，默认为 1，最多 16 天
    - start_offset: 从今天起往后偏移几天开始查，默认 0 表示从今天开始；
      查“明天”传 1，查“后天”传 2

    返回 JSON 数组，包含每天的日期、星期、天气、最高温、最低温、湿度、
    风力、降水概率等真实预报字段。调用方不得修改或编造返回值。
    """
    try:
        city_info = weather_api.resolve_city(city)
        if not city_info:
            return json.dumps(
                {
                    "error": (
                        f"未找到城市：{city}。"
                        "请换一个城市名试试，例如“广州”“杭州”，或者输入更完整的名称。"
                    )
                },
                ensure_ascii=False,
            )

        forecast = weather_api.fetch_forecast(
            city_info,
            start_offset=start_offset,
            days=days,
        )
        payload = {
            "城市": city_info["name"],
            "地区": city_info["admin"],
            "数据源": "Open-Meteo 真实预报",
            "当前实况": forecast.get("current"),
            "预报": forecast["items"],
        }
        return json.dumps(payload, ensure_ascii=False)
    except Exception as exc:
        return json.dumps(
            {"error": f"天气查询失败：{exc}。请稍后重试。"},
            ensure_ascii=False,
        )


# ---------------------------------------------------------------------------
# 自然语言解析
# ---------------------------------------------------------------------------

WEATHER_KEYWORDS = re.compile(
    r"天气|气温|温度|降雨|下雨|雨|雪|湿度|风力|风速|冷|热|闷|潮|穿|伞|"
    r"紫外线|防晒|雾|霾|台风|晴|阴|多云|结冰|体感"
)

FOLLOWUP_HINT = re.compile(
    r"那|呢|未来|明天|明日|后天|大后天|最近|接下来|这几天|近几|"
    r"\d+\s*天|几\s*天|一周|一星期|这周|本周"
)


def is_weather_question(text: str, has_city_context: bool = False) -> bool:
    """判断用户输入是否属于天气查询问题。

    has_city_context 为 True 时（前文已经出现过城市），
    “那未来 5 天呢”这类省略主语的追问也算天气问题。
    """
    text = text or ""
    if WEATHER_KEYWORDS.search(text):
        return True
    if has_city_context and FOLLOWUP_HINT.search(text):
        return True
    return False


def extract_city_from_text(text: str) -> str:
    """从用户输入中提取城市名，返回标准城市名，提取不到返回空串。"""
    text = text or ""
    if not text:
        return ""

    # 1) 俗称 / 别名，例如“羊城”“鹏城”
    for alias, city in weather_api.CITY_ALIASES.items():
        if alias in text:
            return city

    # 2) 本地城市表，长名优先，避免“吉林”和“吉林市”这类包含关系出错
    for city in sorted(weather_api.list_supported_cities(), key=len, reverse=True):
        if city in text:
            return city

    # 3) “XX市 / XX区 / XX县”这类显式写法
    for match in re.finditer(r"([\u4e00-\u9fa5]{2,8}?(?:市|区|县|州))", text):
        candidate = match.group(1)
        resolved = weather_api.resolve_city(candidate)
        if resolved and resolved.get("source") != "online":
            return resolved["name"]

    # 4) “XX天气 / XX温度”这类紧邻写法
    for match in re.finditer(
        r"([\u4e00-\u9fa5]{2,6}?)(?:的)?(?:今天|明天|后天|大后天|未来|最近|接下来)?"
        r"(?:天气|气温|温度|降雨|下雨|湿度|风力|冷不冷|热不热)",
        text,
    ):
        candidate = match.group(1)
        resolved = weather_api.resolve_city(candidate)
        if resolved:
            return resolved["name"]

    return ""


def parse_time_range(text: str) -> tuple[int, int, str] | None:
    """解析时间范围，解析不到返回 None。"""
    text = text or ""
    if not text:
        return None
    offset, days, label = weather_api.time_range_from_text(text)
    if label == "今天" and not re.search(r"今天|今日|现在|现时|此刻", text):
        # time_range_from_text 在没有任何时间词时会返回默认的“今天”，
        # 这里区分“用户明确说了今天”和“用户什么都没说”。
        has_explicit = re.search(
            r"明天|明日|后天|大后天|未来|最近|接下来|这几天|近几|一周|一星期|这周|本周|"
            r"\d+\s*天|几\s*天",
            text,
        )
        if not has_explicit:
            return None
    return offset, days, label


def detect_focus(text: str) -> str:
    """识别用户关心的重点，用于组织答案的侧重点。"""
    text = text or ""
    if re.search(r"雨|雪|带伞|伞|降水|会不会下", text):
        return "rain"
    if re.search(r"紫外线|防晒|晒", text):
        return "uv"
    if re.search(r"风力|风大|大风", text):
        return "wind"
    if re.search(r"穿|冷不冷|冷吗|热不热|热吗|衣服|外套", text):
        return "clothing"
    return "general"


# ---------------------------------------------------------------------------
# 把工具返回的数据整理成自然语言
# ---------------------------------------------------------------------------

def parse_weather_payload(observations: list[str]) -> tuple[list[dict], list[str], dict]:
    """从工具返回里解析出预报列表、错误信息、当前实况。"""
    items: list[dict] = []
    errors: list[str] = []
    current: dict = {}

    for observation in observations:
        try:
            data = json.loads(observation)
        except (TypeError, json.JSONDecodeError):
            continue

        if isinstance(data, dict):
            if data.get("error"):
                errors.append(str(data["error"]))
                continue
            current = data.get("当前实况") or current
            for item in data.get("预报") or []:
                if isinstance(item, dict) and item.get("城市"):
                    items.append(item)

        if isinstance(data, list):
            for item in data:
                if isinstance(item, dict) and item.get("城市"):
                    items.append(item)
                elif isinstance(item, dict) and item.get("error"):
                    errors.append(str(item["error"]))

    return items, errors, current


def _to_int(text: str) -> int | None:
    match = re.search(r"-?\d+", str(text or ""))
    return int(match.group()) if match else None


def weather_advice(item: dict, focus: str = "general") -> str:
    """根据天气数据给出生活提示，让回答不像在念表格。"""
    tips: list[str] = []
    condition = str(item.get("天气", ""))
    pop = _to_int(item.get("降水概率"))
    high = _to_int(item.get("最高温"))
    low = _to_int(item.get("最低温"))
    humidity = _to_int(item.get("湿度"))
    uv = item.get("紫外线")

    if "雨" in condition or "雪" in condition or (pop is not None and pop >= 50):
        tips.append("有降水，出门记得带伞")
    elif pop is not None and 25 <= pop < 50:
        tips.append("有零星降水的可能，可以备一把伞")

    if high is not None:
        if high >= 33:
            tips.append("白天很热，注意防暑补水")
        elif high >= 28:
            tips.append("白天偏热，注意防晒")

    if low is not None:
        if low <= 5:
            tips.append("早晚很冷，务必保暖")
        elif low <= 12:
            tips.append("早晚偏凉，建议加件外套")
    if high is not None and low is not None and high - low >= 12:
        tips.append("昼夜温差较大，注意增减衣物")

    if isinstance(uv, (int, float)) and uv >= 7:
        tips.append("紫外线较强，做好防晒")

    if humidity is not None and humidity >= 85:
        tips.append("湿度偏高，体感闷湿")

    if focus == "rain" and not any("伞" in tip or "降水" in tip for tip in tips):
        tips.append("今天降水概率不大，不用特意带伞")

    if not tips:
        return ""
    return "；".join(tips[:3]) + "。"


def uv_text(value) -> str:
    """把紫外线指数翻译成人话。"""
    if not isinstance(value, (int, float)):
        return "未知"
    if value <= 2:
        return f"{int(value)}（很弱）"
    if value <= 5:
        return f"{int(value)}（中等）"
    if value <= 7:
        return f"{int(value)}（较强）"
    if value <= 10:
        return f"{int(value)}（很强）"
    return f"{int(value)}（极强）"


def weather_advice_multi(items: list[dict]) -> str:
    """多天预报的生活提示：重点说哪天要带伞、整体冷暖。"""
    tips: list[str] = []

    rainy = [
        item
        for item in items
        if "雨" in str(item.get("天气", ""))
        or "雪" in str(item.get("天气", ""))
        or (_to_int(item.get("降水概率")) or 0) >= 50
    ]
    if rainy:
        marks = "、".join(item["日期"][5:] for item in rainy[:3])
        more = "等" if len(rainy) > 3 else ""
        tips.append(f"{marks}{more} 有降水，记得带伞")
    else:
        tips.append("这几天降水概率都不高，一般不用带伞")

    highs = [v for v in (_to_int(item.get("最高温")) for item in items) if v is not None]
    lows = [v for v in (_to_int(item.get("最低温")) for item in items) if v is not None]

    if highs and lows:
        hottest, coldest = max(highs), min(lows)
        if hottest >= 33:
            tips.append(f"最高可能到 {hottest}°C，注意防暑补水")
        if coldest <= 5:
            tips.append(f"最低可能到 {coldest}°C，注意保暖")
        elif coldest <= 12:
            tips.append(f"最低约 {coldest}°C，早晚需要加外套")
        if hottest - coldest >= 12:
            tips.append("温差较大，注意增减衣物")

    return "；".join(tips[:3]) + "。"


def summarize_weather_items(
    items: list[dict],
    current: dict | None = None,
    focus: str = "general",
) -> str:
    """根据工具原始数据生成不会被模型篡改的天气摘要。"""
    if not items:
        return "没有查询到天气数据。"

    city = items[0]["城市"]
    area = items[0].get("地区") or ""
    header_city = f"{city}（{area}）" if area and area != city else city

    if len(items) == 1:
        item = items[0]
        icon = item.get("天气图标", "")
        lines = [
            f"{header_city} {item['日期']} {item.get('星期', '')}："
            f"{icon} {item['天气']}，{item['最低温']} ~ {item['最高温']}，"
            f"湿度 {item['湿度']}，风力 {item['风力']}，"
            f"降水概率 {item.get('降水概率', '未知')}，紫外线 {uv_text(item.get('紫外线'))}。"
        ]

        if current:
            lines.append(
                f"实况：当前 {current.get('温度', '未知')}，"
                f"体感 {current.get('体感温度', '未知')}，"
                f"{current.get('天气', '')}，湿度 {current.get('湿度', '未知')}。"
            )

        advice = weather_advice(item, focus)
        if advice:
            lines.append(f"小提示：{advice}")
        return "\n\n".join(lines)

    lines = [f"{header_city} 未来 {len(items)} 天预报（真实数据）："]
    for item in items:
        extra = f"，紫外线 {uv_text(item.get('紫外线'))}" if focus == "uv" else ""
        lines.append(
            f"- {item['日期']} {item.get('星期', '')}：{item.get('天气图标', '')} {item['天气']}，"
            f"{item['最低温']} ~ {item['最高温']}，湿度 {item['湿度']}，"
            f"风力 {item['风力']}，降水概率 {item.get('降水概率', '未知')}{extra}"
        )

    advice = weather_advice_multi(items)
    if advice:
        lines.append("")
        lines.append(f"小提示：{advice}")
    return "\n".join(lines)


def validate_answer_against_tool_data(
    answer: str, observations: list[str]
) -> tuple[str, str]:
    """对比模型回答和工具原始数据。

    返回 (警告信息, 修正后的答案)。如果模型回答中的关键数值缺失，
    则优先使用工具原始数据生成答案。
    """
    items, errors, current = parse_weather_payload(observations)

    if errors:
        warning = "工具返回异常，已停止使用模型天气结论，以下为工具原始错误信息。"
        return warning, "\n\n".join(errors)

    if not items:
        return "", answer

    normalized_answer = re.sub(r"\s+", "", answer)
    missing_values = []
    for item in items:
        for key in ("天气", "最高温", "最低温", "湿度", "风力", "降水概率"):
            value = str(item.get(key, ""))
            normalized_value = re.sub(r"\s+", "", value)
            if value and normalized_value not in normalized_answer:
                missing_values.append(value)

    if missing_values:
        warning = "检测到模型输出与工具原始数据不一致，已优先使用工具原始数据。"
        return warning, summarize_weather_items(items, current)

    return "", answer


# ---------------------------------------------------------------------------
# 本地 Agent（无 LLM_API_KEY 时使用）
# ---------------------------------------------------------------------------

GREETING_KEYWORDS = re.compile(r"你好|您好|hi|hello|hey|在吗|哈喽", re.IGNORECASE)
CAPABILITY_KEYWORDS = re.compile(r"你是谁|你叫什么|能做什么|会什么|干什么|有什么功能|介绍一下你|帮助")
CITY_LIST_KEYWORDS = re.compile(r"支持.*城市|哪些城市|城市列表|能查.*城市")


class LocalWeatherAgent:
    """不依赖大模型的本地 Agent。

    用规则解析自然语言，但天气数据同样来自真实的 Open-Meteo 预报接口，
    并且同样会真实调用 query_weather 工具，推理日志与 LLM 模式完全一致。
    """

    def __init__(self) -> None:
        self.last_city: str = ""
        self.last_offset: int = 0
        self.last_days: int = 1

    # -- 入口 ---------------------------------------------------------------
    def invoke(self, inputs: dict) -> dict:
        user_input = (inputs.get("input") or "").strip()
        history = inputs.get("chat_history", [])
        history_text = " ".join(
            str(message.content)
            for message in history
            if isinstance(message, HumanMessage)
        )

        if not user_input:
            return self._reply("请告诉我你想查询哪个城市的天气，例如：广州今天天气怎么样？")

        if CITY_LIST_KEYWORDS.search(user_input):
            return self._reply(self._city_list_answer())

        if CAPABILITY_KEYWORDS.search(user_input):
            return self._reply(self._capability_answer())

        if GREETING_KEYWORDS.search(user_input) and not is_weather_question(user_input):
            return self._reply(
                "你好！我可以查询全国 100+ 个城市的真实天气，"
                "数据来自 Open-Meteo 实时预报。\n\n"
                "你可以这样问：\n"
                "- 广州今天天气怎么样？\n"
                "- 肇庆未来 3 天会下雨吗？\n"
                "- 杭州明天要穿外套吗？\n"
                "- 上海这周温度多少？"
            )

        has_city_context = bool(self.last_city)
        if not is_weather_question(user_input, has_city_context=has_city_context):
            return self._reply(
                "我主要负责查天气。你可以告诉我城市和时间，例如："
                "“肇庆未来 3 天天气”“杭州明天冷不冷”。"
            )

        return self._handle_weather(user_input, history_text)

    # -- 天气处理 -----------------------------------------------------------
    def _handle_weather(self, user_input: str, history_text: str) -> dict:
        city_in_input = extract_city_from_text(user_input)
        city = city_in_input
        if not city and history_text:
            city = extract_city_from_text(history_text)
        if not city:
            city = self.last_city

        if not city:
            return self._reply(
                "好的，请告诉我城市名，例如“广州”“肇庆”“杭州”，"
                "我就能查到它的真实天气。"
            )

        time_range = parse_time_range(user_input)
        if time_range is None:
            if city_in_input or not self.last_city:
                # 本轮明确说了城市，或没有上文可继承：默认查今天
                time_range = (0, 1, "今天")
            else:
                # 真正省略主语的追问（“那呢？”“再来一次”），沿用上一次的时间范围
                time_range = (
                    self.last_offset,
                    self.last_days,
                    self._label(self.last_offset, self.last_days),
                )

        offset, days, label = time_range

        city_info = weather_api.resolve_city(city)
        if not city_info:
            return self._reply(
                f"抱歉，我没能找到城市“{city}”。换个写法试试？"
                "例如“广州”“杭州”，或者带上省份，如“广东肇庆”。"
            )

        tool_input = {
            "city": city_info["name"],
            "days": days,
            "start_offset": offset,
        }
        observation = query_weather.invoke(tool_input)
        action = SimpleNamespace(tool="query_weather", tool_input=tool_input)

        items, errors, current = parse_weather_payload([observation])

        if errors:
            output = "\n\n".join(errors)
        elif items:
            self.last_city = city_info["name"]
            self.last_offset = offset
            self.last_days = days
            focus = detect_focus(user_input)
            output = summarize_weather_items(items, current if offset == 0 else None, focus)
        else:
            output = "没有查询到天气数据。"

        return {
            "output": output,
            "intermediate_steps": [(action, observation)],
        }

    # -- 文案 ---------------------------------------------------------------
    @staticmethod
    def _label(offset: int, days: int) -> str:
        if days == 1:
            return {0: "今天", 1: "明天", 2: "后天", 3: "大后天"}.get(offset, f"{offset} 天后")
        return f"未来 {days} 天"

    @staticmethod
    def _city_list_answer() -> str:
        cities = weather_api.list_supported_cities()
        preview = "、".join(cities[:24])
        return (
            f"内置城市表已收录 {len(cities)} 个中国城市，例如：{preview} 等。\n\n"
            "表里没有的城市我也会去在线地名库现查，所以直接输入城市名就行。"
        )

    @staticmethod
    def _capability_answer() -> str:
        return (
            "我是天气查询 Agent，能力包括：\n"
            "- 查任意中国城市的真实天气（数据来自 Open-Meteo 实时预报）\n"            "- 支持“今天 / 明天 / 后天 / 未来 N 天 / 这周”等时间表达\n"
            "- 支持“羊城、鹏城、魔都”这类俗称，也能听懂“要不要带伞”“冷不冷”这类问法\n"
            "- 每一轮都会真实调用天气工具，并在右侧展示调用日志"
        )

    @staticmethod
    def _reply(text: str) -> dict:
        return {"output": text, "intermediate_steps": []}


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------

def create_llm() -> ChatOpenAI:
    """根据环境变量创建 OpenAI 兼容的大模型客户端。"""
    api_key = os.getenv("LLM_API_KEY")
    base_url = os.getenv("LLM_BASE_URL")
    model_name = os.getenv("LLM_MODEL") or "gpt-4o-mini"

    if not api_key:
        raise RuntimeError(
            "未检测到 LLM_API_KEY。请在项目根目录的 .env 文件中填写你的 API Key。"
        )

    llm_kwargs = {"model": model_name, "temperature": 0}
    llm_kwargs["api_key"] = api_key
    if base_url:
        llm_kwargs["base_url"] = base_url

    return ChatOpenAI(**llm_kwargs)


def build_agent() -> AgentExecutor:
    """创建带天气查询工具的 Agent。"""
    llm = create_llm()
    tools = [query_weather]

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "你是一个严格的 AI 天气查询 Agent。"
                "所有天气、温度、降雨、湿度、风力问题都必须调用 query_weather 工具。"
                "用户说“明天”时 start_offset 传 1，“后天”传 2，“未来 N 天”时 days 传 N。"
                "严禁编造、推测、修改或补充工具返回的天气数据，"
                "温度、天气、湿度、风力、降水概率必须原样使用。"
                "如果工具返回城市不存在、参数为空或接口异常，必须如实告知用户错误信息，"
                "不得继续编造天气结果。回答要自然、简洁、使用中文，"
                "可以基于工具给出的数据补充一句生活提示（如是否需要带伞、如何穿衣）。",
            ),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    agent = create_tool_calling_agent(llm=llm, tools=tools, prompt=prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


def format_thinking_log(intermediate_steps) -> str:
    """把 Agent 的工具调用过程整理成面试演示可读的日志。"""
    if not intermediate_steps:
        return "未调用工具，Agent 直接回答。"

    lines = []
    for index, (action, observation) in enumerate(intermediate_steps, start=1):
        tool_name = getattr(action, "tool", "unknown_tool")
        tool_input = getattr(action, "tool_input", "")
        lines.append(f"{index}. 调用工具：{tool_name}")
        lines.append(f"   参数：{tool_input}")
        lines.append(f"   返回：{observation}")

    lines.append(f"{len(intermediate_steps) + 1}. 根据工具结果生成最终回答")
    return "\n".join(lines)


class StreamHandler(BaseCallbackHandler):
    """捕获大模型输出的 token，并实时更新 Streamlit 占位符。"""

    def __init__(self, placeholder) -> None:
        self.placeholder = placeholder
        self.answer_parts: list[str] = []
        self.first_call_buffer: list[str] = []
        self.llm_call_count = 0
        self.saw_tool = False
        self.streaming = False

    def on_llm_start(self, serialized, prompts, **kwargs) -> None:
        self.llm_call_count += 1
        self.streaming = self.llm_call_count > 1
        if not self.streaming:
            self.first_call_buffer = []

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        if self.streaming:
            self.answer_parts.append(token)
            self.placeholder.markdown("".join(self.answer_parts))
        else:
            # 第一次 LLM 调用可能只是生成工具调用参数，先缓存，不直接展示
            self.first_call_buffer.append(token)

    def on_tool_start(self, serialized, input_str, **kwargs) -> None:
        self.saw_tool = True
        self.first_call_buffer = []
        self.answer_parts = []
        self.streaming = True

    def on_llm_end(self, response, **kwargs) -> None:
        if self.llm_call_count == 1 and not self.saw_tool:
            self.answer_parts = list(self.first_call_buffer)
            self.placeholder.markdown("".join(self.answer_parts))

    @property
    def has_output(self) -> bool:
        return bool(self.answer_parts)


# ---------------------------------------------------------------------------
# 页面
# ---------------------------------------------------------------------------

EXAMPLES = [
    "广州今天天气怎么样？",
    "肇庆未来 3 天会下雨吗？",
    "杭州明天要穿外套吗？",
    "上海这周温度多少？",
]

EMPTY_HINT = (
    "输入任意城市 + 时间就能查询，例如“广州今天天气怎么样”。"
    "支持“明天 / 后天 / 未来 5 天 / 这周”等说法，也支持“羊城”“魔都”这类俗称。"
)


def initialize_state() -> None:
    """初始化 Streamlit 会话状态。"""
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "display_messages" not in st.session_state:
        st.session_state.display_messages = []

    if "reasoning_logs" not in st.session_state:
        st.session_state.reasoning_logs = []

    if "pending_question" not in st.session_state:
        st.session_state.pending_question = ""

    config_signature = (
        bool(os.getenv("LLM_API_KEY")),
        os.getenv("LLM_BASE_URL", ""),
        os.getenv("LLM_MODEL", ""),
    )

    if st.session_state.get("llm_config_signature") != config_signature:
        try:
            if config_signature[0]:
                st.session_state.agent_executor = build_agent()
                st.session_state.agent_mode = "llm"
            else:
                st.session_state.agent_executor = LocalWeatherAgent()
                st.session_state.agent_mode = "local"
            st.session_state.agent_error = ""
        except Exception as exc:
            st.session_state.agent_executor = None
            st.session_state.agent_error = str(exc)
            st.session_state.agent_mode = "error"

        st.session_state.llm_config_signature = config_signature


def clear_conversation() -> None:
    """清空当前会话中的全部对话和推理日志。"""
    st.session_state.chat_history = []
    st.session_state.display_messages = []
    st.session_state.reasoning_logs = []
    executor = st.session_state.get("agent_executor")
    if isinstance(executor, LocalWeatherAgent):
        executor.last_city = ""


def render_sidebar() -> None:
    """渲染侧边栏，提供标题、配置状态和清空对话功能。"""
    with st.sidebar:
        st.title("AI天气查询Agent")
        st.caption("真实天气数据 + 工具调用，展示 Agent 推理过程。")
        st.divider()

        api_key = os.getenv("LLM_API_KEY")
        base_url = os.getenv("LLM_BASE_URL")
        model_name = os.getenv("LLM_MODEL") or "gpt-4o-mini"

        if api_key:
            st.success("大模型已配置")
            st.caption(f"模型：{model_name}")
            st.caption(f"接口：{base_url or 'OpenAI 默认地址'}")
        else:
            st.info("本地解析模式：未配置大模型，用规则解析 + 真实天气工具")

        st.caption(f"城市库：{len(weather_api.list_supported_cities())} 个")
        st.caption("数据源：Open-Meteo 真实预报")

        st.divider()
        st.markdown("**试试这样问**")
        for example in EXAMPLES:
            if st.button(example, use_container_width=True, key=f"sidebar_{example}"):
                st.session_state.pending_question = example
                st.rerun()

        st.divider()
        if st.button("一键清空对话", type="primary", use_container_width=True):
            clear_conversation()
            st.rerun()


def render_chat_message(message: dict) -> None:
    """渲染单条聊天消息，区分用户、Agent 和工具消息。"""
    role = message["role"]
    avatars = {"user": "🧑", "assistant": "🤖", "tool": "🛠️"}
    with st.chat_message(role, avatar=avatars.get(role)):
        st.markdown(message["content"])


def render_reasoning_panel(placeholder) -> None:
    """在右侧栏渲染 Agent 推理日志和工具原始 JSON。"""
    placeholder.empty()
    panel = placeholder.container()

    with panel:
        st.subheader("Agent 推理日志")
        logs = st.session_state.reasoning_logs

        if not logs:
            st.info("暂无推理日志，发送天气问题后会自动记录。")
            return

        for item in reversed(logs[-8:]):
            st.markdown(f"**用户提问：** {item['question']}")
            st.code(item["thinking"], language="text")

            observations = item.get("tool_observations", [])
            if observations:
                st.markdown("**工具原始返回 JSON：**")
                for observation in observations:
                    try:
                        json_data = json.loads(observation)
                    except (TypeError, json.JSONDecodeError):
                        json_data = observation
                    st.json(json_data)

            if item.get("warning"):
                st.warning(item["warning"])
            st.divider()


def apply_chat_styles() -> None:
    """注入聊天区域和侧边栏的轻量样式。"""
    st.markdown(
        """
        <style>
        [data-testid="stChatMessage"] {
            border-radius: 12px;
            border: 1px solid rgba(128, 128, 128, 0.18);
            padding: 0.35rem 0.55rem;
        }
        [data-testid="stChatMessage"] > div:first-child {
            gap: 0.35rem;
        }
        [data-testid="stSidebar"] {
            background: rgba(248, 250, 252, 0.72);
        }
        div[data-testid="stCodeBlock"] {
            border-radius: 8px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def run_agent(executor, user_input: str, stream_placeholder, agent_mode: str) -> dict:
    """执行 LLM Agent 或本地 Agent，并返回结果。"""
    inputs = {
        "input": user_input,
        "chat_history": list(st.session_state.chat_history),
    }

    if agent_mode == "llm":
        handler = StreamHandler(stream_placeholder)
        result = executor.invoke(inputs, {"callbacks": [handler]})
    else:
        handler = None
        result = executor.invoke(inputs)

    return {"result": result, "handler": handler}


def main() -> None:
    """Streamlit 页面主流程。"""
    st.set_page_config(page_title="AI天气查询Agent", page_icon="🌤️", layout="wide")
    initialize_state()
    apply_chat_styles()
    render_sidebar()

    st.title("AI天气查询Agent")
    st.caption("用自然语言查询真实天气，Agent 会自动判断是否需要调用工具。")

    if st.session_state.agent_error:
        st.error(f"Agent 初始化失败：{st.session_state.agent_error}")

    left_col, right_col = st.columns([3, 2], gap="medium")
    reasoning_area = right_col.empty()
    render_reasoning_panel(reasoning_area)

    with left_col:
        if not st.session_state.display_messages:
            st.info(EMPTY_HINT)
        for message in st.session_state.display_messages:
            render_chat_message(message)

    user_input = st.chat_input("请输入天气问题，例如：广州今天天气怎么样")
    if not user_input and st.session_state.pending_question:
        user_input = st.session_state.pending_question
        st.session_state.pending_question = ""
    if not user_input:
        return

    user_input = user_input.strip()
    if not user_input:
        st.warning("输入内容不能为空，请重新输入。")
        return

    st.session_state.display_messages.append({"role": "user", "content": user_input})
    with left_col:
        render_chat_message({"role": "user", "content": user_input})

    executor = st.session_state.agent_executor
    agent_mode = st.session_state.get("agent_mode", "local")

    if executor is None:
        with left_col:
            with st.chat_message("assistant", avatar="🤖"):
                st.error("Agent 尚未成功初始化，请检查 .env 配置后刷新页面。")
        return

    with left_col:
        tool_area = st.container()
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Agent 思考中..."):
                stream_placeholder = st.empty()
                result = None

                try:
                    run_result = run_agent(
                        executor=executor,
                        user_input=user_input,
                        stream_placeholder=stream_placeholder,
                        agent_mode=agent_mode,
                    )
                    result = run_result["result"]
                    handler = run_result["handler"]
                    answer = result["output"]
                    thinking = format_thinking_log(result.get("intermediate_steps", []))
                    tool_observations = [
                        str(observation)
                        for _, observation in result.get("intermediate_steps", [])
                    ]
                    forced_tool_warning = ""

                    # 兜底约束：天气问题必须经过天气工具，防止模型绕过工具直接回答
                    has_history_city = bool(
                        extract_city_from_text(
                            " ".join(
                                message["content"]
                                for message in st.session_state.display_messages[:-1]
                                if message["role"] == "user"
                            )
                        )
                    )
                    if (
                        not tool_observations
                        and is_weather_question(user_input, has_city_context=has_history_city)
                    ):
                        city = extract_city_from_text(user_input)
                        offset, days, _ = parse_time_range(user_input) or (0, 1, "今天")
                        if city:
                            tool_input = {
                                "city": city,
                                "days": days,
                                "start_offset": offset,
                            }
                            observation = query_weather.invoke(tool_input)
                            action = SimpleNamespace(
                                tool="query_weather",
                                tool_input=tool_input,
                            )
                            result["intermediate_steps"] = [(action, observation)]
                            tool_observations = [observation]
                            thinking = format_thinking_log(result["intermediate_steps"])
                            forced_tool_warning = (
                                "检测到模型未调用天气工具，已强制调用工具获取真实数据。"
                            )

                    warning, corrected_answer = validate_answer_against_tool_data(
                        answer,
                        tool_observations,
                    )
                    if forced_tool_warning:
                        warning = "\n".join(
                            part for part in (forced_tool_warning, warning) if part
                        )
                    if corrected_answer:
                        answer = corrected_answer

                    stream_placeholder.markdown(answer)

                    if warning:
                        st.warning(warning)

                    st.session_state.display_messages.append(
                        {"role": "assistant", "content": answer}
                    )
                    st.session_state.reasoning_logs.append(
                        {
                            "question": user_input,
                            "thinking": thinking,
                            "tool_observations": tool_observations,
                            "warning": warning,
                        }
                    )
                    st.session_state.chat_history.extend(
                        [HumanMessage(content=user_input), AIMessage(content=answer)]
                    )
                except Exception as exc:
                    error_message = f"请求失败：{exc}"
                    st.error(error_message)
                    st.session_state.display_messages.append(
                        {"role": "assistant", "content": error_message}
                    )
                    st.session_state.reasoning_logs.append(
                        {
                            "question": user_input,
                            "thinking": f"异常信息：\n{exc}",
                            "tool_observations": [],
                            "warning": "",
                        }
                    )

    with left_col:
        with tool_area:
            if result is not None:
                for action, observation in result.get("intermediate_steps", []):
                    tool_name = getattr(action, "tool", "unknown_tool")
                    tool_input = getattr(action, "tool_input", "")
                    tool_message = (
                        f"调用工具：`{tool_name}`\n\n"
                        f"参数：`{tool_input}`\n\n"
                        f"返回：\n```json\n{observation}\n```"
                    )
                    st.session_state.display_messages.append(
                        {"role": "tool", "content": tool_message}
                    )
                    render_chat_message({"role": "tool", "content": tool_message})

    render_reasoning_panel(reasoning_area)


if __name__ == "__main__":
    main()
