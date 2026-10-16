# AI 天气查询 Agent

一个带「工具调用 + 推理日志」的 AI 天气查询 Agent，用作个人作品集与面试现场演示。

**两个版本，同一套能力：**

| 版本 | 入口 | 大模型 | 说明 |
|---|---|---|---|
| 网页版（推荐演示） | `web/index.html` → <https://weather-agent.app.workbuddy.host/> | 本地规则引擎（默认，`USE_CLOUD_LLM = false`）；改成 `true` 即用 WorkBuddy 云端模型 | 浏览器端 Agent，工具调用用 JavaScript 实现，已发布上线 |
| 本地版 | `app.py`（Streamlit） | 可选，未配置则用本地规则引擎 | Python + LangChain，方便讲代码 |

两个版本的天气数据都来自 **Open-Meteo 真实预报接口**，不再是模拟数据。

## 核心特点

- **真实天气数据**：接入 Open-Meteo 免费预报接口，无需付费 Key，支持未来 16 天
- **真实的工具调用**：Agent 先调用 `query_weather` 工具拿数据，再根据数据组织回答
- **看得见的推理过程**：右侧「Agent 推理日志」展示工具参数和工具原始返回 JSON
- **自然语言理解**：支持「明天 / 后天 / 未来 N 天 / 这周」等时间表达，支持「羊城」「魔都」等俗称，也听得懂「要不要带伞」「冷不冷」这类问法
- **多轮对话**：追问「那未来 5 天呢？」会沿用上一轮的城市
- **城市覆盖广**：内置中国城市和部分区县的精确坐标表；区县输入优先匹配，表外区县自动走在线地名库，找不到区县时降级到上级市并提示
- **不编造数据**：工具返回值会被二次校验，模型输出与工具数据不一致时，以工具数据为准

## 项目结构

```text
weather/
├── web/
│   ├── index.html          # 网页版：UI + Agent 循环 + JS 版天气工具
│   └── city_coords.json    # 城市坐标表
├── app.py                  # Streamlit 版：页面 + Agent + 自然语言解析
├── weather_api.py          # 真实天气数据层（Open-Meteo + 城市解析）
├── city_coords.json        # 城市坐标表（Python 版共用）
├── requirements.txt
└── README.md
```

## 网页版（云端模型开关）

`web/index.html` 顶部有一个总开关：

```js
var USE_CLOUD_LLM = false;   // false = 关闭云端大模型；true = 使用 WorkBuddy 云端模型
```

- **默认 `false`**：完全关闭云端模型调用，不初始化云端 SDK、不消耗云服务资源点，全部回答由页面内的本地规则引擎生成（真实天气数据仍走 Open-Meteo）。
- **改成 `true`**：使用 WorkBuddy 云服务的 LLM 能力，**免密钥**，但只能在应用的正式域名下调用（服务端有 Origin 校验），因此需要通过发布才能使用。

Agent 循环在浏览器里执行：

```text
用户提问
  → 调用云端模型（stream: true，携带 query_weather 工具定义）
  → 模型返回 tool_calls
  → 浏览器执行 JS 版 query_weather（区县优先解析，请求 Open-Meteo）
  → 把工具结果作为 tool 消息回灌给模型
  → 模型生成最终回答
```

云端模型不可用时（未发布、网络异常、模型未就绪），页面会自动切换到本地规则引擎完成查询，并在推理日志里明确标注，不会静默失败。

## 本地版（Streamlit）

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run app.py
```

默认走本地规则引擎（不需要任何 API Key）。若要接大模型，在 `.env` 中填写：

```dotenv
LLM_API_KEY=你的API Key
LLM_BASE_URL=你的兼容接口地址
LLM_MODEL=gpt-4o-mini
```

配置后自动切换为 LangChain Tool Calling Agent，其余功能不变。

## 数据来源

天气预报：[Open-Meteo](https://open-meteo.com/) 免费预报接口（无需 API Key，支持未来 16 天）。
天气代码（WMO）与风速蒲福风级在 `weather_api.py` / `index.html` 中做了中文映射。

## 演示建议

1. 「广州今天天气怎么样？」
2. 「肇庆高要今天温度？」——展示区县定位与工具调用日志
3. 「肇庆未来 3 天会下雨吗？」——展示多天预报
4. 「那未来 5 天呢？」——展示多轮记忆
5. 「杭州明天要穿外套吗？」——展示生活化提问
6. 展开右侧推理日志，讲工具调用链路

## 注意事项

- 天气数据来自第三方公共接口，仅供演示，请勿用于生产决策。
- `.env` 已被 `.gitignore` 忽略，请勿提交真实密钥。
