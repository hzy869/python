# AI天气查询Agent

Streamlit + LangChain + DeepSeek + 和风天气的作品集 Demo。天气问题调用 `get_real_weather`，数据来自和风天气 GeoAPI、实时天气或每日预报；不包含模拟天气数据。

页面样式集中在 `styles.css`：浅蓝天气主题、响应式顶部品牌区、侧栏说明卡、聊天气泡、输入框和工具调用日志样式均通过该文件维护。

## 功能

- 自然语言查询：“广州今天天气怎么样”“肇庆高要未来3天温度”“那明天呢”。
- 支持城市别名/常见错别字（如“羊城”“广洲”）、市/区/县组合名称和同一句多城市查询；日期、单位会拆成独立工具任务逐项核对。
- 地点解析优先使用区县级 GeoAPI 结果；区县没有可用数据时自动回退到上级市，并在回答中明确提示。
- DeepSeek 负责工具选择；天气回答经本地校验，只由本轮 API JSON 生成，避免温度幻觉。
- DeepSeek 不可用时仍尝试直接调用真实天气工具，不回退到模拟数值。
- 页面保留侧栏、对话记忆、逐段流式展示和 Agent 工具日志；日志包含 GeoAPI 与天气 API 原始 JSON。
- 城市不存在、网络超时、鉴权或接口错误均给出明确提示，不生成虚假天气。

## 安装与启动

需要 Python 3.10+。在 `demo/` 目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

复制 `.env.example` 为 `.env`，仅在本机填入：

```dotenv
LLM_API_KEY=你的DeepSeek密钥
LLM_BASE_URL="https://api.deepseek.com/v1"
LLM_MODEL=deepseek-flash
QWEATHER_KEY=你的和风天气API_KEY
QWEATHER_API_HOST=控制台设置中的专属域名
```

`QWEATHER_API_HOST` 是类似 `abcxyz.qweatherapi.com` 的账号专属域名，不能填旧的公共域名。密钥通过 `X-QW-Api-Key` 请求头发送；`.env` 已被 Git 忽略。和风天气账号需开通 GeoAPI、实时天气和每日预报相应服务。今天单日查询显示实时实况；明天、多日查询显示预报（最多 10 天），不会将预报称为实时观测。

## Agent 原理

```text
用户问题 + Streamlit 会话历史
          ↓
DeepSeek 判断并调用 get_real_weather
          ↓
GeoAPI 定位省市区县 → 区县优先，必要时回退市级 → 实时天气 / 每日预报 API
          ↓
核对本轮城市与日期参数；必要时强制补查
          ↓
只从工具 JSON 生成天气回答并逐段展示
```

ReAct 在这里是“决定动作 → 调用工具 → 观察结果 → 继续回答”的循环。推理面板展示的是工具调用和 API 结果，不是模型隐藏思维链。面试时可以强调：即使模型跳过工具或编造温度，最终天气文本仍由经校验的真实 API 数据生成；API 失败就明确报错，不给出天气数值。

## 区县级测试用例

| 输入 | 预期结果 |
| --- | --- |
| `肇庆高要今天的天气` | GeoAPI 使用 `location=高要`、`adm=肇庆`，结果标记为区县定位。 |
| `肇庆市高要区明天会下雨吗` | 识别“肇庆市 + 高要区”，调用预报接口，回答中保留明天的日期。 |
| `高要区未来3天温度` | 直接匹配高要区；返回每日预报，不把预报称为实时实况。 |
| `肇庆高要`（区县无数据的模拟响应） | 自动查询上级市肇庆，结果提示“已降级为市级天气”。 |
| `肇庆不存在区今天天气` | 返回“未找到地点”，不输出温度、降雨等编造数据。 |
| `广州和高要今天怎么样` | 拆成两个独立天气工具任务，并分别校验城市与工具结果 ID。 |
| 先问 `肇庆高要天气`，再问 `那明天呢` | 会话记忆沿用上一次区县地点，第二轮查询 `start_offset=1`。 |

参考：[和风天气 API Host](https://dev.qweather.com/docs/configuration/api-host/)、[城市搜索](https://dev.qweather.com/docs/api/geoapi/city-lookup/)、[实时天气](https://dev.qweather.com/docs/api/weather/weather-current/)、[每日预报](https://dev.qweather.com/docs/api/weather/weather-daily-forecast/)。
