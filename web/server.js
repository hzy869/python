/**
 * AI 天气查询 Agent - 轻量服务端
 *
 * 职责：
 * 1. 托管静态页面（index.html / city_coords.json）
 * 2. 代理 Open-Meteo 天气与地名接口
 *
 * 为什么需要代理：浏览器直连 api.open-meteo.com 在部分网络环境下不可达，
 * 由服务器代取可以把「天气数据」变成同源请求，和云端模型一样稳定。
 *
 * 仅使用 Node 标准库，无任何第三方依赖。
 */

const http = require("http");
const https = require("https");
const fs = require("fs");
const path = require("path");

const PORT = parseInt(process.env.PORT || "3000", 10);
const ROOT = __dirname;
const HOST_FORECAST = "api.open-meteo.com";
const HOST_GEOCODE = "geocoding-api.open-meteo.com";
const USER_AGENT = "ai-weather-agent-demo/1.0";
const UPSTREAM_TIMEOUT_MS = 12000;
const CACHE_TTL_MS = 120 * 1000;

const MIME = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".json": "application/json; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".ico": "image/x-icon"
};

const cache = new Map();

function cacheGet(key) {
  const entry = cache.get(key);
  if (!entry) return null;
  if (Date.now() - entry.t > CACHE_TTL_MS) {
    cache.delete(key);
    return null;
  }
  return entry.v;
}

function cacheSet(key, value) {
  if (cache.size > 300) cache.clear();
  cache.set(key, { t: Date.now(), v: value });
}

function sendJson(res, status, payload) {
  const body = JSON.stringify(payload);
  res.writeHead(status, {
    "Content-Type": "application/json; charset=utf-8",
    "Content-Length": Buffer.byteLength(body),
    "Access-Control-Allow-Origin": "*",
    "Cache-Control": "no-store"
  });
  res.end(body);
}

function fetchUpstreamJson(host, targetPath, callback) {
  const request = https.request(
    {
      host: host,
      path: targetPath,
      method: "GET",
      headers: { "User-Agent": USER_AGENT, Accept: "application/json" },
      timeout: UPSTREAM_TIMEOUT_MS
    },
    function (upstream) {
      let raw = "";
      upstream.setEncoding("utf8");
      upstream.on("data", function (chunk) { raw += chunk; });
      upstream.on("end", function () {
        try {
          callback(null, JSON.parse(raw));
        } catch (err) {
          callback(new Error("上游返回的不是合法 JSON"));
        }
      });
    }
  );
  request.on("timeout", function () { request.destroy(new Error("上游接口超时")); });
  request.on("error", function (err) { callback(err); });
  request.end();
}

function proxyJson(res, host, targetPath, cacheKey) {
  const hit = cacheGet(cacheKey);
  if (hit) {
    sendJson(res, 200, hit);
    return;
  }
  fetchUpstreamJson(host, targetPath, function (err, data) {
    if (err) {
      sendJson(res, 502, { error: "天气数据源请求失败：" + err.message });
      return;
    }
    if (data && data.error) {
      sendJson(res, 502, { error: "天气数据源返回错误：" + (data.reason || "unknown") });
      return;
    }
    cacheSet(cacheKey, data);
    sendJson(res, 200, data);
  });
}

function serveApi(res, pathname, query) {
  if (pathname === "/api/forecast") {
    const lat = parseFloat(query.get("lat"));
    const lon = parseFloat(query.get("lon"));
    const days = Math.min(16, Math.max(1, parseInt(query.get("forecast_days") || "7", 10) || 7));
    if (!isFinite(lat) || !isFinite(lon)) {
      sendJson(res, 400, { error: "缺少或非法的 lat / lon 参数" });
      return;
    }
    const target =
      "/v1/forecast?" +
      new URLSearchParams({
        latitude: String(lat),
        longitude: String(lon),
        daily: "weather_code,temperature_2m_max,temperature_2m_min,precipitation_probability_max,relative_humidity_2m_mean,wind_speed_10m_max,uv_index_max",
        current: "temperature_2m,apparent_temperature,relative_humidity_2m,weather_code,wind_speed_10m",
        timezone: "Asia/Shanghai",
        forecast_days: String(days)
      }).toString();
    proxyJson(res, HOST_FORECAST, target, "forecast:" + lat.toFixed(3) + "," + lon.toFixed(3) + ":" + days);
    return;
  }

  if (pathname === "/api/geocode") {
    const name = (query.get("name") || "").trim();
    if (!name) {
      sendJson(res, 400, { error: "缺少 name 参数" });
      return;
    }
    const target =
      "/v1/search?" +
      new URLSearchParams({
        name: name,
        count: "10",
        language: query.get("language") || "zh",
        format: "json"
      }).toString();
    proxyJson(res, HOST_GEOCODE, target, "geocode:" + name + ":" + query.get("language"));
    return;
  }

  if (pathname === "/api/health") {
    sendJson(res, 200, { ok: true, time: new Date().toISOString() });
    return;
  }

  sendJson(res, 404, { error: "未知接口：" + pathname });
}

function serveStatic(res, pathname) {
  let relative = decodeURIComponent(pathname);
  if (relative === "/" || relative === "") relative = "/index.html";

  const filePath = path.normalize(path.join(ROOT, relative));
  if (filePath !== ROOT && filePath.indexOf(ROOT) !== 0) {
    res.writeHead(403, { "Content-Type": "text/plain; charset=utf-8" });
    res.end("Forbidden");
    return;
  }

  fs.readFile(filePath, function (err, buffer) {
    if (err) {
      res.writeHead(404, { "Content-Type": "text/plain; charset=utf-8" });
      res.end("Not Found");
      return;
    }
    const ext = path.extname(filePath).toLowerCase();
    const headers = {
      "Content-Type": MIME[ext] || "application/octet-stream",
      "Cache-Control": ext === ".html" ? "no-cache" : "public, max-age=300"
    };
    res.writeHead(200, headers);
    res.end(buffer);
  });
}

const server = http.createServer(function (req, res) {
  const parsed = new URL(req.url, "http://" + (req.headers.host || "localhost"));
  const pathname = parsed.pathname;

  if (pathname.indexOf("/api/") === 0) {
    serveApi(res, pathname, parsed.searchParams);
    return;
  }
  serveStatic(res, pathname);
});

server.listen(PORT, "0.0.0.0", function () {
  console.log("weather agent server listening on 0.0.0.0:" + PORT);
});
