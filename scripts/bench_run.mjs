#!/usr/bin/env node
// 脱敏基准集评测脚本（模型侧指标）
//
// 用法：
//   node scripts/bench_run.mjs                      # 运行全部有截图的 case
//   node scripts/bench_run.mjs --case chat-email    # 仅运行指定 id（子串匹配）
//   node scripts/bench_run.mjs --base benchmarks/results/run-xxx.json   # 与历史结果对比
//   BAODOU_LLAMA_URL=http://127.0.0.1:8765/v1/chat/completions node scripts/bench_run.mjs
//
// 记录：首个可读句时延、最终结果时延、首 token 时延、请求总耗时、usage token、跳过率占位、
// 关键事实命中率、禁止词命中、重复运行的同义抖动次数。结果为 JSON + CSV（benchmarks/results/）。
//
// 应用侧指标（截图编码、prefill/首 token、生成、UI）由运行中的 Baodou 写入
// RuntimeSnapshot.metrics；本脚本专注于“同一基准图”下的模型端可复现比较。

import {
  existsSync,
  readFileSync,
  writeFileSync,
  mkdirSync,
  statSync,
} from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const MANIFEST_PATH = join(ROOT, "benchmarks", "cases", "manifest.json");
const RESULTS_DIR = join(ROOT, "benchmarks", "results");

const args = process.argv.slice(2);
function flag(name, fallback) {
  const index = args.indexOf(name);
  if (index === -1) return fallback;
  return args[index + 1];
}
const onlyCase = flag("--case", "");
const baselinePath = flag("--base", "");
const endpoint =
  flag("--endpoint", "") ||
  process.env.BAODOU_LLAMA_URL ||
  "http://127.0.0.1:8765/v1/chat/completions";

const PROMPT = [
  "/no_think",
  "你是 Baodou，一位常驻用户桌面的本地 AI 伴侣，气质温暖、沉稳、敏锐，善于观察但绝不虚构看不见的内容。此消息附有一张刚截取的屏幕图片，请直接观察图片作答。用户当前关注：描述当前屏幕上的关键可见内容",
  "回答要求：",
  "1. 第一句给出当前画面上最重要且最明确的可见状态或内容。",
  "2. 第二句补充 1–2 项与当前关注点直接相关的信息。",
  "3. 只依据这一张截图作答；单张截图无法观察到时间上的“变化”，不要声称看到过程或前后的变化。",
  "4. 模糊、小字、图标或数字无法确认时，如实说明“看不清/无法确认”，严禁补全、猜测或推测用户的意图。",
  "5. 禁止给出点击、输入、打开、关闭等操作步骤，禁止输出坐标或 ACTION。",
  "最多两句、短而完整的中文摘要。",
  "当前截图：",
].join("\n");

function normalize(text) {
  return text
    .replace(/[\s，。！？；：、,.!?;:"'“”‘’（）()…·~\-_—]+/g, "")
    .trim();
}

function hasReadableBoundary(text) {
  return /[。！？!?；;\n…]/.test(text) || text.trim().length >= 24;
}

function toBase64(buffer) {
  return Buffer.from(buffer).toString("base64");
}

async function runOnce(caseData, imageBase64) {
  const started = Date.now();
  let firstTokenMs = -1;
  let firstReadableMs = -1;
  let finalMs = -1;
  let text = "";
  let promptTokens = null;
  let completionTokens = null;
  let httpStatus = null;
  let error = null;

  const payload = {
    model: caseData.model ?? "local-vision",
    temperature: 0.1,
    max_tokens: 160,
    cache_prompt: true,
    stream: true,
    reasoning_format: "none",
    chat_template_kwargs: { enable_thinking: false },
    messages: [
      {
        role: "user",
        content: [
          { type: "text", text: PROMPT },
          { type: "image_url", image_url: { url: `data:image/png;base64,${imageBase64}` } },
        ],
      },
    ],
  };

  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    httpStatus = response.status;
    if (!response.ok || !response.body) {
      error = `HTTP ${response.status} ${response.statusText}`;
    } else {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";
      for (;;) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        let lineEnd;
        while ((lineEnd = buffer.indexOf("\n")) !== -1) {
          const line = buffer.slice(0, lineEnd).trim();
          buffer = buffer.slice(lineEnd + 1);
          if (!line.startsWith("data:")) continue;
          const data = line.slice(5).trim();
          if (data === "[DONE]") {
            if (finalMs === -1) finalMs = Date.now() - started;
            continue;
          }
          let chunk;
          try {
            chunk = JSON.parse(data);
          } catch {
            continue;
          }
          const delta =
            chunk?.choices?.[0]?.delta?.content ??
            chunk?.choices?.[0]?.delta?.text ??
            "";
          if (delta) {
            if (firstTokenMs === -1) firstTokenMs = Date.now() - started;
            text += delta;
            if (firstReadableMs === -1 && hasReadableBoundary(text)) {
              firstReadableMs = Date.now() - started;
            }
          }
          if (chunk?.usage) {
            promptTokens = chunk.usage.prompt_tokens ?? promptTokens;
            completionTokens = chunk.usage.completion_tokens ?? completionTokens;
          }
        }
      }
      if (finalMs === -1) finalMs = Date.now() - started;
    }
  } catch (cause) {
    error = String(cause);
  }

  return {
    text: text.trim(),
    firstTokenMs,
    firstReadableMs,
    finalMs,
    totalMs: finalMs === -1 ? Date.now() - started : finalMs,
    promptTokens,
    completionTokens,
    httpStatus,
    error,
  };
}

function evaluate(caseData, output) {
  const facts = (caseData.expectedKeyFacts ?? []).filter(
    (fact) => !/^TODO/.test(fact),
  );
  const forbidden = caseData.forbidden ?? [];
  const factsHit = facts.filter((fact) => output.text.includes(fact)).length;
  const forbiddenHits = forbidden.filter((word) => output.text.includes(word));
  const lowInfo = /看不清|不清晰|无法确认|无法辨认|模糊/.test(output.text);
  return { factsTotal: facts.length, factsHit, forbiddenHits, lowInfo };
}

function unstablePairs(results) {
  let jitter = 0;
  for (let i = 1; i < results.length; i += 1) {
    if (normalize(results[i].text) !== normalize(results[i - 1].text)) jitter += 1;
  }
  return jitter > 0 ? jitter : results.length > 1 ? 0 : 0;
}

const manifest = JSON.parse(readFileSync(MANIFEST_PATH, "utf8"));
const cases = manifest.cases.filter(
  (item) => !onlyCase || item.id.includes(onlyCase),
);

const missing = [];
const results = [];
for (const caseData of cases) {
  const shotPath = resolve(join(dirname(MANIFEST_PATH), caseData.screenshotPath));
  if (!existsSync(shotPath)) {
    missing.push(caseData.id);
    continue;
  }
  process.stdout.write(`running ${caseData.id} (x${caseData.repeat}) ... `);
  const imageBase64 = toBase64(readFileSync(shotPath));
  const runs = [];
  for (let i = 0; i < (caseData.repeat ?? 1); i += 1) {
    runs.push(await runOnce(caseData, imageBase64));
  }
  const firstReasonable = runs.find((r) => r.finalMs !== -1 && !r.error) ?? runs[0];
  const evalResult = evaluate(caseData, firstReasonable);
  const jitter = unstablePairs(runs);
  const failed = runs.filter((r) => r.error || !r.text).length;
  results.push({
    id: caseData.id,
    category: caseData.category,
    screenshot: caseData.screenshotPath,
    ok: failed === 0,
    firstTokenMs: firstReasonable.firstTokenMs,
    firstReadableMs: firstReasonable.firstReadableMs,
    finalMs: firstReasonable.finalMs,
    totalMs: firstReasonable.totalMs,
    promptTokens: firstReasonable.promptTokens,
    completionTokens: firstReasonable.completionTokens,
    factsHit: evalResult.factsHit,
    factsTotal: evalResult.factsTotal,
    forbiddenHits: evalResult.forbiddenHits,
    lowInfo: evalResult.lowInfo,
    jitter,
    repeats: runs.length,
    failedRequests: failed,
    error: firstReasonable.error ?? null,
    lastOutput: firstReasonable.text.slice(0, 120),
  });
  process.stdout.write(`OK=${failed === 0}, firstToken=${firstReasonable.firstTokenMs}ms, total=${firstReasonable.totalMs}ms\n`);
}

const stamp = new Date().toISOString().replace(/[:.]/g, "-");
const output = {
  generatedAt: new Date().toISOString(),
  endpoint,
  promptHash: undefined,
  missing,
  results,
};
const jsonPath = join(RESULTS_DIR, `run-${stamp}.json`);
mkdirSync(RESULTS_DIR, { recursive: true });
writeFileSync(jsonPath, JSON.stringify(output, null, 2));

function csvEscape(value) {
  return typeof value === "string" ? `"${value.replace(/"/g, '""')}"` : String(value);
}
const header = [
  "id", "category", "ok", "firstTokenMs", "firstReadableMs", "finalMs", "totalMs",
  "promptTokens", "completionTokens", "factsHit", "factsTotal", "forbiddenHits",
  "lowInfo", "jitter", "repeats", "failedRequests", "error",
];
const rows = [header.join(",")];
for (const r of results) {
  rows.push(header.map((key) => csvEscape(r[key])).join(","));
}
const csvPath = join(RESULTS_DIR, `run-${stamp}.csv`);
writeFileSync(csvPath, rows.join("\n"));

// --- 控制台汇总 ---
console.log("\n== summary ==");
console.log(`endpoint: ${endpoint}`);
console.log(`cases: ${results.length} run, ${missing.length} missing (screenshots needed)`);
if (results.length) {
  const firstTokens = results.map((r) => r.firstTokenMs).filter((v) => v >= 0);
  const finals = results.map((r) => r.finalMs).filter((v) => v >= 0);
  const facts = results.filter((r) => r.factsTotal > 0);
  const factsRate = facts.length
    ? facts.reduce((sum, r) => sum + r.factsHit / r.factsTotal, 0) / facts.length
    : null;
  const forbiddenRate = results.filter((r) => r.forbiddenHits.length > 0).length;
  const jitterSum = results.reduce((sum, r) => sum + r.jitter, 0);
  if (firstTokens.length) {
    console.log(`avg firstToken: ${(firstTokens.reduce((a, b) => a + b, 0) / firstTokens.length).toFixed(0)}ms`);
  }
  if (finals.length) {
    console.log(`avg final: ${(finals.reduce((a, b) => a + b, 0) / finals.length).toFixed(0)}ms`);
  }
  if (factsRate != null) {
    console.log(`key-fact hit rate: ${(factsRate * 100).toFixed(0)}% (${facts.length} cases with defined facts)`);
  }
  console.log(`cases with forbidden words: ${forbiddenRate}`);
  console.log(`total jitter across repeats: ${jitterSum}`);
}

if (baselinePath) {
  const baseline = JSON.parse(readFileSync(baselinePath, "utf8"));
  console.log("\n== vs baseline ==");
  const byId = new Map(baseline.results.map((r) => [r.id, r]));
  for (const r of results) {
    const before = byId.get(r.id);
    if (!before) continue;
    const delta = (name) => {
      const a = before[name];
      const b = r[name];
      return !Number.isFinite(a) || !Number.isFinite(b) ? "n/a" : `${(b - a).toFixed(0)}ms`;
    };
    console.log(
      `${r.id.padEnd(22)} firstToken ${delta("firstTokenMs").padStart(10)} final ${delta("finalMs").padStart(10)}`,
    );
  }
}

if (missing.length) {
  console.log(`\nmissing screenshots (add to benchmarks/cases/artifacts/):\n  ${missing.join("\n  ")}`);
}
console.log(`\nwrote ${jsonPath}`);
console.log(`wrote ${csvPath}`);
