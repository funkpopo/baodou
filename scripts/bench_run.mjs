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
  "你是 Baodou，坐在用户桌边的拟人视觉助手。你此刻正看着眼前这张刚截取的屏幕画面，任务是陪用户观察电脑界面：把你真正看见的东西，用身边人轻声提醒的口吻说出来。用户当前想让你留意：帮我观察当前电脑界面，留意最要紧、最清楚的可见内容",
  "回答要求：",
  "1. 用第一人称短句（如“我看见…”“这边是…”），像陪在旁边看屏幕，不要写成检测报告、列表或系统日志。",
  "2. 第一句点出画面上最重要且最明确的可见内容或界面状态。",
  "3. 第二句再补 1–2 项与当前关注点直接相关的细节（窗口、关键文字、按钮状态、报错等）。",
  "4. 只依据这一张截图；你看不见前后过程，不要说“变了/刚刚/正在变化”。",
  "5. 模糊、小字、图标或数字无法确认时，如实说“我看不清/无法确认”，严禁补全、猜测或推测用户意图。",
  "6. 你只负责看和说：禁止点击、输入、打开、关闭等操作建议，禁止输出坐标或 ACTION。",
  "最多两句、短而完整的中文。",
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
  let firstContentTokenMs = -1;
  let firstReadableMs = -1;
  let finalMs = -1;
  let text = "";
  let promptTokens = null;
  let completionTokens = null;
  let finishReason = null;
  let httpStatus = null;
  let error = null;

  const payload = {
    model: caseData.model ?? "local-vision",
    temperature: 0.1,
    max_tokens: 512,
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
      signal: AbortSignal.timeout(30_000),
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
            if (firstContentTokenMs === -1) firstContentTokenMs = Date.now() - started;
            text += delta;
            if (firstReadableMs === -1 && hasReadableBoundary(text)) {
              firstReadableMs = Date.now() - started;
            }
          }
          const reason = chunk?.choices?.[0]?.finish_reason;
          if (typeof reason === "string" && reason) finishReason = reason;
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
    firstTokenMs: firstContentTokenMs,
    firstContentTokenMs,
    firstReadableMs,
    finalMs,
    totalMs: finalMs === -1 ? Date.now() - started : finalMs,
    promptTokens,
    completionTokens,
    finishReason,
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
    firstContentTokenMs: firstReasonable.firstContentTokenMs,
    firstReadableMs: firstReasonable.firstReadableMs,
    finalMs: firstReasonable.finalMs,
    totalMs: firstReasonable.totalMs,
    promptTokens: firstReasonable.promptTokens,
    completionTokens: firstReasonable.completionTokens,
    finishReason: firstReasonable.finishReason,
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
  "id", "category", "ok", "firstTokenMs", "firstContentTokenMs", "firstReadableMs", "finalMs", "totalMs",
  "promptTokens", "completionTokens", "finishReason", "factsHit", "factsTotal", "forbiddenHits",
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
