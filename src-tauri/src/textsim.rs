//! Lightweight text-analysis helpers for recognition results.
//!
//! These power P1 semantic de-duplication (normalization, character n-gram
//! Jaccard similarity, key-fact extraction) and P3 result grading
//! (readability classification, status-extraction and contradiction
//! detection).  Everything here is intentionally dependency free and cheap to
//! run on every recognition round.

use serde::{Deserialize, Serialize};
use std::collections::HashSet;

/// Words that carry status / state semantics.  A change in any of these
/// between two similar-looking results is worth surfacing to the user.
pub const STATUS_WORDS: &[&str] = &[
    "成功",
    "失败",
    "完成",
    "已完成",
    "错误",
    "异常",
    "停止",
    "已停止",
    "运行中",
    "正在",
    "加载中",
    "等待",
    "就绪",
    "无响应",
    "未响应",
    "已连接",
    "已断开",
    "已打开",
    "已关闭",
    "打开",
    "关闭",
    "新建",
    "删除",
    "保存",
    "开始",
    "结束",
    "审核中",
    "已提交",
    "收到",
    "发送",
];

/// Pairs that are mutually exclusive.  Finding members of one pair across two
/// consecutive results means the model is uncertain / scene changed a lot.
pub const CONTRADICTION_PAIRS: &[(&str, &str)] = &[
    ("成功", "失败"),
    ("完成", "失败"),
    ("已打开", "已关闭"),
    ("打开", "关闭"),
    ("运行中", "已停止"),
    ("运行中", "无响应"),
    ("已连接", "已断开"),
    ("正在", "已停止"),
];

/// Removes whitespace, punctuation and other tokens that carry no meaning for
/// similarity comparison.  Digits, error codes and status words survive.
pub fn normalize(text: &str) -> String {
    text.chars()
        .filter(|c| {
            !c.is_whitespace()
                && !matches!(
                    *c,
                    '，' | '。'
                        | '！'
                        | '？'
                        | '；'
                        | '：'
                        | '、'
                        | ','
                        | '.'
                        | '!'
                        | '?'
                        | ';'
                        | ':'
                        | '-'
                        | '_'
                        | '—'
                        | '…'
                        | '·'
                        | '"'
                        | '\''
                        | '“'
                        | '”'
                        | '（'
                        | '）'
                        | '('
                        | ')'
                        | '['
                        | ']'
                        | '{'
                        | '}'
                        | '/'
                        | '\\'
                        | '|'
                        | '<'
                        | '>'
                        | '`'
                        | '@'
                        | '#'
                        | '$'
                        | '%'
                        | '^'
                        | '&'
                        | '*'
                        | '='
                        | '+'
                )
        })
        .collect()
}

/// Builds the set of character n-grams for a text.
///
/// Exposed for the benchmark harness and for later SimHash-style tuning of the
/// exact similarity metric; the live dedup path uses [`unigram_containment`].
#[allow(dead_code)]
pub fn char_ngrams(text: &str, n: usize) -> HashSet<String> {
    let chars: Vec<char> = text.chars().collect();
    if chars.is_empty() {
        return HashSet::new();
    }
    if chars.len() <= n {
        return [chars.into_iter().collect::<String>()]
            .into_iter()
            .collect();
    }
    chars
        .windows(n)
        .map(|w| w.iter().collect::<String>())
        .collect()
}

/// Jaccard similarity of the character n-gram sets of two normalized texts.
#[allow(dead_code)]
pub fn ngram_jaccard(a: &str, b: &str, n: usize) -> f64 {
    let aset = char_ngrams(a, n);
    let bset = char_ngrams(b, n);
    let union = aset.union(&bset).count();
    if union == 0 {
        return 0.0;
    }
    (aset.intersection(&bset).count() as f64) / (union as f64)
}

/// Fraction of `inner`'s distinct characters that also appear in `outer`.
/// More robust than n-gram Jaccard for short Chinese paraphrases: two phrasings
/// of the same fact share most characters, while a genuinely new fact (new
/// number / code / status) introduces characters the previous result lacked.
pub fn unigram_containment(inner: &str, outer: &str) -> f64 {
    let inner_set: HashSet<char> = inner.chars().collect();
    let outer_set: HashSet<char> = outer.chars().collect();
    if inner_set.is_empty() {
        return 0.0;
    }
    inner_set.iter().filter(|c| outer_set.contains(c)).count() as f64 / inner_set.len() as f64
}

/// Extracts key facts: numbers, times, dates, versions and error-code-like
/// tokens.  A token is kept only when it looks like a concrete fact, never a
/// helper word.
pub fn key_facts(text: &str) -> Vec<String> {
    let chars: Vec<char> = text.chars().collect();
    let n = chars.len();
    let mut facts = Vec::new();
    let mut i = 0;
    while i < n {
        let c = chars[i];
        if i + 1 < n && c == '0' && chars[i + 1] == 'x' {
            // hex error codes / ids: 0x1f4a
            let mut j = i + 2;
            while j < n && chars[j].is_ascii_hexdigit() {
                j += 1;
            }
            let token: String = chars[i..j].iter().collect();
            if token.chars().count() > 2 {
                facts.push(token);
            }
            i = j;
        } else if c.is_ascii_digit() {
            let mut j = i;
            // digits with separators, e.g. 12 / 3.5 / 12:30 / 2024-05-01 / 1,000
            let mut last_sep = false;
            while j < n {
                let d = chars[j];
                if d.is_ascii_digit() {
                    last_sep = false;
                    j += 1;
                } else if matches!(d, '.' | ':' | '-' | '/' | ',') && !last_sep {
                    // Require a digit after the separator to avoid trailing `-`
                    // or decimal points produced next to punctuation.
                    if j + 1 < n && chars[j + 1].is_ascii_digit() {
                        last_sep = true;
                        j += 1;
                    } else {
                        break;
                    }
                } else {
                    break;
                }
            }
            let token: String = chars[i..j].iter().collect();
            let trimmed: String = token
                .trim_matches(|ch| matches!(ch, '.' | ':' | '-' | '/' | ','))
                .chars()
                .collect();
            if !trimmed.is_empty() {
                facts.push(trimmed);
            }
            i = j;
        } else if c.is_ascii_uppercase() || c == '0' && i + 1 < n && chars[i + 1] == 'x' {
            // error code / id tokens such as ERR_1001, 404, 0x1f4a
            let mut j = i + 1;
            while j < n && (chars[j].is_ascii_alphanumeric() || matches!(chars[j], '_' | '-' | '#'))
            {
                // Stop a long uppercase run that is really a normal English word.
                if chars[j].is_ascii_lowercase() && j - i > 0 {
                    // allow prefixes like `WARN` then remainders; keep going
                }
                j += 1;
            }
            let token: String = chars[i..j].iter().collect();
            let has_digit = token.chars().any(|d| d.is_ascii_digit());
            let looks_random = looks_like_code(&token);
            if (has_digit || looks_random) && token.chars().count() >= 3 {
                // error codes usually mix letters and digits
                facts.push(token);
            }
            i = j;
        } else {
            i += 1;
        }
    }
    facts
}

fn looks_like_code(token: &str) -> bool {
    let count_upper = token.chars().filter(|c| c.is_ascii_uppercase()).count();
    let count_digit = token.chars().filter(|c| c.is_ascii_digit()).count();
    // `HTTP`, `GPT` etc. from llm text are not error codes, but short all-upper
    // words followed by digits or a hex prefix are.
    token.starts_with("0x") || (count_upper >= 2 && count_digit >= 1)
}

/// Distinct set of key facts for fast comparison.
pub fn key_fact_set(text: &str) -> HashSet<String> {
    key_facts(text).into_iter().collect()
}

/// Low-information results (no visible change / unclear image) should update
/// the bubble far less aggressively, otherwise the model repeatedly disturbs
/// the user with identical "no change" messages.
pub fn is_low_information(text: &str) -> bool {
    let t = normalize(text);
    if t.is_empty() {
        return true;
    }
    [
        "无明显变化",
        "没有明显变化",
        "没什么变化",
        "未检测到变化",
        "画面不清晰",
        "图片不清晰",
        "看不太清",
        "看不清",
        "不清晰",
        "无法确认",
        "无法辨认",
        "无法确定",
        "模糊",
        "不清楚",
        "很难判断",
        "无法看清",
        "难以分辨",
        "不是很清晰",
    ]
    .iter()
    .any(|keyword| t.contains(keyword))
}

/// Extracts the subset of known status words present in a text.
pub fn status_words(text: &str) -> Vec<String> {
    STATUS_WORDS
        .iter()
        .filter(|word| text.contains(**word))
        .map(|word| word.to_string())
        .collect()
}

/// Result readability / confidence classification used to decide how firmly
/// the UI may present a fact.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum Readability {
    /// The model read the scene confidently.
    Clear,
    /// The model hedged (可能 / 大概 / 似乎 …).
    Partial,
    /// The model said the image is unclear / cannot confirm.
    Unclear,
}

pub fn classify_readability(text: &str) -> Readability {
    if [
        "看不清",
        "不清晰",
        "图片不清晰",
        "无法确认",
        "无法辨认",
        "无法确定",
        "模糊",
        "不清楚",
        "难以分辨",
        "不是很清晰",
    ]
    .iter()
    .any(|keyword| text.contains(keyword))
    {
        Readability::Unclear
    } else if [
        "可能",
        "大概",
        "疑似",
        "似乎",
        "估计",
        "好像",
        "应该",
        "大约是",
    ]
    .iter()
    .any(|keyword| text.contains(keyword))
    {
        Readability::Partial
    } else {
        Readability::Clear
    }
}

/// True when two consecutive results carry mutually exclusive status words.
/// In that case the app should fall back to a conservative message instead of
/// printing an apparently-firm but unstable fact.
pub fn contradicts(previous: &str, current: &str) -> bool {
    let previous_words = status_words(previous);
    let current_words = status_words(current);
    if previous_words.is_empty() || current_words.is_empty() {
        return false;
    }
    CONTRADICTION_PAIRS.iter().any(|(a, b)| {
        (previous_words.contains(&a.to_string()) && current_words.contains(&b.to_string()))
            || (previous_words.contains(&b.to_string()) && current_words.contains(&a.to_string()))
    })
}

/// Decides whether a new result should refresh the floating bubble.
///
/// - identical / semantically-equivalent results do not refresh;
/// - low-information results use a stricter similarity threshold;
/// - any new key fact (number / time / error code / status change) refreshes
///   even when the surrounding text is otherwise similar.
pub fn should_refresh(previous: &str, current: &str) -> bool {
    if previous == current {
        return false;
    }
    let normalized_previous = normalize(previous);
    let normalized_current = normalize(current);
    if normalized_previous.is_empty() || normalized_current.is_empty() {
        return true;
    }
    if normalized_previous == normalized_current {
        return false;
    }

    let similarity = unigram_containment(&normalized_current, &normalized_previous);
    let strict = is_low_information(current) || is_low_information(previous);
    let threshold = if strict { 0.45 } else { 0.80 };
    if similarity < threshold {
        return true;
    }

    let previous_facts = key_fact_set(previous);
    let current_facts = key_fact_set(current);
    let facts_changed = current_facts
        .iter()
        .any(|fact| !previous_facts.contains(fact));

    let previous_status = status_words(previous);
    let current_status = status_words(current);
    let status_changed = !previous_status.is_empty()
        && current_status
            .iter()
            .any(|word| !previous_status.contains(word));

    facts_changed || status_changed
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_drops_punctuation_but_keeps_facts() {
        assert_eq!(normalize("错误码 404，无法连接。"), "错误码404无法连接");
        assert_eq!(normalize("  余额： 1,234.50 元 "), "余额123450元");
    }

    #[test]
    fn identical_results_do_not_refresh() {
        assert!(!should_refresh(
            "屏幕上显示一个浏览器窗口。",
            "屏幕上显示一个浏览器窗口。"
        ));
    }

    #[test]
    fn high_similarity_without_new_facts_does_not_refresh() {
        assert!(!should_refresh(
            "屏幕上是一个浏览器窗口。",
            "屏幕上显示一个浏览器窗口。"
        ));
    }

    #[test]
    fn new_number_refreshes_even_when_text_is_similar() {
        assert!(should_refresh("下载进度为 42%。", "下载进度为 74%。"));
    }

    #[test]
    fn new_error_code_refreshes() {
        assert!(should_refresh(
            "程序遇到错误，请重试。",
            "程序遇到错误 0x1f4a，请重试。"
        ));
    }

    #[test]
    fn status_change_refreshes() {
        assert!(should_refresh("任务正在运行。", "任务已停止。"));
    }

    #[test]
    fn low_information_needs_real_diff() {
        assert!(!should_refresh("画面没有明显变化。", "画面无明显变化"));
        assert!(!should_refresh("图片不清晰，无法确认。", "画面不清晰。"));
    }

    #[test]
    fn extracts_key_facts() {
        let facts = key_facts("错误码 0x1f4a，时间 12:30，版本 2.4.1");
        for expected in ["0x1f4a", "12:30", "2.4.1"] {
            assert!(
                facts.iter().any(|f| f.contains(expected)),
                "missing {expected}"
            );
        }
    }

    #[test]
    fn classifies_readability() {
        assert_eq!(
            classify_readability("屏幕上有一个浏览器窗口。"),
            Readability::Clear
        );
        assert_eq!(
            classify_readability("可能是登录页面。"),
            Readability::Partial
        );
        assert_eq!(
            classify_readability("图片不清晰，无法确认内容。"),
            Readability::Unclear
        );
    }

    #[test]
    fn detects_contradicting_status_words() {
        assert!(contradicts("任务运行成功。", "任务运行失败。"));
        assert!(contradicts("窗口已打开。", "窗口已关闭。"));
        assert!(!contradicts("正在下载文件。", "文件下载完成。"));
    }
}
