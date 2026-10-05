//! Text post-processing for Confucius4-R2T2 output.
//!
//! Faithful ports of the helpers the reference Python implementation
//! (`r2t2/r2t2_asr.py` in the Confucius4-R2T2 repository, plus
//! `qwen_asr/inference/utils.py` from Qwen3-ASR) runs on every decode
//! step. The streaming loop's stability guarantees depend on these
//! behaving exactly like the reference, so they're kept separate and
//! unit-tested.

/// Separator between the metadata header (`language Chinese`) and the
/// transcript in Qwen3-ASR output.
pub(crate) const ASR_TEXT_TAG: &str = "<asr_text>";
const LANG_PREFIX: &str = "language ";

/// Marker R2T2 emits where the stable prefix ends; everything after it
/// is speculative and is always discarded.
pub(crate) const STABLE_PREFIX_END: char = '|';

/// Everything before the first [`STABLE_PREFIX_END`] marker.
pub(crate) fn before_marker(s: &str) -> &str {
    s.split(STABLE_PREFIX_END).next().unwrap_or("")
}

pub(crate) fn is_cjk(c: char) -> bool {
    ('\u{4e00}'..='\u{9fff}').contains(&c)
}

/// Normalise punctuation to match the script of the preceding character:
/// Chinese punctuation after Hanzi, ASCII punctuation after ASCII
/// letters/digits/quotes. Mirrors `_normalize_punct_by_context`.
pub(crate) fn normalize_punct_by_context(text: &str) -> String {
    const PAIRS: [(char, char); 8] = [
        (',', '，'),
        ('.', '。'),
        ('!', '！'),
        ('?', '？'),
        (';', '；'),
        (':', '：'),
        ('(', '（'),
        (')', '）'),
    ];
    let to_zh = |c: char| PAIRS.iter().find(|(en, _)| *en == c).map(|(_, zh)| *zh);
    let to_en = |c: char| PAIRS.iter().find(|(_, zh)| *zh == c).map(|(en, _)| *en);
    let is_punct = |c: char| to_zh(c).is_some() || to_en(c).is_some();

    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    for (i, &c) in chars.iter().enumerate() {
        if !is_punct(c) {
            out.push(c);
            continue;
        }
        let prev = chars[..i].iter().rev().find(|ch| !ch.is_whitespace());
        let replaced = match prev {
            Some(&p) if is_cjk(p) => to_zh(c).unwrap_or(c),
            Some(&p) if p.is_ascii() && (p.is_ascii_alphanumeric() || p == '"' || p == '\'') => {
                to_en(c).unwrap_or(c)
            }
            _ => c,
        };
        out.push(replaced);
    }
    out
}

/// Remove whitespace between two adjacent Hanzi, keeping spaces inside
/// English runs of mixed-language text.
pub(crate) fn remove_spaces_between_cjk(text: &str) -> String {
    let chars: Vec<char> = text.chars().collect();
    let mut out = String::with_capacity(text.len());
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        if c.is_whitespace() {
            let start = i;
            while i < chars.len() && chars[i].is_whitespace() {
                i += 1;
            }
            let prev_cjk = start > 0 && is_cjk(chars[start - 1]);
            let next_cjk = i < chars.len() && is_cjk(chars[i]);
            if !(prev_cjk && next_cjk) {
                out.extend(&chars[start..i]);
            }
            continue;
        }
        out.push(c);
        i += 1;
    }
    out
}

/// Parse raw Qwen3-ASR output into `(language, text)`. Mirrors
/// `qwen_asr.parse_asr_output`, including its repetition clean-up.
///
/// - `language Chinese<asr_text>你好` → `("Chinese", "你好")`
/// - no tag → `("", whole string)`
/// - `language None<asr_text>` → `("", "")` (silence)
/// - with `forced_language` the whole string is the transcript.
pub(crate) fn parse_asr_output(raw: &str, forced_language: Option<&str>) -> (String, String) {
    let s = raw.trim();
    if s.is_empty() {
        return (String::new(), String::new());
    }
    let s = detect_and_fix_repetitions(s, 20);
    if let Some(lang) = forced_language {
        return (lang.to_string(), s);
    }
    let Some((meta, text)) = s.split_once(ASR_TEXT_TAG) else {
        return (String::new(), s.trim().to_string());
    };
    if meta.to_lowercase().contains("language none") {
        return (String::new(), text.trim().to_string());
    }
    let lang = meta
        .lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .and_then(|l| {
            l.to_lowercase()
                .starts_with(LANG_PREFIX)
                .then(|| l[LANG_PREFIX.len()..].trim())
        })
        .map(normalize_language_name)
        .unwrap_or_default();
    (lang, text.trim().to_string())
}

/// The language name from a `language X<asr_text>` header, without the
/// repetition clean-up (`parse_language_output` in the reference).
pub(crate) fn parse_language(raw: &str) -> String {
    let s = raw.trim();
    let Some((meta, _)) = s.split_once(ASR_TEXT_TAG) else {
        return String::new();
    };
    if meta.to_lowercase().contains("language none") {
        return String::new();
    }
    meta.lines()
        .map(str::trim)
        .find(|l| !l.is_empty())
        .and_then(|l| {
            l.to_lowercase()
                .starts_with(LANG_PREFIX)
                .then(|| l[LANG_PREFIX.len()..].trim())
        })
        .map(normalize_language_name)
        .unwrap_or_default()
}

/// `chinese` / `CHINESE` → `Chinese`, matching Qwen3-ASR's canonical names.
pub(crate) fn normalize_language_name(name: &str) -> String {
    let name = name.trim();
    let mut chars = name.chars();
    match chars.next() {
        Some(first) => first
            .to_uppercase()
            .chain(chars.flat_map(char::to_lowercase))
            .collect(),
        None => String::new(),
    }
}

/// Collapse runaway repetitions (a character repeated more than
/// `threshold` times, or a pattern of up to 20 characters repeated at
/// least `threshold` times). Port of `detect_and_fix_repetitions`.
pub(crate) fn detect_and_fix_repetitions(text: &str, threshold: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    let fixed = fix_char_repeats(&chars, threshold);
    fix_pattern_repeats(&fixed, threshold, 20)
        .into_iter()
        .collect()
}

fn fix_char_repeats(s: &[char], thresh: usize) -> Vec<char> {
    let mut res = Vec::with_capacity(s.len());
    let mut i = 0;
    while i < s.len() {
        let mut count = 1;
        while i + count < s.len() && s[i + count] == s[i] {
            count += 1;
        }
        if count > thresh {
            res.push(s[i]);
        } else {
            res.extend_from_slice(&s[i..i + count]);
        }
        i += count;
    }
    res
}

fn fix_pattern_repeats(s: &[char], thresh: usize, max_len: usize) -> Vec<char> {
    let n = s.len();
    let min_repeat_chars = thresh * 2;
    if n < min_repeat_chars {
        return s.to_vec();
    }
    let mut result = Vec::with_capacity(n);
    let mut i = 0;
    let mut found = false;
    while i + min_repeat_chars <= n {
        for k in 1..=max_len {
            if i + k * thresh > n {
                break;
            }
            let pattern = &s[i..i + k];
            let valid = (1..thresh).all(|rep| {
                let start = i + rep * k;
                &s[start..start + k] == pattern
            });
            if valid {
                let mut end = i + thresh * k;
                while end + k <= n && &s[end..end + k] == pattern {
                    end += k;
                }
                result.extend_from_slice(pattern);
                result.extend(fix_pattern_repeats(&s[end..], thresh, max_len));
                found = true;
                break;
            }
        }
        if found {
            break;
        }
        result.push(s[i]);
        i += 1;
    }
    if !found {
        result.extend_from_slice(&s[i..]);
    }
    result
}

/// Whether the last word of `text` (ignoring trailing punctuation and
/// whitespace) contains a Hanzi. Drives the token-budget heuristic: a
/// Chinese character is usually one token, an English word often two.
pub(crate) fn ends_with_cjk_word(text: &str) -> bool {
    text.chars()
        .rev()
        .find(|c| !c.is_whitespace() && !is_punctuation(*c))
        .is_some_and(is_cjk)
}

fn is_punctuation(c: char) -> bool {
    c.is_ascii_punctuation() || "，。！？、；：（）《》「」『』【】“”‘’…—～·".contains(c)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn punct_follows_previous_script() {
        assert_eq!(normalize_punct_by_context("你好,世界."), "你好，世界。");
        assert_eq!(normalize_punct_by_context("hello，world。"), "hello,world.");
        // Leading punctuation has no context and is kept as-is.
        assert_eq!(normalize_punct_by_context(",a"), ",a");
        // Whitespace before the mark is skipped when looking back.
        assert_eq!(normalize_punct_by_context("好 ?"), "好 ？");
    }

    #[test]
    fn cjk_spaces_removed_but_english_spaces_kept() {
        assert_eq!(
            remove_spaces_between_cjk("你 好 world wide 世 界"),
            "你好 world wide 世界"
        );
    }

    #[test]
    fn parses_tagged_output() {
        assert_eq!(
            parse_asr_output("language Chinese<asr_text>你好", None),
            ("Chinese".into(), "你好".into())
        );
        assert_eq!(
            parse_asr_output("language None<asr_text>", None),
            (String::new(), String::new())
        );
        assert_eq!(
            parse_asr_output("just text", None),
            (String::new(), "just text".into())
        );
        assert_eq!(
            parse_asr_output("hello", Some("English")),
            ("English".into(), "hello".into())
        );
        assert_eq!(parse_language("language english<asr_text>hi"), "English");
        assert_eq!(parse_language("no tag"), "");
    }

    #[test]
    fn collapses_runaway_repetitions() {
        let s = "a".repeat(30);
        assert_eq!(detect_and_fix_repetitions(&s, 20), "a");
        let s = format!("x{}y", "ab".repeat(25));
        assert_eq!(detect_and_fix_repetitions(&s, 20), "xaby");
        // Ordinary text is untouched.
        assert_eq!(detect_and_fix_repetitions("hello world", 20), "hello world");
    }

    #[test]
    fn marker_and_language_helpers() {
        assert_eq!(before_marker("abc|def"), "abc");
        assert_eq!(before_marker("abc"), "abc");
        assert!(ends_with_cjk_word("hello 你好。"));
        assert!(!ends_with_cjk_word("你好 hello."));
        assert_eq!(normalize_language_name("cHINESE"), "Chinese");
    }
}
