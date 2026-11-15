"""Unified 1:1 protected OpenCC normalization, encoding hygiene, and provenance checks."""

from __future__ import annotations

import re
from typing import Any, Iterable

import opencc

from tome_distill.profile import TextNormalizationConfig

RE_CJK = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002ebef]")
RE_CJK_EXT = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff\U00020000-\U0003134f]")
RE_CJK_SEQ = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\U00020000-\U0002ebef]{1,16}")
RE_TEMPLATE_OR_BAD = re.compile(r"[A-Za-z0-9…□■⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻＊*]")
RE_IDS = re.compile(r"[⿰⿱⿲⿳⿴⿵⿶⿷⿸⿹⿺⿻]")
RE_STRIP_FOR_LEAKAGE = re.compile(r"[\s\u200b\ufeff\u3000，。！？、：；“”\"'‘’（）()\[\]【】《》—…·~～|]+")


def is_cjk_char(ch: str) -> bool:
    return bool(ch) and bool(RE_CJK.match(ch))


def has_bad_chars(text: str) -> list[str]:
    """Check raw text for replacement chars, PUA glyphs, black squares, and C0/C1 controls."""
    issues: list[str] = []
    if "■" in text:
        issues.append("contains_black_square")
    if "\ufffd" in text:
        issues.append("contains_replacement_char_U+FFFD")
    for ch in text:
        cp = ord(ch)
        if 0xE000 <= cp <= 0xF8FF:
            issues.append(f"contains_PUA_U+{cp:04X}")
            break
    for ch in text:
        cp = ord(ch)
        if (cp < 0x20 and ch not in ("\n", "\r", "\t")) or (0x7F <= cp <= 0x9F):
            issues.append(f"contains_ctrl_0x{cp:02X}")
            break
    return issues


def find_bad_chars_in_obj(obj: Any, prefix: str = "") -> list[str]:
    """Recursively inspect raw strings BEFORE json.dumps so C0 controls cannot hide as \\u00XX."""
    findings: list[str] = []
    if isinstance(obj, str):
        for issue in has_bad_chars(obj):
            findings.append(f"{prefix or '<root>'}:{issue}")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            sub_prefix = f"{prefix}.{k}" if prefix else str(k)
            if isinstance(k, str):
                for issue in has_bad_chars(k):
                    findings.append(f"{sub_prefix}<key>:{issue}")
            findings.extend(find_bad_chars_in_obj(v, sub_prefix))
    elif isinstance(obj, (list, tuple)):
        for idx, item in enumerate(obj):
            findings.extend(find_bad_chars_in_obj(item, f"{prefix}[{idx}]"))
    return findings


def scrub_bad_chars(text: str) -> str:
    """Remove ■, U+FFFD, PUA glyphs, and illegal C0/C1 control characters."""
    if not text:
        return ""
    out: list[str] = []
    for ch in text:
        cp = ord(ch)
        if ch in ("■", "\ufffd") or (0xE000 <= cp <= 0xF8FF):
            continue
        if (cp < 0x20 and ch not in ("\n", "\r", "\t")) or (0x7F <= cp <= 0x9F):
            continue
        out.append(ch)
    return re.sub(r"\s{2,}", " ", "".join(out)).strip()


def clean_latex_fragments(text: str) -> str:
    """Strip OCR-emitted $...$ LaTeX wrappers while retaining actual CJK/text content."""
    if not text or "$" not in text:
        return text

    def _repl(m: re.Match[str]) -> str:
        inner = m.group(1)
        inner = re.sub(r"\\underset\{\\cdot\}\{([^}]*)\}", r"\1", inner)
        inner = re.sub(r"\\uwave\{\\text\{([^}]*)\}\}", r"\1", inner)
        inner = re.sub(r"\\underline\{\\text\{([^}]*)\}\}", r"\1", inner)
        inner = re.sub(r"\\text\{([^}]*)\}", r"\1", inner)
        inner = re.sub(r"\\[a-zA-Z]+", "", inner)
        return inner.replace("{", "").replace("}", "").strip()

    cleaned = re.sub(r"\$([^$]*)\$", _repl, text)
    return re.sub(r"\s{2,}", " ", cleaned).strip()


class TextNormalizer:
    """Single source of truth for protected-character OpenCC conversion and form cleanup."""

    def __init__(self, cfg: TextNormalizationConfig | None = None) -> None:
        self.cfg = cfg or TextNormalizationConfig()
        self.cc = opencc.OpenCC(self.cfg.opencc_config or "t2s")
        self.protected_chars: frozenset[str] = frozenset(self.cfg.protected_chars)
        self.compat_aliases: dict[str, str] = dict(self.cfg.compat_char_aliases)
        self._split_re: re.Pattern[str] | None = None
        if self.protected_chars:
            escaped = "".join(re.escape(ch) for ch in sorted(self.protected_chars))
            self._split_re = re.compile(f"([{escaped}])")
        self._circled_re = (
            re.compile(f"[{re.escape(self.cfg.circled_chars)}\\s]+")
            if self.cfg.circled_chars
            else re.compile(r"\s+")
        )

    def apply_compat_aliases(self, text: str) -> str:
        out = text or ""
        for src, dst in self.compat_aliases.items():
            if src in out:
                out = out.replace(src, dst)
        return out

    def _convert_chunk(self, chunk: str, preserve_length: bool) -> str:
        if not chunk:
            return ""
        conv = self.cc.convert(chunk)
        if not preserve_length or len(conv) == len(chunk):
            return conv
        # Safe per-char 1:1 fallback ONLY on chunks whose OpenCC output length drifted.
        out: list[str] = []
        for ch in chunk:
            c = self.cc.convert(ch)
            out.append(c if len(c) == 1 else ch)
        return "".join(out)

    def to_simp_keep_protected(self, text: str, preserve_length: bool = True) -> str:
        """Convert to simplified form while protecting domain characters and keeping span length 1:1."""
        if not text:
            return ""
        if self._split_re is None:
            return self._convert_chunk(text, preserve_length=preserve_length)

        parts = self._split_re.split(text)
        out: list[str] = []
        for idx, part in enumerate(parts):
            if not part:
                continue
            if idx % 2 == 1 and part in self.protected_chars:
                out.append(part)
            else:
                out.append(self._convert_chunk(part, preserve_length=preserve_length))
        res = "".join(out)
        if preserve_length and len(res) != len(text):
            raise AssertionError("Span length invariant broken in to_simp_keep_protected")
        return res

    def strip_circled_and_space(self, text: str) -> str:
        return self._circled_re.sub("", (text or "").strip())

    def clean_form_token(self, raw: str) -> str:
        """Clean a surface headword token while quarantining missing-glyph/IDS placeholders."""
        s = self.apply_compat_aliases(str(raw or "").strip())
        s = re.sub(r"\[[^\]]*\]", "", s)
        s = re.sub(r"〖[^〗]*〗", "", s)
        s = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼]+", "", s)
        s = re.sub(r"[（(][^）)]*[）)]", "", s)
        if any(ch in s for ch in ("□", "■", "…")) or RE_IDS.search(s):
            return ""
        s = re.sub(r"[＊*]+", "", s)
        s = s.replace("·", "").replace("•", "")
        return s.strip(" -—~～，、/？?！!。；;：:“”\"'‘’《》〈〉")

    def is_valid_pure_form(self, s: str) -> bool:
        if not s or any(alias in s for alias in self.compat_aliases):
            return False
        if RE_TEMPLATE_OR_BAD.search(s):
            return False
        if any(tok in s for tok in ("…", "……", "□", "■", "(", ")", "（", "）", "[", "]", "/")):
            return False
        cjk_cnt = sum(1 for ch in s if is_cjk_char(ch))
        return cjk_cnt >= 1 and all(is_cjk_char(ch) or ch in "，—" for ch in s)

    def is_valid_zh_anchor(self, s: str) -> bool:
        if not s:
            return False
        if RE_TEMPLATE_OR_BAD.search(s):
            return False
        if any(tok in s for tok in ("…", "……", "□", "■", "(", ")", "（", "）", "[", "]", "/", "；", ";", "、")):
            return False
        cjk_cnt = sum(1 for ch in s if is_cjk_char(ch))
        return cjk_cnt >= 1 and all(is_cjk_char(ch) for ch in s)

    def norm_text_for_leakage(self, s: str) -> str:
        conv = self.to_simp_keep_protected(str(s or ""), preserve_length=False).strip()
        return RE_STRIP_FOR_LEAKAGE.sub("", conv)


def get_attested_whole_forms(
    card_ids: Iterable[str],
    cards_by_id: dict[str, dict[str, Any]],
    normalizer: TextNormalizer,
    card_to_confs: dict[str, list[str]] | None = None,
    confs_by_id: dict[str, dict[str, Any]] | None = None,
) -> tuple[set[str], str]:
    """Build exact attested surface forms and normalized provenance corpus across source cards."""
    exact_forms: set[str] = set()
    corpus_chunks: list[str] = []

    for cid in card_ids:
        c = cards_by_id[cid]
        jv = c.get("judge_view") or {}
        pv = c.get("provenance") or {}
        for raw_f in jv.get("forms") or []:
            cf = normalizer.clean_form_token(raw_f)
            if cf:
                exact_forms.add(cf)
                corpus_chunks.append(cf)
            if not any(b in raw_f for b in ("□", "■")) and not RE_IDS.search(raw_f):
                for sub in RE_CJK_SEQ.findall(normalizer.apply_compat_aliases(raw_f)):
                    exact_forms.add(sub)
        for fld in ("meaning", "usage", "standard_counterpart", "mandarin_counterpart", "misuse_or_confusable"):
            val = normalizer.apply_compat_aliases(str(jv.get(fld) or ""))
            if val:
                val_clean = re.sub(r"\[[^\]]*\]", "", val)
                val_clean = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼*＊·•]+", "", val_clean)
                corpus_chunks.append(val_clean)
        for ex in jv.get("examples") or []:
            if isinstance(ex, dict):
                raw_ex = ex.get("dialect") or ex.get("surface") or ex.get("form") or ""
                d_ex = normalizer.apply_compat_aliases(str(raw_ex))
                d_clean = re.sub(r"\[[^\]]*\]", "", d_ex)
                d_clean = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼*＊·•]+", "", d_clean)
                if d_clean:
                    corpus_chunks.append(d_clean)
        for pvf in ("quote", "notes"):
            val = normalizer.apply_compat_aliases(str(pv.get(pvf) or ""))
            if val:
                val_clean = re.sub(r"\[[^\]]*\]", "", val)
                val_clean = re.sub(r"[⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻⁼*＊·•]+", "", val_clean)
                corpus_chunks.append(val_clean)

        if card_to_confs and confs_by_id:
            for gid in card_to_confs.get(cid, []):
                for uf in confs_by_id[gid].get("_clean_forms_union", []):
                    exact_forms.add(uf)
                    corpus_chunks.append(uf)

    return exact_forms, " | ".join(corpus_chunks)


def is_whole_form_attested(
    form: str,
    exact_forms: set[str],
    joined_corpus: str,
    normalizer: TextNormalizer,
) -> bool:
    if not normalizer.is_valid_pure_form(form):
        return False
    return form in exact_forms or form in joined_corpus
