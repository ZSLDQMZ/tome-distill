"""Budgeted sparse KB retriever: compound masking, 1-char deferral, hybrid diff, and multi-pass slot allocation."""

from __future__ import annotations

import difflib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from .profile import DomainProfile, RetrieverConfig, default_generic_profile
from .text_utils import TextNormalizer

_CLAUSE_SPLIT_RE = re.compile(r"([，。！？；：、\n]+)")


def _compute_diff_spans(draft_text: str, query_text: str) -> list[tuple[int, int]]:
    """Return character ranges in query_text modified/inserted relative to draft_text."""
    if not draft_text or not query_text:
        return []
    sm = difflib.SequenceMatcher(None, draft_text, query_text, autojunk=False)
    spans: list[tuple[int, int]] = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in {"replace", "insert"} and j2 > j1:
            spans.append((j1, j2))
    return spans


def _overlaps_any(span: tuple[int, int], windows: Sequence[tuple[int, int]]) -> bool:
    s0, s1 = span
    return any(s0 < w1 and s1 > w0 for w0, w1 in windows)


class SparseKBRetriever:
    """Zero-ML deterministic sparse retriever with compound masking and budgeted slot control."""

    def __init__(
        self,
        compiled_source: dict[str, Any] | str | Path,
        *,
        profile: DomainProfile | None = None,
        external_compounds: Iterable[str] | str | Path | None = None,
    ) -> None:
        self.profile = profile or default_generic_profile()
        self.normalizer = TextNormalizer(self.profile.text_normalization)

        if isinstance(compiled_source, (str, Path)):
            root = Path(compiled_source)
            self.index = json.loads((root / "compiled_lookup_index.json").read_text(encoding="utf-8"))
        elif "compiled_lookup_index" in compiled_source:
            self.index = dict(compiled_source["compiled_lookup_index"])
        else:
            self.index = dict(compiled_source)

        self.entry_meta: dict[str, dict[str, Any]] = dict(self.index.get("entry_meta") or {})
        self.form_map: dict[str, list[str]] = dict(self.index.get("dialect_form_to_entries") or {})
        self.t2s_map: dict[str, list[str]] = dict(self.index.get("dialect_t2s_to_entries") or {})
        self.anchor_map: dict[str, list[str]] = dict(self.index.get("standard_anchor_to_entries") or {})
        self.regex_rules: list[dict[str, Any]] = list(self.index.get("dialect_regex_rules") or [])

        exempt = set(self.index.get("grammar_multi_char_exemptions") or [])
        lex_compounds = set(self.index.get("non_grammar_lexical_compounds") or [])
        ext_set = self._load_external_compounds(external_compounds)
        # Keep >=2-char lexical/common compounds for masking, minus grammar multi-char exemptions.
        mask_pool = (ext_set | lex_compounds) - exempt
        self.mask_compounds: list[str] = sorted(
            (w for w in mask_pool if len(w) >= 2),
            key=lambda w: (-len(w), w),
        )

        # Longest-first surface tokens for deterministic span scanning.
        self._surface_keys: list[str] = sorted(
            set(self.form_map.keys()) | set(self.t2s_map.keys()),
            key=lambda k: (-len(k), k),
        )
        self._anchor_keys: list[str] = sorted(
            (k for k in self.anchor_map.keys() if len(k) >= 2),
            key=lambda k: (-len(k), k),
        )

    @staticmethod
    def _load_external_compounds(src: Iterable[str] | str | Path | None) -> set[str]:
        if src is None:
            return set()
        if isinstance(src, (str, Path)):
            p = Path(src)
            if not p.is_file():
                return set()
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "compounds" in data:
                return set(data["compounds"].keys())
            if isinstance(data, dict):
                return set(data.keys())
            if isinstance(data, list):
                return {str(x) for x in data}
            return set()
        return {str(x) for x in src if str(x)}

    def _build_mask(self, norm_text: str) -> list[bool]:
        n = len(norm_text)
        masked = [False] * n
        for comp in self.mask_compounds:
            start = 0
            clen = len(comp)
            while True:
                pos = norm_text.find(comp, start)
                if pos == -1:
                    break
                for i in range(pos, pos + clen):
                    masked[i] = True
                start = pos + 1
        return masked

    def retrieve(
        self,
        query_text: str,
        *,
        mode: str = "dialect_only",
        draft_standard: str = "",
        cfg: RetrieverConfig | None = None,
        strategy: str | None = None,
        max_cards: int | None = None,
        max_chars: int | None = None,
        format_style: str = "compact",
    ) -> dict[str, Any]:
        """Retrieve relevant KB entries under compound masking, 1-char deferral, and budget caps."""
        if mode not in {"dialect_only", "standard_anchor", "hybrid"}:
            raise ValueError(f"Unsupported retrieval mode: {mode!r}")

        rc = cfg or self.profile.retriever
        strat = strategy or rc.strategy
        if strat not in {"full", "longest", "budgeted", "R1_full", "R2_longest", "R3_budgeted"}:
            raise ValueError(f"Unsupported retrieval strategy: {strat!r}")

        card_cap = int(max_cards if max_cards is not None else rc.max_cards)
        default_chars = rc.max_chars_json if format_style == "json" else rc.max_chars_compact
        char_cap = int(max_chars if max_chars is not None else default_chars)

        raw_q = str(query_text or "")
        q_norm = self.normalizer.to_simp_keep_protected(raw_q, preserve_length=True)
        q_alias = self.normalizer.apply_compat_aliases(q_norm)
        # Preserve exact length for span mapping; fallback to q_norm if alias changed length.
        scan_text = q_alias if len(q_alias) == len(raw_q) else q_norm

        mask = self._build_mask(scan_text)
        diff_spans = _compute_diff_spans(draft_standard, raw_q) if (mode == "hybrid" and draft_standard) else []
        stopwords_1 = set(rc.open_class_1char_stop)
        stopwords_2 = set(rc.target_2char_stopwords)
        defer_1 = set(rc.defer_1char_gram)
        enclitic_exempt = set(rc.enclitic_tail_exempt)

        # candidate_eid -> best score dict
        hits: dict[str, dict[str, Any]] = {}
        suppressed_hits: list[dict[str, Any]] = []

        def _record_hit(eid: str, matched: str, source: str, span: tuple[int, int], bonus: float = 0.0) -> None:
            meta = self.entry_meta.get(eid)
            if not meta:
                return
            in_diff = 1.0 if (diff_spans and _overlaps_any(span, diff_spans)) else 0.0
            score = (len(matched) * 10.0) + (50.0 if meta.get("has_redline") else 0.0) + (15.0 * in_diff) + bonus
            cur = hits.get(eid)
            if cur is None or score > cur["score"]:
                hits[eid] = {
                    "entry_id": eid,
                    "matched_term": matched,
                    "match_source": source,
                    "span": list(span),
                    "score": score,
                    "in_diff": bool(in_diff),
                    "meta": meta,
                }

        # 1. Surface / t2s scanning (unless strict standard_anchor-only mode).
        if mode in {"dialect_only", "hybrid"}:
            for term in self._surface_keys:
                tlen = len(term)
                for txt_source in (raw_q, scan_text):
                    start = 0
                    while True:
                        pos = txt_source.find(term, start)
                        if pos == -1:
                            break
                        end = pos + tlen
                        if tlen == 1:
                            if term in stopwords_1:
                                suppressed_hits.append({"term": term, "reason": "open_class_1char_stop", "span": [pos, end]})
                                start = pos + 1
                                continue
                            if mask[pos]:
                                is_tail = (end == len(scan_text) or not mask[end]) and term in enclitic_exempt
                                if not is_tail:
                                    suppressed_hits.append({"term": term, "reason": "masked_by_compound", "span": [pos, end]})
                                    start = pos + 1
                                    continue
                        eids = self.form_map.get(term, []) or self.t2s_map.get(term, [])
                        defer_penalty = -5.0 if (tlen == 1 and term in defer_1) else 0.0
                        for eid in eids:
                            _record_hit(eid, term, "dialect_surface", (pos, end), bonus=defer_penalty)
                        start = pos + 1

            # 2. Clause-bounded regex rules.
            offset = 0
            for seg in _CLAUSE_SPLIT_RE.split(scan_text):
                if seg and not _CLAUSE_SPLIT_RE.fullmatch(seg):
                    for rule in self.regex_rules:
                        eid = str(rule.get("entry_id") or "")
                        pat = str(rule.get("pattern") or "")
                        if not eid or not pat:
                            continue
                        for m in re.finditer(pat, seg):
                            g0, g1 = offset + m.start(), offset + m.end()
                            if g1 > g0 and all(mask[i] for i in range(g0, min(g1, len(mask)))):
                                suppressed_hits.append({"term": m.group(0), "reason": "regex_inside_compound_mask", "span": [g0, g1]})
                                continue
                            _record_hit(eid, m.group(0), "regex_rule", (g0, g1), bonus=8.0)
                offset += len(seg)

        # 3. >=2-char standard-gloss anchor scan.
        if mode in {"standard_anchor", "hybrid"}:
            target_text = draft_standard if (mode == "hybrid" and draft_standard) else scan_text
            for anchor in self._anchor_keys:
                if len(anchor) == 2 and anchor in stopwords_2:
                    continue
                start = 0
                while True:
                    pos = target_text.find(anchor, start)
                    if pos == -1:
                        break
                    end = pos + len(anchor)
                    for eid in self.anchor_map.get(anchor, []):
                        _record_hit(eid, anchor, "standard_anchor", (pos, end), bonus=4.0)
                    start = pos + 1

        ranked = sorted(
            hits.values(),
            key=lambda h: (-h["score"], -len(h["matched_term"]), h["entry_id"]),
        )
        selected, stats = self._allocate_budget(ranked, strat, card_cap, char_cap, rc, format_style)
        return {
            "query": raw_q,
            "mode": mode,
            "strategy": strat,
            "selected_entries": selected,
            "formatted_context": self._format_entries(selected, format_style),
            "suppressed_hits": suppressed_hits,
            "stats": stats,
        }

    def _entry_text_len(self, meta: dict[str, Any], format_style: str) -> int:
        if format_style == "json":
            return len(json.dumps(meta, ensure_ascii=False))
        return len(str(meta.get("compact_text") or "")) + 2

    def _allocate_budget(
        self,
        ranked: list[dict[str, Any]],
        strategy: str,
        card_cap: int,
        char_cap: int,
        rc: RetrieverConfig,
        format_style: str,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        if strategy in {"full", "R1_full"}:
            return ranked, {"total_candidates": len(ranked), "selected_count": len(ranked), "used_chars": sum(self._entry_text_len(h["meta"], format_style) for h in ranked)}

        seen_clusters: set[str] = set()
        seen_eids: set[str] = set()
        fw_slots: dict[str, int] = {}
        leaf_slots: dict[str, int] = {}
        selected: list[dict[str, Any]] = []
        used_chars = 0

        def _can_pick(hit: dict[str, Any], *, max_c: int, max_ch: int, check_slots: bool) -> bool:
            nonlocal used_chars
            eid = hit["entry_id"]
            if eid in seen_eids or len(selected) >= max_c:
                return False
            meta = hit["meta"]
            for cid in meta.get("conflict_ids") or []:
                if cid in seen_clusters:
                    return False
            if check_slots and meta.get("table") == "grammar_and_function_words":
                skey = str(meta.get("slot_key") or eid)
                leaf = str(meta.get("taxonomy_leaf") or "")
                if fw_slots.get(skey, 0) >= rc.max_slots_per_fw:
                    return False
                if leaf and leaf_slots.get(leaf, 0) >= rc.max_slots_per_leaf:
                    return False
            elen = self._entry_text_len(meta, format_style)
            if selected and (used_chars + elen > max_ch):
                return False
            return True

        def _take(hit: dict[str, Any], pass_name: str) -> None:
            nonlocal used_chars
            meta = hit["meta"]
            eid = hit["entry_id"]
            seen_eids.add(eid)
            for cid in meta.get("conflict_ids") or []:
                seen_clusters.add(cid)
            if meta.get("table") == "grammar_and_function_words":
                skey = str(meta.get("slot_key") or eid)
                leaf = str(meta.get("taxonomy_leaf") or "")
                fw_slots[skey] = fw_slots.get(skey, 0) + 1
                if leaf:
                    leaf_slots[leaf] = leaf_slots.get(leaf, 0) + 1
            used_chars += self._entry_text_len(meta, format_style)
            rec = dict(hit)
            rec["allocated_pass"] = pass_name
            selected.append(rec)

        if strategy in {"longest", "R2_longest"}:
            by_len = sorted(ranked, key=lambda h: (-len(h["matched_term"]), -h["score"], h["entry_id"]))
            for h in by_len:
                if _can_pick(h, max_c=card_cap, max_ch=char_cap, check_slots=False):
                    _take(h, "longest")
            return selected, {"total_candidates": len(ranked), "selected_count": len(selected), "used_chars": used_chars}

        # Budgeted 4-pass priority allocation.
        per_lex = rc.per_lex_est_json if format_style == "json" else rc.per_lex_est_compact
        lex_res_cards = min(rc.lex_reserve, max(0, card_cap - 1))
        lex_res_chars = min(lex_res_cards * per_lex, max(0, char_cap // 2))

        # Pass 1: Hard redlines / conflict records first.
        for h in ranked:
            m = h["meta"]
            if (m.get("has_redline") or m.get("conflict_ids")) and _can_pick(
                h, max_c=card_cap, max_ch=char_cap, check_slots=False
            ):
                _take(h, "P1_redline")

        # Pass 2: Grammar & function words up to lexical reserve caps.
        gram_card_cap = max(len(selected), card_cap - lex_res_cards)
        gram_char_cap = max(used_chars, char_cap - lex_res_chars)
        for h in ranked:
            m = h["meta"]
            if m.get("table") == "grammar_and_function_words" and _can_pick(
                h, max_c=gram_card_cap, max_ch=gram_char_cap, check_slots=True
            ):
                _take(h, "P2_grammar")

        # Pass 3: Lexical / idiom hits into reserved budget.
        for h in ranked:
            m = h["meta"]
            if m.get("table") != "grammar_and_function_words" and _can_pick(
                h, max_c=card_cap, max_ch=char_cap, check_slots=False
            ):
                _take(h, "P3_lexical")

        # Pass 4: Backfill remaining budget regardless of table.
        for h in ranked:
            if _can_pick(h, max_c=card_cap, max_ch=char_cap, check_slots=False):
                _take(h, "P4_backfill")

        return selected, {
            "total_candidates": len(ranked),
            "selected_count": len(selected),
            "used_chars": used_chars,
            "lex_reserve_cards": lex_res_cards,
        }

    @staticmethod
    def _format_entries(selected: list[dict[str, Any]], format_style: str) -> str:
        if format_style == "json":
            payload = [
                {
                    "entry_id": h["entry_id"],
                    "forms": h["meta"].get("forms"),
                    "compact_text": h["meta"].get("compact_text"),
                    "redlines": h["meta"].get("redlines"),
                }
                for h in selected
            ]
            return json.dumps(payload, ensure_ascii=False, indent=2)
        return "\n".join(f"[{h['entry_id']}] {h['meta'].get('compact_text', '')}" for h in selected)
