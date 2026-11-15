"""Incremental field_diffs patching, page/ID expansion, stratified blind sampling, and QA prechecks."""

from __future__ import annotations

import copy
import difflib
import random
import re
from typing import Any, Sequence

from .profile import DomainProfile, default_generic_profile
from .schema import validate_card

_RANGE_RE = re.compile(r"(\d+)\s*[-–—~]\s*(\d+)")
_INT_RE = re.compile(r"\d+")


def parse_page_nums(page_str: str | int, *, max_page_span: int = 50) -> list[int]:
    """Parse page numbers and expand bounded page ranges."""
    if isinstance(page_str, int):
        return [page_str] if page_str > 0 else []
    s = str(page_str or "").strip()
    if not s:
        return []

    pages: set[int] = set()
    consumed_spans: list[tuple[int, int]] = []
    for m in _RANGE_RE.finditer(s):
        start, end = int(m.group(1)), int(m.group(2))
        if start > end:
            raise ValueError(f"Inverted page range {start}-{end} in {page_str!r}")
        if end - start > max_page_span:
            raise ValueError(
                f"Page range span {end - start} exceeds max_page_span={max_page_span} in {page_str!r}"
            )
        pages.update(range(start, end + 1))
        consumed_spans.append(m.span())

    # Parse standalone numbers outside matched ranges.
    chars = list(s)
    for sp_start, sp_end in consumed_spans:
        for i in range(sp_start, sp_end):
            chars[i] = " "
    remainder = "".join(chars)
    for m in _INT_RE.finditer(remainder):
        val = int(m.group(0))
        if val > 0:
            pages.add(val)
    return sorted(pages)


def _insert_suffix_for_index(idx: int) -> str:
    """Return B..Z for idx=1..25, then multi-letter AA, AB, ... for idx>=26."""
    if idx < 1:
        raise ValueError("idx must be >= 1")
    if idx <= 25:
        return chr(ord("A") + idx)
    # idx >= 26 -> 0-based offset into AA, AB, ... AZ, BA, ...
    n = idx - 26
    width = 2
    cap = 26 ** width
    while n >= cap:
        n -= cap
        width += 1
        cap = 26 ** width
    letters: list[str] = []
    for _ in range(width):
        letters.append(chr(ord("A") + (n % 26)))
        n //= 26
    return "".join(reversed(letters))


def next_inserted_card_id(anchor_id: str, existing_ids: set[str]) -> str:
    """Allocate the next unused insertion ID after anchor_id (B..Z, then AA, AB, ...)."""
    idx = 1
    while True:
        cand = f"{anchor_id}{_insert_suffix_for_index(idx)}"
        if cand not in existing_ids:
            return cand
        idx += 1


def _deep_merge_card(target: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(target)
    for key, val in patch.items():
        if key == "card_id":
            continue
        if isinstance(val, dict) and isinstance(out.get(key), dict):
            merged_sub = dict(out[key])
            merged_sub.update(copy.deepcopy(val))
            out[key] = merged_sub
        else:
            out[key] = copy.deepcopy(val)
    return out


def apply_field_diffs(
    cards: Sequence[dict[str, Any]],
    diff_payload: dict[str, Any],
    *,
    reviewer_id: str = "",
    require_distinct_reviewer: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Apply incremental updates, insert_after, and delete_ids patches while enforcing provenance rules."""
    work: list[dict[str, Any]] = [copy.deepcopy(c) for c in cards]
    existing_ids: set[str] = {str(c.get("card_id") or "") for c in work}

    delete_ids = {str(cid) for cid in (diff_payload.get("delete_ids") or [])}
    updates = list(diff_payload.get("updates") or [])
    inserts = list(diff_payload.get("insert_after") or [])

    update_by_id: dict[str, dict[str, Any]] = {}
    for item in updates:
        cid = str(item.get("card_id") or "").strip()
        if not cid:
            raise ValueError("field_diffs update entry missing card_id")
        if cid not in existing_ids:
            raise KeyError(f"field_diffs update references unknown card_id={cid!r}")
        update_by_id[cid] = item

    # 1. Apply deletions first, tracking error if someone tries to both delete and update same ID.
    both = delete_ids & set(update_by_id.keys())
    if both:
        raise ValueError(f"card_id(s) present in both delete_ids and updates: {sorted(both)}")

    after_del: list[dict[str, Any]] = []
    deleted_count = 0
    for c in work:
        cid = str(c.get("card_id") or "")
        if cid in delete_ids:
            deleted_count += 1
            existing_ids.discard(cid)
            continue
        if cid in update_by_id:
            patch = {k: v for k, v in update_by_id[cid].items() if k != "card_id"}
            if "fields" in patch and isinstance(patch["fields"], dict):
                patch = patch["fields"]
            c = _deep_merge_card(c, patch)
            if reviewer_id:
                pv = dict(c.get("provenance") or {})
                if require_distinct_reviewer and pv.get("extracted_by") == reviewer_id:
                    raise ValueError(
                        f"Dual-blind violation applying patch to {cid!r}: reviewer {reviewer_id!r} == extracted_by"
                    )
                pv["verified_by"] = reviewer_id
                c["provenance"] = pv
        after_del.append(c)

    # 2. Group insert_after items by anchor_id, preserving insertion order.
    inserts_by_anchor: dict[str, list[dict[str, Any]]] = {}
    for item in inserts:
        anchor = str(item.get("after_card_id") or item.get("anchor_id") or "").strip()
        if not anchor:
            raise ValueError("field_diffs insert_after entry missing after_card_id/anchor_id")
        if anchor not in {str(c.get("card_id") or "") for c in after_del}:
            raise KeyError(f"field_diffs insert_after references unknown/deleted anchor {anchor!r}")
        payloads = item.get("cards") if isinstance(item.get("cards"), list) else [item.get("card")]
        for raw_new in payloads:
            if not isinstance(raw_new, dict):
                raise ValueError(f"insert_after entry for {anchor!r} missing card object")
            inserts_by_anchor.setdefault(anchor, []).append(copy.deepcopy(raw_new))

    final_cards: list[dict[str, Any]] = []
    inserted_ids: list[str] = []
    for c in after_del:
        final_cards.append(c)
        anchor = str(c.get("card_id") or "")
        for new_card in inserts_by_anchor.get(anchor, []):
            req_id = str(new_card.get("card_id") or "").strip()
            if not req_id or req_id == "AUTO":
                new_id = next_inserted_card_id(anchor, existing_ids)
                new_card["card_id"] = new_id
            elif req_id in existing_ids:
                raise ValueError(f"insert_after requested duplicate card_id {req_id!r}")
            else:
                new_id = req_id
            existing_ids.add(new_id)
            if reviewer_id:
                pv = dict(new_card.get("provenance") or {})
                if require_distinct_reviewer and pv.get("extracted_by") == reviewer_id:
                    raise ValueError(
                        f"Dual-blind violation on insert {new_id!r}: reviewer {reviewer_id!r} == extracted_by"
                    )
                pv["verified_by"] = reviewer_id
                new_card["provenance"] = pv
            inserted_ids.append(new_id)
            final_cards.append(new_card)

    summary = {
        "input_count": len(cards),
        "updated_count": len(update_by_id),
        "inserted_count": len(inserted_ids),
        "deleted_count": deleted_count,
        "output_count": len(final_cards),
        "inserted_ids": inserted_ids,
        "empty_nodes": list(diff_payload.get("empty_nodes") or []),
        "visual_audit_notes": list(diff_payload.get("visual_audit_notes") or []),
    }
    return final_cards, summary


def stratified_blind_sample(
    cards: Sequence[dict[str, Any]],
    *,
    ratio: float = 0.05,
    min_per_doc: int = 3,
    seed: int = 42,
    redact_provenance: bool = False,
) -> list[dict[str, Any]]:
    """Deterministically draw a stratified QA sample grouped by provenance.source_doc."""
    by_doc: dict[str, list[dict[str, Any]]] = {}
    for card in cards:
        doc = str((card.get("provenance") or {}).get("source_doc") or "__unknown__")
        by_doc.setdefault(doc, []).append(card)

    rng = random.Random(seed)
    sampled: list[dict[str, Any]] = []
    for doc in sorted(by_doc.keys()):
        pool = list(by_doc[doc])
        rng.shuffle(pool)
        k = min(len(pool), max(min_per_doc, round(len(pool) * ratio)))
        for card in pool[:k]:
            item = copy.deepcopy(card)
            if redact_provenance:
                pv = dict(item.get("provenance") or {})
                pv.pop("extracted_by", None)
                pv.pop("verified_by", None)
                item["provenance"] = pv
            sampled.append(item)
    return sampled


def find_best_quote_window(
    quote: str,
    source_text: str,
    *,
    window_slack: int = 40,
) -> dict[str, Any]:
    """Find the closest matching span for a provenance quote inside plain source text."""
    q = re.sub(r"\s+", " ", (quote or "").strip())
    if not q or not source_text:
        return {"found_exact": False, "similarity": 0.0, "matched_snippet": "", "span": [0, 0]}

    idx = source_text.find(q)
    if idx != -1:
        return {
            "found_exact": True,
            "similarity": 1.0,
            "matched_snippet": source_text[idx : idx + len(q)],
            "span": [idx, idx + len(q)],
        }

    # Slide a bounded window anchored on character-matching step for deterministic speed.
    q_comp = re.sub(r"\s+", "", q)
    win = min(len(source_text), len(q) + window_slack)
    step = max(4, len(q) // 4)
    best_sim = -1.0
    best_span = (0, min(len(source_text), win))
    for start in range(0, max(1, len(source_text) - win + 1), step):
        seg = source_text[start : start + win]
        seg_comp = re.sub(r"\s+", "", seg)
        sim = difflib.SequenceMatcher(None, q_comp, seg_comp).ratio()
        if sim > best_sim:
            best_sim = sim
            best_span = (start, start + win)

    s0, s1 = best_span
    return {
        "found_exact": False,
        "similarity": round(max(0.0, best_sim), 4),
        "matched_snippet": source_text[s0:s1],
        "span": [s0, s1],
    }


def deterministic_precheck(
    cards: Sequence[dict[str, Any]],
    *,
    source_texts: dict[str, str] | None = None,
    profile: DomainProfile | None = None,
    min_quote_similarity: float = 0.65,
) -> dict[str, Any]:
    """Zero-LLM deterministic precheck for schema health, quote grounding, and visual triggers."""
    prof = profile or default_generic_profile()
    triggers = set(prof.gates.visual_trigger_markers)
    trigger_docs = set(prof.gates.visual_trigger_docs)
    drift_terms = list(prof.gates.speculative_drift_terms)

    findings: list[dict[str, Any]] = []
    for card in cards:
        cid = str(card.get("card_id") or "<unknown>")
        pv = card.get("provenance") or {}
        doc = str(pv.get("source_doc") or "")
        quote = str(pv.get("quote") or "")
        schema_errs = validate_card(card, prof)

        quote_check = None
        if source_texts and doc in source_texts:
            quote_check = find_best_quote_window(quote, source_texts[doc])
            if not quote_check["found_exact"] and quote_check["similarity"] < min_quote_similarity:
                schema_errs.append(
                    f"low quote grounding similarity={quote_check['similarity']} < {min_quote_similarity}"
                )

        hit_markers = sorted(mk for mk in triggers if mk and mk in quote)
        if doc in trigger_docs and hit_markers:
            needs_visual = True
        else:
            needs_visual = bool(hit_markers)

        drift_hits = [term for term in drift_terms if term and term in str(card.get("judge_view") or "")]
        if drift_hits:
            schema_errs.append(f"speculative_drift_terms matched: {drift_hits}")

        if schema_errs or needs_visual:
            findings.append(
                {
                    "card_id": cid,
                    "errors": schema_errs,
                    "needs_visual_review": needs_visual,
                    "visual_markers": hit_markers,
                    "quote_check": quote_check,
                }
            )

    return {
        "total_checked": len(cards),
        "flagged_count": len(findings),
        "error_card_count": sum(1 for f in findings if f["errors"]),
        "visual_review_count": sum(1 for f in findings if f["needs_visual_review"]),
        "findings": findings,
    }
