"""Dual-layer card schema validation, meta-leak guards, batch gates, and ordered merge."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

from .profile import DomainProfile, default_generic_profile
from .text_utils import find_bad_chars_in_obj

ALLOWED_TOP_KEYS: frozenset[str] = frozenset({"card_id", "judge_view", "provenance"})
REQUIRED_TOP_KEYS: frozenset[str] = ALLOWED_TOP_KEYS

ALLOWED_JUDGE_KEYS: frozenset[str] = frozenset(
    {
        "card_type",
        "topic",
        "forms",
        "meaning",
        "usage",
        "examples",
        "standard_counterpart",
        "mandarin_counterpart",
        "misuse_or_confusable",
        "taxonomy_leaf",
        "confidence",
        "related_cards",
        "tags",
    }
)
REQUIRED_JUDGE_KEYS: frozenset[str] = frozenset(
    {"card_type", "topic", "forms", "meaning", "confidence"}
)

ALLOWED_PROV_KEYS: frozenset[str] = frozenset(
    {
        "source_doc",
        "node_id",
        "section",
        "page",
        "quote",
        "notes",
        "extracted_by",
        "verified_by",
        "audit_status",
    }
)
REQUIRED_PROV_KEYS: frozenset[str] = frozenset(
    {"source_doc", "page", "quote", "extracted_by", "verified_by"}
)

STANDARD_CARD_TYPES: frozenset[str] = frozenset(
    {"vocab", "grammar", "idiom", "orthography", "conflict_resolution"}
)
VALID_CONFIDENCE: frozenset[str] = frozenset({"high", "medium", "low"})

_NEGATIVE_NOTE_RE = re.compile(r"(?:病句|误说|误例|误用|不可说|不能说|反例|negative|ungrammatical)")


def _compile_meta_patterns(profile: DomainProfile) -> list[tuple[str, re.Pattern[str]]]:
    patterns = profile.text_normalization.meta_leak_patterns
    compiled: list[tuple[str, re.Pattern[str]]] = []
    for item in patterns:
        if isinstance(item, dict):
            pat = str(item.get("pattern") or "")
            name = str(item.get("name") or pat)
            ignore_case = bool(item.get("ignore_case", False))
        else:
            pat = str(getattr(item, "pattern", "") or "")
            name = str(getattr(item, "name", "") or pat)
            ignore_case = bool(getattr(item, "ignore_case", False))
        if not pat:
            continue
        flags = re.IGNORECASE if ignore_case else 0
        compiled.append((name, re.compile(pat, flags)))
    return compiled


def _iter_judge_text_fields(jv: dict[str, Any]) -> Iterable[tuple[str, str]]:
    for key in (
        "topic",
        "meaning",
        "usage",
        "standard_counterpart",
        "mandarin_counterpart",
        "misuse_or_confusable",
    ):
        val = jv.get(key)
        if isinstance(val, str) and val:
            yield (f"judge_view.{key}", val)
    for idx, form in enumerate(jv.get("forms") or []):
        if isinstance(form, str) and form:
            yield (f"judge_view.forms[{idx}]", form)
    for idx, ex in enumerate(jv.get("examples") or []):
        if isinstance(ex, dict):
            for ex_key in ("dialect", "surface", "form", "standard", "gloss", "note"):
                val = ex.get(ex_key)
                if isinstance(val, str) and val:
                    yield (f"judge_view.examples[{idx}].{ex_key}", val)


def find_star_redline_violations(card: dict[str, Any]) -> list[str]:
    """Detect unannotated '*' redline markers in forms, meaning, or examples (leading or inline)."""
    errs: list[str] = []
    jv = card.get("judge_view") or {}
    for idx, form in enumerate(jv.get("forms") or []):
        if isinstance(form, str) and ("*" in form or "＊" in form):
            errs.append(f"unannotated '*' marker in judge_view.forms[{idx}]: {form!r}")
    for idx, ex in enumerate(jv.get("examples") or []):
        if not isinstance(ex, dict):
            continue
        if ex.get("is_negative") is True:
            continue
        note = str(ex.get("note") or "")
        if _NEGATIVE_NOTE_RE.search(note):
            continue
        for k in ("dialect", "surface", "form", "standard", "gloss"):
            val = str(ex.get(k) or "")
            if "*" in val or "＊" in val:
                errs.append(
                    f"unmarked '*' in judge_view.examples[{idx}].{k} (set is_negative=true or negative note): {val!r}"
                )
    return errs


def validate_card(
    card: Any,
    profile: DomainProfile | None = None,
    *,
    require_distinct_verifiers: bool = True,
    check_meta_leaks: bool = True,
) -> list[str]:
    """Validate a single dual-layer lexicon card and return human-readable error strings."""
    prof = profile or default_generic_profile()
    errors: list[str] = []
    if not isinstance(card, dict):
        return ["card must be a JSON object"]

    # Check raw control/placeholder chars BEFORE any json.dumps serialization.
    for bad in find_bad_chars_in_obj(card):
        errors.append(f"bad_char: {bad}")

    top_keys = set(card.keys())
    missing_top = sorted(REQUIRED_TOP_KEYS - top_keys)
    extra_top = sorted(top_keys - ALLOWED_TOP_KEYS)
    if missing_top:
        errors.append(f"missing top-level keys: {missing_top}")
    if extra_top:
        errors.append(f"unexpected top-level keys: {extra_top}")

    cid = card.get("card_id")
    if not isinstance(cid, str) or not cid.strip():
        errors.append("card_id must be a non-empty string")

    jv = card.get("judge_view")
    pv = card.get("provenance")
    if not isinstance(jv, dict):
        errors.append("judge_view must be an object")
        return errors
    if not isinstance(pv, dict):
        errors.append("provenance must be an object")
        return errors

    jv_keys = set(jv.keys())
    missing_jv = sorted(REQUIRED_JUDGE_KEYS - jv_keys)
    extra_jv = sorted(jv_keys - ALLOWED_JUDGE_KEYS)
    if missing_jv:
        errors.append(f"missing judge_view keys: {missing_jv}")
    if extra_jv:
        errors.append(f"unexpected judge_view keys: {extra_jv}")

    pv_keys = set(pv.keys())
    missing_pv = sorted(REQUIRED_PROV_KEYS - pv_keys)
    extra_pv = sorted(pv_keys - ALLOWED_PROV_KEYS)
    if missing_pv:
        errors.append(f"missing provenance keys: {missing_pv}")
    if extra_pv:
        errors.append(f"unexpected provenance keys: {extra_pv}")

    ctype = jv.get("card_type")
    if ctype not in STANDARD_CARD_TYPES:
        errors.append(f"invalid card_type: {ctype!r}")

    conf = jv.get("confidence")
    if conf not in VALID_CONFIDENCE:
        errors.append(f"invalid confidence: {conf!r}")

    forms = jv.get("forms")
    if not isinstance(forms, list) or not forms or not all(isinstance(f, str) and f.strip() for f in forms):
        errors.append("judge_view.forms must be a non-empty list of strings")

    for req_str in ("topic", "meaning"):
        val = jv.get(req_str)
        if not isinstance(val, str) or not val.strip():
            errors.append(f"judge_view.{req_str} must be a non-empty string")

    examples = jv.get("examples")
    if examples is not None:
        if not isinstance(examples, list) or not all(isinstance(x, dict) for x in examples):
            errors.append("judge_view.examples must be a list of objects")

    related = jv.get("related_cards")
    if related is not None:
        if not isinstance(related, list) or not all(isinstance(x, str) and x.strip() for x in related):
            errors.append("judge_view.related_cards must be a list of strings")

    leaf = jv.get("taxonomy_leaf")
    valid_leaves = prof.profile_meta.valid_leaf_ids()
    if leaf and valid_leaves and leaf not in valid_leaves:
        errors.append(f"unknown taxonomy_leaf {leaf!r}; valid={sorted(valid_leaves)}")

    for pkey in ("source_doc", "quote", "extracted_by", "verified_by"):
        pval = pv.get(pkey)
        if not isinstance(pval, str) or not pval.strip():
            errors.append(f"provenance.{pkey} must be a non-empty string")

    ext_by = str(pv.get("extracted_by") or "").strip()
    ver_by = str(pv.get("verified_by") or "").strip()
    if require_distinct_verifiers and ext_by and ver_by and ext_by == ver_by:
        errors.append(f"dual-blind violation: extracted_by == verified_by == {ext_by!r}")

    if check_meta_leaks:
        compiled = _compile_meta_patterns(prof)
        for path_label, txt in _iter_judge_text_fields(jv):
            for pname, pat in compiled:
                m = pat.search(txt)
                if m:
                    errors.append(f"meta_leak ({pname}) in {path_label}: matched {m.group(0)!r}")

    errors.extend(find_star_redline_violations(card))
    return errors


def _validate_coverage_bijection(
    card_ids: list[str],
    coverage: dict[str, Any],
    expected_node_ids: Iterable[str] | None = None,
) -> list[str]:
    errs: list[str] = []
    node_to_cards = coverage.get("node_to_cards") or {}
    empty_nodes = list(coverage.get("empty_nodes") or [])
    if not isinstance(node_to_cards, dict):
        return ["coverage.node_to_cards must be a mapping"]

    overlap = sorted(set(node_to_cards.keys()) & set(empty_nodes))
    if overlap:
        errs.append(f"coverage nodes present in both node_to_cards and empty_nodes: {overlap}")

    mapped_ids: list[str] = []
    for nid, cids in node_to_cards.items():
        if not isinstance(cids, list):
            errs.append(f"coverage.node_to_cards[{nid!r}] must be a list")
            continue
        if not cids:
            errs.append(f"coverage.node_to_cards[{nid!r}] is empty; move to empty_nodes")
        mapped_ids.extend(str(c) for c in cids)

    if len(mapped_ids) != len(set(mapped_ids)):
        errs.append("coverage.node_to_cards assigns at least one card_id to multiple nodes")

    card_id_set = set(card_ids)
    mapped_set = set(mapped_ids)
    if card_id_set != mapped_set:
        missing = sorted(card_id_set - mapped_set)
        orphan = sorted(mapped_set - card_id_set)
        errs.append(f"coverage bijection failed: unmapped_cards={missing}, orphan_cards={orphan}")

    if expected_node_ids is not None:
        exp = set(expected_node_ids)
        seen_nodes = set(node_to_cards.keys()) | set(empty_nodes)
        if exp != seen_nodes:
            errs.append(
                f"coverage node mismatch: missing={sorted(exp - seen_nodes)}, extra={sorted(seen_nodes - exp)}"
            )
    return errs


def inspect_batch(
    cards: Sequence[dict[str, Any]],
    profile: DomainProfile | None = None,
    *,
    coverage: dict[str, Any] | None = None,
    expected_node_ids: Iterable[str] | None = None,
    require_distinct_verifiers: bool = True,
) -> dict[str, Any]:
    """Run batch-level schema, meta-leak, bad-char, dual-blind, and coverage bijection gates."""
    prof = profile or default_generic_profile()
    errors: list[str] = []
    warnings: list[str] = []
    seen_ids: set[str] = set()
    ordered_ids: list[str] = []

    for idx, card in enumerate(cards):
        cid = str((card or {}).get("card_id") or f"<idx:{idx}>")
        if cid in seen_ids:
            errors.append(f"duplicate card_id in batch: {cid!r}")
        seen_ids.add(cid)
        ordered_ids.append(cid)
        for err in validate_card(
            card,
            prof,
            require_distinct_verifiers=require_distinct_verifiers,
        ):
            errors.append(f"[{cid}] {err}")

    if coverage is not None:
        errors.extend(_validate_coverage_bijection(ordered_ids, coverage, expected_node_ids))
    elif expected_node_ids is not None:
        warnings.append("expected_node_ids provided without coverage object; node bijection skipped")

    return {
        "passed": len(errors) == 0,
        "total_cards": len(cards),
        "error_count": len(errors),
        "errors": errors,
        "warnings": warnings,
    }


def _load_cards_source(src: Path | Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    if isinstance(src, (str, Path)):
        p = Path(src)
        out: list[dict[str, Any]] = []
        for raw_line in p.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if line:
                out.append(json.loads(line))
        return out
    return [dict(c) for c in src]


def merge_per_doc_cards(
    sources: Sequence[Path | Sequence[dict[str, Any]]],
    *,
    node_order: Sequence[str] | None = None,
    coverage: dict[str, Any] | None = None,
    profile: DomainProfile | None = None,
    out_jsonl: Path | None = None,
) -> dict[str, Any]:
    """Deterministically order, deduplicate-check, re-validate, and optionally emit per-doc JSONL."""
    all_cards: list[dict[str, Any]] = []
    for src in sources:
        all_cards.extend(_load_cards_source(src))

    if node_order:
        rank = {nid: idx for idx, nid in enumerate(node_order)}
        fallback = len(rank) + 1
        card_to_node: dict[str, str] = {}
        if coverage and isinstance(coverage.get("node_to_cards"), dict):
            for nid, cids in coverage["node_to_cards"].items():
                for cid in cids:
                    card_to_node[str(cid)] = str(nid)

        def _sort_key(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
            orig_idx, c = item
            cid = str(c.get("card_id") or "")
            nid = str((c.get("provenance") or {}).get("node_id") or card_to_node.get(cid) or "")
            return (rank.get(nid, fallback), orig_idx)

        all_cards = [c for _, c in sorted(enumerate(all_cards), key=_sort_key)]

    report = inspect_batch(
        all_cards,
        profile=profile,
        coverage=coverage,
        expected_node_ids=node_order,
    )
    if out_jsonl is not None:
        if not report["passed"]:
            raise ValueError(f"Refusing to write merged JSONL; gate failed: {report['errors'][:5]}")
        out_path = Path(out_jsonl)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for card in all_cards:
                f.write(json.dumps(card, ensure_ascii=False) + "\n")

    return {"cards": all_cards, "gate_report": report}
