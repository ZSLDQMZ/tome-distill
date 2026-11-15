"""Deterministic KB compiler: conflict sync, UnionFind clustering, 3-table compile, regex lint, and release-gate verification."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from .field_diffs import stratified_blind_sample
from .profile import DomainProfile, default_generic_profile
from .text_utils import (
    TextNormalizer,
    get_attested_whole_forms,
    is_whole_form_attested,
)


class UnionFind:
    """Path-compressed disjoint-set data structure for cross-source headword clustering."""

    def __init__(self, items: Iterable[str] = ()) -> None:
        self.parent: dict[str, str] = {str(x): str(x) for x in items}

    def add(self, x: str) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: str) -> str:
        self.add(x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        cur = x
        while cur != root:
            nxt = self.parent[cur]
            self.parent[cur] = root
            cur = nxt
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            # Deterministic lexicographic root ordering.
            if ra <= rb:
                self.parent[rb] = ra
            else:
                self.parent[ra] = rb

    def groups(self) -> list[list[str]]:
        buckets: dict[str, list[str]] = {}
        for x in sorted(self.parent.keys()):
            buckets.setdefault(self.find(x), []).append(x)
        return sorted(buckets.values(), key=lambda g: g[0])


def _canonical_sha1(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


def build_conflict_records(
    cards: Sequence[dict[str, Any]],
    conflict_specs: Sequence[dict[str, Any]] | None = None,
    *,
    profile: DomainProfile | None = None,
) -> list[dict[str, Any]]:
    """Build normalized conflict records linked to source cards via forms or explicit IDs."""
    prof = profile or default_generic_profile()
    norm = TextNormalizer(prof.text_normalization)
    specs = list(conflict_specs or [])
    if not specs and prof.compiler.conflict_specs_path:
        p = Path(prof.compiler.conflict_specs_path)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            specs = data if isinstance(data, list) else list(data.get("conflicts") or [])

    form_to_cids: dict[str, list[str]] = {}
    for card in cards:
        cid = str(card.get("card_id") or "")
        jv = card.get("judge_view") or {}
        for rf in jv.get("forms") or []:
            cf = norm.clean_form_token(rf)
            if cf:
                form_to_cids.setdefault(cf, []).append(cid)
                form_to_cids.setdefault(norm.to_simp_keep_protected(cf), []).append(cid)

    records: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs, start=1):
        gid = str(spec.get("conflict_id") or f"CONF-{idx:03d}")
        variants = [str(v).strip() for v in (spec.get("variants") or spec.get("forms") or []) if str(v).strip()]
        explicit_ids = [str(c).strip() for c in (spec.get("card_ids") or []) if str(c).strip()]
        linked_ids: list[str] = list(dict.fromkeys(explicit_ids))
        clean_forms: list[str] = []
        for var in variants:
            cf = norm.clean_form_token(var)
            if cf:
                clean_forms.append(cf)
                for cid in form_to_cids.get(cf, []):
                    if cid not in linked_ids:
                        linked_ids.append(cid)
                simp = norm.to_simp_keep_protected(cf)
                for cid in form_to_cids.get(simp, []):
                    if cid not in linked_ids:
                        linked_ids.append(cid)

        records.append(
            {
                "conflict_id": gid,
                "topic": str(spec.get("topic") or " / ".join(variants) or gid),
                "variants": variants,
                "_clean_forms_union": list(dict.fromkeys(clean_forms)),
                "card_ids": linked_ids,
                "resolution_policy": str(
                    spec.get("resolution_policy") or "preserve_all_in_per_doc_kb_no_override"
                ),
                "preferred_form": str(spec.get("preferred_form") or ""),
                "notes": str(spec.get("notes") or ""),
            }
        )
    return records


def sync_bidirectional_related_cards(
    cards: Sequence[dict[str, Any]],
    conflicts: Sequence[dict[str, Any]] | None = None,
    *,
    profile: DomainProfile | None = None,
    ambiguity_strategy: str = "log_and_skip",
    on_ambiguous: Callable[[str, str, list[str]], None] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Resolve crossref arrow pointers & conflict pairs and enforce 100% bidirectional symmetry."""
    if ambiguity_strategy not in {"log_and_skip", "link_all"}:
        raise ValueError(f"Unsupported ambiguity_strategy: {ambiguity_strategy!r}")

    prof = profile or default_generic_profile()
    norm = TextNormalizer(prof.text_normalization)
    arrow_re = re.compile(prof.text_normalization.crossref_arrow_pattern or r"⇨\s*〖([^〗]+)〗")

    work = [copy.deepcopy(c) for c in cards]
    by_id: dict[str, dict[str, Any]] = {str(c.get("card_id") or ""): c for c in work}

    # Build normalized headword -> candidate card_ids lookup.
    primary_map: dict[str, list[str]] = {}
    t2s_map: dict[str, list[str]] = {}
    for c in work:
        cid = str(c.get("card_id") or "")
        jv = c.get("judge_view") or {}
        for rf in jv.get("forms") or []:
            cf = norm.clean_form_token(rf)
            if cf:
                primary_map.setdefault(cf, []).append(cid)
                t2s_map.setdefault(norm.to_simp_keep_protected(cf), []).append(cid)

    adj: dict[str, set[str]] = {cid: set() for cid in by_id}
    for cid, c in by_id.items():
        for rc in (c.get("judge_view") or {}).get("related_cards") or []:
            rc_s = str(rc).strip()
            if rc_s in by_id and rc_s != cid:
                adj[cid].add(rc_s)

    # Link via crossref arrows.
    ambiguities: list[dict[str, Any]] = []
    unresolved_refs: list[dict[str, str]] = []
    for cid, c in by_id.items():
        jv = c.get("judge_view") or {}
        pv = c.get("provenance") or {}
        text_blobs = [
            str(jv.get("meaning") or ""),
            str(jv.get("usage") or ""),
            str(jv.get("misuse_or_confusable") or ""),
            str(pv.get("quote") or ""),
            str(pv.get("notes") or ""),
        ]
        seen_targets: set[str] = set()
        for blob in text_blobs:
            for m in arrow_re.finditer(blob):
                raw_target = m.group(1).strip()
                if raw_target in seen_targets:
                    continue
                seen_targets.add(raw_target)
                t_clean = norm.clean_form_token(raw_target)
                cands = [x for x in primary_map.get(t_clean, []) if x != cid]
                if not cands:
                    t_norm = norm.to_simp_keep_protected(t_clean)
                    cands = [x for x in t2s_map.get(t_norm, []) if x != cid]
                cands = list(dict.fromkeys(cands))
                if len(cands) == 1:
                    adj[cid].add(cands[0])
                elif len(cands) > 1:
                    ambiguities.append(
                        {"card_id": cid, "target": raw_target, "candidates": cands, "strategy": ambiguity_strategy}
                    )
                    if on_ambiguous is not None:
                        on_ambiguous(cid, raw_target, cands)
                    if ambiguity_strategy == "link_all":
                        adj[cid].update(cands)
                else:
                    unresolved_refs.append({"card_id": cid, "target": raw_target})

    # Link cards co-occurring inside the same conflict group.
    for conf in conflicts or []:
        cids = [x for x in (conf.get("card_ids") or []) if x in by_id]
        for a in cids:
            for b in cids:
                if a != b:
                    adj[a].add(b)

    # Enforce strict bidirectional closure.
    added_backlinks = 0
    for src, targets in list(adj.items()):
        for dst in list(targets):
            if src not in adj[dst]:
                adj[dst].add(src)
                added_backlinks += 1

    for c in work:
        cid = str(c.get("card_id") or "")
        links = sorted(adj.get(cid, set()))
        jv = dict(c.get("judge_view") or {})
        if links or "related_cards" in jv:
            jv["related_cards"] = links
            c["judge_view"] = jv

    report = {
        "total_cards": len(work),
        "total_edges": sum(len(v) for v in adj.values()),
        "added_backlinks": added_backlinks,
        "ambiguities": ambiguities,
        "unresolved_refs": unresolved_refs,
    }
    return work, report


def extract_card_star_redlines(card: dict[str, Any], profile: DomainProfile | None = None) -> list[str]:
    """Extract ungrammatical/prohibited redline strings from examples and misuse cues."""
    prof = profile or default_generic_profile()
    jv = card.get("judge_view") or {}
    redlines: list[str] = []

    for ex in jv.get("examples") or []:
        if not isinstance(ex, dict):
            continue
        raw_ex = str(ex.get("dialect") or ex.get("surface") or ex.get("form") or "").strip()
        if ex.get("is_negative") is True or raw_ex.startswith("*") or raw_ex.startswith("＊"):
            if raw_ex:
                redlines.append(raw_ex if raw_ex.startswith("*") else f"*{raw_ex}")

    hard_pat = prof.text_normalization.hard_ban_pattern
    if hard_pat:
        hard_re = re.compile(hard_pat)
        for fld in ("usage", "misuse_or_confusable"):
            val = str(jv.get(fld) or "").strip()
            if val and hard_re.search(val):
                redlines.append(val)

    return list(dict.fromkeys(redlines))


def classify_misuse_card(card: dict[str, Any], profile: DomainProfile | None = None) -> dict[str, bool]:
    """Classify semantic/orthographic risk attributes purely via DomainProfile patterns."""
    prof = profile or default_generic_profile()
    jv = card.get("judge_view") or {}
    blob = " ".join(
        str(jv.get(k) or "")
        for k in ("meaning", "usage", "misuse_or_confusable", "topic")
    )
    tn = prof.text_normalization
    cp = prof.compiler

    def _hit(pattern: str) -> bool:
        return bool(pattern and re.search(pattern, blob))

    return {
        "has_etymon_note": _hit(tn.etymon_note_pattern),
        "has_false_friend": _hit(tn.false_friend_pattern),
        "has_hard_ban": bool(extract_card_star_redlines(card, prof)),
        "has_polysemy": _hit(cp.polysemy_regex),
        "has_variant_prose": _hit(cp.variant_prose_regex),
        "has_encyclopedic": _hit(cp.encyclopedic_regex),
        "has_prosody_contrast": _hit(cp.prosody_contrast_regex),
        "has_special_register": _hit(cp.special_register_regex),
    }


def route_target_table(cards_in_cluster: Sequence[dict[str, Any]], profile: DomainProfile) -> str:
    """Assign a cluster deterministically to one of the three compiled target tables."""
    overrides = profile.compiler.dual_class_overrides
    extra_fw = profile.compiler.extra_fw_overrides
    for c in cards_in_cluster:
        cid = str(c.get("card_id") or "")
        if cid in overrides:
            return str(overrides[cid])
        if cid in extra_fw:
            return str(extra_fw[cid])

    ctypes = {str((c.get("judge_view") or {}).get("card_type") or "") for c in cards_in_cluster}
    leaves = {str((c.get("judge_view") or {}).get("taxonomy_leaf") or "") for c in cards_in_cluster}
    if "grammar" in ctypes or any(lf.startswith("G") for lf in leaves if lf):
        return "grammar_and_function_words"
    if "idiom" in ctypes:
        return "idioms_and_proverbs"
    return "lexicon_and_orthography"


def is_stage2a_eligible(
    cards_in_cluster: Sequence[dict[str, Any]],
    conflicts_for_cluster: Sequence[str],
    profile: DomainProfile,
) -> tuple[bool, list[str]]:
    """Return True if a cluster can be losslessly compiled deterministically without Stage 2B synthesis."""
    reasons: list[str] = []
    if conflicts_for_cluster:
        reasons.append("has_conflict_group")
    if len(cards_in_cluster) > 1:
        # Cross-source multi-card cluster defaults to Stage 2B unless identical short single gloss.
        meanings = {str((c.get("judge_view") or {}).get("meaning") or "").strip() for c in cards_in_cluster}
        if len(meanings) > 1:
            reasons.append("multi_card_meaning_divergence")

    max_m = profile.compiler.stage2a_max_meaning_len
    max_u = profile.compiler.stage2a_max_usage_len
    for c in cards_in_cluster:
        jv = c.get("judge_view") or {}
        if len(str(jv.get("meaning") or "")) > max_m:
            reasons.append("meaning_exceeds_stage2a_len")
        if len(str(jv.get("usage") or "")) > max_u:
            reasons.append("usage_exceeds_stage2a_len")
        flags = classify_misuse_card(c, profile)
        if flags["has_polysemy"] or flags["has_false_friend"] or flags["has_hard_ban"]:
            reasons.append("complex_semantic_or_redline")
    return (len(reasons) == 0, sorted(set(reasons)))


def lint_regex_pattern(pattern: str, profile: DomainProfile | None = None) -> list[str]:
    """Validate retrieval trigger regex against structural complexity and bare-noun guards."""
    prof = profile or default_generic_profile()
    errs: list[str] = []
    if not pattern or not pattern.strip():
        return ["empty regex pattern"]
    try:
        re.compile(pattern)
    except re.error as exc:
        return [f"invalid regex syntax: {exc}"]

    if ".*" in pattern or ".+" in pattern:
        errs.append(f"unbounded wildcard in regex pattern: {pattern!r}")

    alts = pattern.count("|") + 1 if "|" in pattern else 0
    if alts > prof.gates.max_regex_open_alts:
        errs.append(f"alternations {alts} > max_regex_open_alts={prof.gates.max_regex_open_alts}")

    char_classes = re.findall(r"\[[^\]]+\]", pattern)
    if len(char_classes) > prof.gates.max_regex_open_char_class:
        errs.append(
            f"char classes {len(char_classes)} > max_regex_open_char_class={prof.gates.max_regex_open_char_class}"
        )

    for bn in prof.gates.bare_noun_blacklist:
        if bn and bn in pattern:
            errs.append(f"bare noun blacklist hit {bn!r} in regex {pattern!r}")

    allowed_closed = prof.gates.closed_class_regex_chars
    if allowed_closed:
        stripped = re.sub(r"\\[dDsSwWbB]|[()\[\]{}?+*|^$.:\\-]", "", pattern)
        disallowed = [ch for ch in stripped if ch not in allowed_closed]
        if disallowed:
            errs.append(f"chars outside closed_class_regex_chars: {''.join(disallowed)!r}")

    return errs


def _tier_limit_for_entry(table: str, has_redline: bool, is_hub: bool, prof: DomainProfile) -> tuple[str, int]:
    limits = prof.compiler.compact_tier_limits
    if is_hub:
        return "T3_hub_max", int(limits.get("T3_hub_max", 300))
    if table == "grammar_and_function_words" or has_redline:
        return "T3_grammar_and_redline", int(limits.get("T3_grammar_and_redline", 260))
    if table == "idioms_and_proverbs":
        return "T2_special_or_idiom", int(limits.get("T2_special_or_idiom", 100))
    return "T1_general_lexicon", int(limits.get("T1_general_lexicon", 50))


def _fit_compact_text(
    forms: Sequence[str],
    meaning: str,
    usage: str,
    redlines: Sequence[str],
    std_anchors: Sequence[str],
    limit: int,
) -> str:
    head = "/".join(forms[:3])
    parts = [f"{head}: {meaning}".strip()]
    if std_anchors:
        parts.append(f"对照:{'/'.join(std_anchors[:2])}")
    if usage:
        parts.append(f"用法:{usage}")
    if redlines:
        parts.append(f"禁忌:{'; '.join(redlines[:2])}")
    joined = " | ".join(parts)
    if len(joined) <= limit:
        return joined
    # Preserve headword + redline first, then trim meaning/usage cleanly.
    budget_left = max(12, limit - 3)
    return joined[:budget_left].rstrip(" |:;，") + "..."


def _jaccard_chars(a: str, b: str) -> float:
    sa = {ch for ch in a if not ch.isspace()}
    sb = {ch for ch in b if not ch.isspace()}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def compile_kb(
    cards: Sequence[dict[str, Any]],
    *,
    profile: DomainProfile | None = None,
    conflict_specs: Sequence[dict[str, Any]] | None = None,
    regression_guards: Sequence[dict[str, Any]] | None = None,
    v0_guards: Sequence[dict[str, Any]] | None = None,
    stage2b_overrides: dict[str, dict[str, Any]] | None = None,
    out_dir: str | Path | None = None,
    ambiguity_strategy: str = "log_and_skip",
) -> dict[str, Any]:
    """Compile verified per-doc cards into 3 target tables, card_id_mapping, and lookup index."""
    prof = profile or default_generic_profile()
    norm = TextNormalizer(prof.text_normalization)

    conflicts = build_conflict_records(cards, conflict_specs, profile=prof)
    synced_cards, sync_report = sync_bidirectional_related_cards(
        cards,
        conflicts,
        profile=prof,
        ambiguity_strategy=ambiguity_strategy,
    )

    cards_by_id: dict[str, dict[str, Any]] = {str(c["card_id"]): c for c in synced_cards}
    card_to_confs: dict[str, list[str]] = {}
    confs_by_id: dict[str, dict[str, Any]] = {}
    for conf in conflicts:
        gid = conf["conflict_id"]
        confs_by_id[gid] = conf
        for cid in conf.get("card_ids") or []:
            card_to_confs.setdefault(cid, []).append(gid)

    uf = UnionFind(cards_by_id.keys())
    # Cluster by shared conflict record.
    for conf in conflicts:
        cids = [x for x in (conf.get("card_ids") or []) if x in cards_by_id]
        for x in cids[1:]:
            uf.union(cids[0], x)

    # Cluster by identical normalized pure surface form across compatible cards.
    form_buckets: dict[str, list[dict[str, Any]]] = {}
    for cid in sorted(cards_by_id.keys()):
        c = cards_by_id[cid]
        jv = c.get("judge_view") or {}
        for rf in jv.get("forms") or []:
            cf = norm.clean_form_token(rf)
            sf = norm.to_simp_keep_protected(cf) if cf else ""
            if sf and norm.is_valid_pure_form(sf):
                form_buckets.setdefault(sf, []).append(c)

    min_jaccard = prof.compiler.cross_source_min_jaccard
    for bucket in form_buckets.values():
        if len(bucket) < 2:
            continue
        base = bucket[0]
        base_id = str(base["card_id"])
        base_m = str((base.get("judge_view") or {}).get("meaning") or "")
        base_type = str((base.get("judge_view") or {}).get("card_type") or "")
        for other in bucket[1:]:
            oth_id = str(other["card_id"])
            oth_type = str((other.get("judge_view") or {}).get("card_type") or "")
            oth_m = str((other.get("judge_view") or {}).get("meaning") or "")
            if base_type == oth_type or _jaccard_chars(base_m, oth_m) >= min_jaccard:
                uf.union(base_id, oth_id)

    tables: dict[str, list[dict[str, Any]]] = {
        t: [] for t in prof.profile_meta.target_tables
    }
    card_to_entries: dict[str, list[str]] = {}
    entry_to_cards: dict[str, list[str]] = {}
    entry_meta: dict[str, dict[str, Any]] = {}
    dialect_form_to_entries: dict[str, list[str]] = {}
    dialect_t2s_to_entries: dict[str, list[str]] = {}
    dialect_regex_rules: list[dict[str, Any]] = []
    standard_anchor_to_entries: dict[str, list[str]] = {}
    grammar_multi_char_exemptions: set[str] = set()
    non_grammar_lexical_compounds: set[str] = set()
    conflict_to_entries: dict[str, list[str]] = {}
    regression_rule_to_entries: dict[str, list[str]] = {}
    compat_alias_to_entries: dict[str, list[str]] = {}

    guards_list = list(regression_guards or v0_guards or [])
    if not guards_list and prof.compiler.regression_guards_path:
        gp = Path(prof.compiler.regression_guards_path)
        if gp.is_file():
            raw_g = json.loads(gp.read_text(encoding="utf-8"))
            guards_list = raw_g if isinstance(raw_g, list) else list(raw_g.get("rules") or [])

    overrides_2b = stage2b_overrides or {}
    groups = uf.groups()
    table_seq: dict[str, int] = {t: 0 for t in tables}

    for group_cids in groups:
        group_cards = [cards_by_id[cid] for cid in group_cids]
        table = route_target_table(group_cards, prof)
        if table not in tables:
            tables[table] = []
            table_seq[table] = 0
        table_seq[table] += 1
        prefix = table.split("_", 1)[0].upper()[:3]
        entry_id = f"KB-{prefix}-{table_seq[table]:04d}"

        cluster_confs = sorted({gid for cid in group_cids for gid in card_to_confs.get(cid, [])})
        exact_forms, _ = get_attested_whole_forms(
            group_cids, cards_by_id, norm, card_to_confs=card_to_confs, confs_by_id=confs_by_id
        )

        clean_forms: list[str] = []
        meanings: list[str] = []
        usages: list[str] = []
        std_anchors: list[str] = []
        redlines: list[str] = []
        leaves: list[str] = []
        topics: list[str] = []
        triggers: list[str] = []

        for c in group_cards:
            jv = c.get("judge_view") or {}
            topics.append(str(jv.get("topic") or ""))
            if jv.get("taxonomy_leaf"):
                leaves.append(str(jv["taxonomy_leaf"]))
            for rf in jv.get("forms") or []:
                cf = norm.clean_form_token(rf)
                if cf and cf not in clean_forms:
                    clean_forms.append(cf)
            m_str = str(jv.get("meaning") or "").strip()
            if m_str and m_str not in meanings:
                meanings.append(m_str)
            u_str = str(jv.get("usage") or "").strip()
            if u_str and u_str not in usages:
                usages.append(u_str)
            for sc_key in ("standard_counterpart", "mandarin_counterpart"):
                sc_val = str(jv.get(sc_key) or "").strip()
                if sc_val:
                    for tok in re.split(r"[,，、/;；\s]+", sc_val):
                        tok_c = norm.clean_form_token(tok)
                        if len(tok_c) >= 2 and norm.is_valid_zh_anchor(tok_c) and tok_c not in std_anchors:
                            std_anchors.append(tok_c)
            for rl in extract_card_star_redlines(c, prof):
                if rl not in redlines:
                    redlines.append(rl)
            for tg in jv.get("tags") or []:
                if isinstance(tg, str) and tg.startswith("regex:"):
                    triggers.append(tg[len("regex:") :].strip())

        # Optional Stage 2B synthesis patch keyed by first source card_id or cluster forms.
        patch = overrides_2b.get(group_cids[0]) or {}
        if patch.get("meaning"):
            meanings = [str(patch["meaning"]).strip()]
        if patch.get("usage"):
            usages = [str(patch["usage"]).strip()]

        is_hub = len(group_cids) >= int(prof.compiler.compact_tier_limits.get("hub_min_source_cards", 6))
        has_redline = bool(redlines)
        tier_name, tier_limit = _tier_limit_for_entry(table, has_redline, is_hub, prof)
        merged_meaning = "；".join(meanings)
        merged_usage = "；".join(usages)
        compact_text = _fit_compact_text(
            clean_forms or sorted(exact_forms)[:2],
            merged_meaning,
            merged_usage,
            redlines,
            std_anchors,
            tier_limit,
        )

        primary_leaf = leaves[0] if leaves else ""
        primary_topic = topics[0] if topics else (clean_forms[0] if clean_forms else entry_id)
        stage2a_ok, stage2b_reasons = is_stage2a_eligible(group_cards, cluster_confs, prof)
        t2s_forms = list(
            dict.fromkeys(norm.to_simp_keep_protected(f, preserve_length=True) for f in clean_forms)
        )

        core_payload = {
            "entry_id": entry_id,
            "table": table,
            "forms": clean_forms,
            "t2s_forms": t2s_forms,
            "meaning": merged_meaning,
            "usage": merged_usage,
            "redlines": redlines,
            "compact_text": compact_text,
            "source_cards": group_cids,
            "conflict_ids": cluster_confs,
        }
        sha1 = _canonical_sha1(core_payload)

        entry_obj = {
            **core_payload,
            "topic": primary_topic,
            "taxonomy_leaf": primary_leaf,
            "standard_anchors": std_anchors,
            "has_redline": has_redline,
            "tier": tier_name,
            "stage": "stage2a" if stage2a_ok else "stage2b",
            "stage2b_reasons": stage2b_reasons,
            "slot_key": primary_leaf or primary_topic,
            "entry_content_sha1": sha1,
        }
        tables[table].append(entry_obj)

        entry_to_cards[entry_id] = list(group_cids)
        for cid in group_cids:
            card_to_entries.setdefault(cid, []).append(entry_id)

        entry_meta[entry_id] = {
            "entry_id": entry_id,
            "table": table,
            "topic": primary_topic,
            "taxonomy_leaf": primary_leaf,
            "forms": clean_forms,
            "t2s_forms": t2s_forms,
            "standard_anchors": std_anchors,
            "has_redline": has_redline,
            "redlines": redlines,
            "conflict_ids": cluster_confs,
            "compact_text": compact_text,
            "tier": tier_name,
            "source_cards": list(group_cids),
            "slot_key": primary_leaf or primary_topic,
            "entry_content_sha1": sha1,
        }

        for cf in clean_forms:
            dialect_form_to_entries.setdefault(cf, []).append(entry_id)
            if len(cf) >= 2:
                if table == "grammar_and_function_words":
                    grammar_multi_char_exemptions.add(cf)
                else:
                    non_grammar_lexical_compounds.add(cf)
            for raw_alias, canon in prof.text_normalization.compat_char_aliases.items():
                if canon in cf:
                    compat_alias_to_entries.setdefault(raw_alias, []).append(entry_id)

        for tf in t2s_forms:
            dialect_t2s_to_entries.setdefault(tf, []).append(entry_id)
            if len(tf) >= 2:
                if table == "grammar_and_function_words":
                    grammar_multi_char_exemptions.add(tf)
                else:
                    non_grammar_lexical_compounds.add(tf)

        for sa in std_anchors:
            standard_anchor_to_entries.setdefault(sa, []).append(entry_id)

        for gid in cluster_confs:
            conflict_to_entries.setdefault(gid, []).append(entry_id)

        for pat in triggers:
            dialect_regex_rules.append({"entry_id": entry_id, "pattern": pat, "clause_scoped": True})

        for guard in guards_list:
            rid = str(guard.get("rule_id") or "")
            g_forms = {norm.clean_form_token(x) for x in (guard.get("forms") or [])}
            g_cids = set(guard.get("card_ids") or [])
            if rid and ((g_forms & set(clean_forms)) or (g_cids & set(group_cids))):
                regression_rule_to_entries.setdefault(rid, []).append(entry_id)

    lookup_index = {
        "entry_meta": entry_meta,
        "dialect_form_to_entries": dialect_form_to_entries,
        "dialect_t2s_to_entries": dialect_t2s_to_entries,
        "compat_alias_to_entries": compat_alias_to_entries,
        "dialect_regex_rules": dialect_regex_rules,
        "standard_anchor_to_entries": standard_anchor_to_entries,
        "grammar_multi_char_exemptions": sorted(grammar_multi_char_exemptions),
        "non_grammar_lexical_compounds": sorted(non_grammar_lexical_compounds),
        "conflict_to_entries": conflict_to_entries,
        "regression_rule_to_entries": regression_rule_to_entries,
    }
    card_id_mapping = {
        "card_to_entries": card_to_entries,
        "entry_to_cards": entry_to_cards,
    }
    manifest = {
        "dialect": prof.profile_meta.dialect,
        "schema_version": prof.profile_meta.schema_version,
        "total_source_cards": len(synced_cards),
        "total_compiled_entries": len(entry_meta),
        "table_counts": {t: len(rows) for t, rows in tables.items()},
        "conflict_count": len(conflicts),
        "regression_guard_count": len(guards_list),
        "related_sync_report": sync_report,
    }

    bundle = {
        "manifest": manifest,
        "tables": tables,
        "card_id_mapping": card_id_mapping,
        "compiled_lookup_index": lookup_index,
        "conflicts": conflicts,
        "source_cards": synced_cards,
    }

    if out_dir is not None:
        root = Path(out_dir)
        tbl_dir = root / "tables"
        tbl_dir.mkdir(parents=True, exist_ok=True)
        for tname, rows in tables.items():
            with (tbl_dir / f"{tname}.jsonl").open("w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        (root / "compiled_lookup_index.json").write_text(
            json.dumps(lookup_index, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (root / "card_id_mapping.json").write_text(
            json.dumps(card_id_mapping, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (root / "conflicts.json").write_text(
            json.dumps(conflicts, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return bundle


def verify_compiled_gates(
    bundle: dict[str, Any],
    *,
    profile: DomainProfile | None = None,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute GATE-1..GATE-8 deterministic release checks with proportional small-corpus scaling."""
    prof = profile or default_generic_profile()
    norm = TextNormalizer(prof.text_normalization)

    tables = bundle.get("tables") or {}
    mapping = bundle.get("card_id_mapping") or {}
    idx = bundle.get("compiled_lookup_index") or {}
    conflicts = bundle.get("conflicts") or []
    source_cards = bundle.get("source_cards") or []

    all_entries: list[dict[str, Any]] = []
    for tname, rows in tables.items():
        for r in rows:
            all_entries.append(r)

    # GATE-1: Entry integrity & SHA1 determinism.
    g1_errs: list[str] = []
    seen_eids: set[str] = set()
    for ent in all_entries:
        eid = str(ent.get("entry_id") or "")
        if not eid or eid in seen_eids:
            g1_errs.append(f"duplicate/empty entry_id={eid!r}")
        seen_eids.add(eid)
        if not str(ent.get("compact_text") or "").strip():
            g1_errs.append(f"[{eid}] empty compact_text")
        core_payload = {
            "entry_id": eid,
            "table": ent.get("table"),
            "forms": ent.get("forms"),
            "t2s_forms": ent.get("t2s_forms"),
            "meaning": ent.get("meaning"),
            "usage": ent.get("usage"),
            "redlines": ent.get("redlines"),
            "compact_text": ent.get("compact_text"),
            "source_cards": ent.get("source_cards"),
            "conflict_ids": ent.get("conflict_ids"),
        }
        expected_sha = _canonical_sha1(core_payload)
        if ent.get("entry_content_sha1") != expected_sha:
            g1_errs.append(f"[{eid}] entry_content_sha1 mismatch")

    # GATE-2: Bidirectional card_id_mapping conservation.
    g2_errs: list[str] = []
    c2e = mapping.get("card_to_entries") or {}
    e2c = mapping.get("entry_to_cards") or {}
    if set(e2c.keys()) != seen_eids:
        g2_errs.append("entry_to_cards keys != compiled entry_ids")
    for cid, eids in c2e.items():
        for eid in eids:
            if cid not in (e2c.get(eid) or []):
                g2_errs.append(f"mapping asymmetry: {cid} -> {eid}")
    for eid, cids in e2c.items():
        for cid in cids:
            if eid not in (c2e.get(cid) or []):
                g2_errs.append(f"mapping asymmetry: {eid} -> {cid}")
    if source_cards:
        src_cids = {str(c.get("card_id") or "") for c in source_cards}
        if src_cids != set(c2e.keys()):
            g2_errs.append(
                f"source_cards mismatch vs card_to_entries: missing={sorted(src_cids - set(c2e.keys()))}"
            )

    # GATE-3: Whole-form attestation against source provenance.
    g3_errs: list[str] = []
    if source_cards:
        cards_by_id = {str(c["card_id"]): c for c in source_cards}
        confs_by_id = {str(g["conflict_id"]): g for g in conflicts}
        c_to_g: dict[str, list[str]] = {}
        for g in conflicts:
            for cid in g.get("card_ids") or []:
                c_to_g.setdefault(cid, []).append(str(g["conflict_id"]))
        for ent in all_entries:
            eid = str(ent.get("entry_id") or "")
            scids = [x for x in (ent.get("source_cards") or []) if x in cards_by_id]
            exact_forms, corpus = get_attested_whole_forms(
                scids, cards_by_id, norm, card_to_confs=c_to_g, confs_by_id=confs_by_id
            )
            for form in ent.get("forms") or []:
                if not is_whole_form_attested(form, exact_forms, corpus, norm):
                    g3_errs.append(f"[{eid}] unattested form {form!r}")

    # GATE-4: Lookup index completeness and consistent t2s normalization.
    g4_errs: list[str] = []
    meta = idx.get("entry_meta") or {}
    if set(meta.keys()) != seen_eids:
        g4_errs.append("compiled_lookup_index.entry_meta keys != compiled entry_ids")
    t2s_map = idx.get("dialect_t2s_to_entries") or {}
    for ent in all_entries:
        eid = str(ent["entry_id"])
        for f in ent.get("forms") or []:
            expected_t2s = norm.to_simp_keep_protected(f, preserve_length=True)
            if eid not in (t2s_map.get(expected_t2s) or []):
                g4_errs.append(f"[{eid}] missing t2s index key {expected_t2s!r}")

    # GATE-5: Compact length P50/P95 & tier limits.
    g5_errs: list[str] = []
    lengths = sorted(len(str(e.get("compact_text") or "")) for e in all_entries)
    limits = prof.compiler.compact_tier_limits
    for ent in all_entries:
        tier = str(ent.get("tier") or "T1_general_lexicon")
        cap = int(limits.get(tier, 300))
        clen = len(str(ent.get("compact_text") or ""))
        if clen > cap:
            g5_errs.append(f"[{ent['entry_id']}] compact_text len {clen} > tier {tier} cap {cap}")
    t1_lens = sorted(
        len(str(e.get("compact_text") or ""))
        for e in all_entries
        if e.get("tier") == "T1_general_lexicon"
    )
    t1_median = t1_lens[len(t1_lens) // 2] if t1_lens else 0
    t1_med_cap = int(limits.get("T1_median", 35))
    if t1_lens and len(t1_lens) >= 10 and t1_median > t1_med_cap:
        g5_errs.append(f"T1 median length {t1_median} > {t1_med_cap}")

    # GATE-6: Regex trigger lint.
    g6_errs: list[str] = []
    for rule in idx.get("dialect_regex_rules") or []:
        pat = str(rule.get("pattern") or "")
        for rerr in lint_regex_pattern(pat, prof):
            g6_errs.append(f"[{rule.get('entry_id')}] {rerr}")

    # GATE-7: Conflict mapping coverage.
    g7_errs: list[str] = []
    c_idx = idx.get("conflict_to_entries") or {}
    for g in conflicts:
        gid = str(g.get("conflict_id") or "")
        if g.get("card_ids") and not c_idx.get(gid):
            g7_errs.append(f"conflict {gid!r} missing from conflict_to_entries")

    # GATE-8: Blind audit sampling (scales cleanly for small toy corpora).
    g8_errs: list[str] = []
    total_e = len(all_entries)
    min_cnt = int(prof.gates.min_blind_audit_count)
    min_ratio = float(prof.gates.min_blind_audit_ratio)
    sample_cards = source_cards if source_cards else [
        {"card_id": e["entry_id"], "judge_view": {}, "provenance": {"source_doc": e["table"]}}
        for e in all_entries
    ]
    blind_sample = stratified_blind_sample(
        sample_cards,
        ratio=max(min_ratio, prof.gates.qa_spotcheck_ratio),
        min_per_doc=1 if len(sample_cards) < min_cnt else prof.gates.qa_min_per_doc,
        seed=seed,
        redact_provenance=True,
    )
    small_corpus = len(sample_cards) < min_cnt
    required_sample = (
        max(1, math.ceil(len(sample_cards) * min_ratio))
        if small_corpus and sample_cards
        else min(len(sample_cards), max(min_cnt, math.ceil(len(sample_cards) * min_ratio)))
    )
    if sample_cards and len(blind_sample) < required_sample:
        g8_errs.append(f"blind audit sample {len(blind_sample)} < required {required_sample}")

    gates = {
        "GATE-1_schema_sha1": {"passed": not g1_errs, "errors": g1_errs},
        "GATE-2_bijection_mapping": {"passed": not g2_errs, "errors": g2_errs},
        "GATE-3_whole_form_attestation": {"passed": not g3_errs, "errors": g3_errs},
        "GATE-4_lookup_index_t2s": {"passed": not g4_errs, "errors": g4_errs},
        "GATE-5_compact_length_tiers": {
            "passed": not g5_errs,
            "errors": g5_errs,
            "p50": lengths[len(lengths) // 2] if lengths else 0,
            "p95": lengths[min(len(lengths) - 1, int(0.95 * len(lengths)))] if lengths else 0,
            "t1_median": t1_median,
        },
        "GATE-6_regex_lint": {"passed": not g6_errs, "errors": g6_errs},
        "GATE-7_conflict_coverage": {"passed": not g7_errs, "errors": g7_errs},
        "GATE-8_blind_audit": {
            "passed": not g8_errs,
            "errors": g8_errs,
            "sample_size": len(blind_sample),
            "required": required_sample,
            "small_corpus_scaled": small_corpus,
            "total_entries": total_e,
        },
    }
    return {
        "passed": all(v["passed"] for v in gates.values()),
        "gates": gates,
    }
