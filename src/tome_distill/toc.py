"""Document Table-of-Contents enrichment and line-level closure verification."""

from __future__ import annotations

import re
from typing import Any, Sequence

from .text_utils import is_cjk_char


def _clean_line_chars(line: str) -> int:
    return sum(1 for ch in line if not ch.isspace())


def _cjk_line_chars(line: str) -> int:
    return sum(1 for ch in line if is_cjk_char(ch))


def calc_text_metrics(text: str, page_delimiter: str = "-----") -> dict[str, Any]:
    """Compute exact per-line and document-wide physical text metrics."""
    lines = text.splitlines()
    non_ws_per_line = [_clean_line_chars(ln) for ln in lines]
    cjk_per_line = [_cjk_line_chars(ln) for ln in lines]
    delim = page_delimiter.strip()
    page_marker_lines = [idx + 1 for idx, ln in enumerate(lines) if delim and ln.strip() == delim]

    return {
        "line_count": len(lines),
        "non_empty_lines": sum(1 for x in non_ws_per_line if x > 0),
        "char_count": sum(non_ws_per_line),
        "cjk_char_count": sum(cjk_per_line),
        "page_marker_count": len(page_marker_lines),
        "page_marker_lines": page_marker_lines,
        "_non_ws_per_line": non_ws_per_line,
        "_cjk_per_line": cjk_per_line,
    }


def _span_metrics(metrics: dict[str, Any], start_line: int, end_line: int) -> dict[str, int]:
    per_line = metrics["_non_ws_per_line"]
    cjk_line = metrics["_cjk_per_line"]
    if start_line < 1 or end_line < start_line or end_line > len(per_line):
        return {"line_count": 0, "char_count": 0, "cjk_char_count": 0}
    sl = slice(start_line - 1, end_line)
    return {
        "line_count": end_line - start_line + 1,
        "char_count": sum(per_line[sl]),
        "cjk_char_count": sum(cjk_line[sl]),
    }


def _iter_leaves(nodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    leaves: list[dict[str, Any]] = []
    for node in nodes:
        children = node.get("children") or []
        if children:
            leaves.extend(_iter_leaves(children))
        else:
            leaves.append(node)
    return leaves


def build_doc_toc(
    doc_id: str,
    nodes: Sequence[dict[str, Any]],
    text: str,
    *,
    page_delimiter: str = "-----",
) -> dict[str, Any]:
    """Attach physical line/char metrics (respecting explicit override_metrics) to a node tree."""
    doc_metrics = calc_text_metrics(text, page_delimiter=page_delimiter)

    def _enrich(raw_nodes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for nd in raw_nodes:
            item = dict(nd)
            children = nd.get("children") or []
            if children:
                enriched_children = _enrich(children)
                item["children"] = enriched_children
                starts = [int(c["start_line"]) for c in enriched_children if "start_line" in c]
                ends = [int(c["end_line"]) for c in enriched_children if "end_line" in c]
                if "start_line" not in item and starts:
                    item["start_line"] = min(starts)
                if "end_line" not in item and ends:
                    item["end_line"] = max(ends)

            s_line = int(item.get("start_line") or 0)
            e_line = int(item.get("end_line") or 0)
            computed = _span_metrics(doc_metrics, s_line, e_line)
            override = item.get("override_metrics")
            if isinstance(override, dict):
                merged = dict(computed)
                merged.update(override)
                item["metrics"] = merged
                item["has_override_metrics"] = True
            else:
                item["metrics"] = computed
                item["has_override_metrics"] = False
            out.append(item)
        return out

    enriched_nodes = _enrich(nodes)
    public_doc_metrics = {k: v for k, v in doc_metrics.items() if not k.startswith("_")}
    return {
        "doc_id": doc_id,
        "doc_metrics": public_doc_metrics,
        "nodes": enriched_nodes,
    }


def verify_document_closure(
    toc_doc: dict[str, Any],
    text: str,
    *,
    mode: str = "strict_line_partition",
    page_delimiter: str = "-----",
) -> dict[str, Any]:
    """Verify 100% line coverage and span consistency in strict or multicolumn-union mode."""
    if mode not in {"strict_line_partition", "multicolumn_line_union"}:
        raise ValueError(f"Unsupported closure mode: {mode!r}")

    doc_metrics = calc_text_metrics(text, page_delimiter=page_delimiter)
    total_lines = int(doc_metrics["line_count"])
    leaves = _iter_leaves(toc_doc.get("nodes") or [])

    errors: list[str] = []
    warnings: list[str] = []
    if not leaves:
        return {
            "passed": False,
            "mode": mode,
            "errors": ["TOC contains zero leaf nodes"],
            "warnings": warnings,
            "leaf_count": 0,
            "total_lines": total_lines,
        }

    coverage_counts: dict[int, int] = {ln: 0 for ln in range(1, total_lines + 1)}
    prev_start_by_group: dict[str, int] = {}
    override_nodes: list[str] = []

    for leaf in leaves:
        nid = str(leaf.get("node_id") or "<missing_node_id>")
        s_line = int(leaf.get("start_line") or 0)
        e_line = int(leaf.get("end_line") or 0)
        if s_line < 1 or e_line < s_line or e_line > total_lines:
            errors.append(f"leaf {nid!r} invalid line span [{s_line}, {e_line}] for 1..{total_lines}")
            continue

        order_group = str(leaf.get("column_id") or "__default__") if mode == "multicolumn_line_union" else "__default__"
        prev_s = prev_start_by_group.get(order_group, 0)
        if s_line < prev_s:
            errors.append(
                f"leaf {nid!r} out-of-order start_line={s_line} < previous {prev_s} in group {order_group!r}"
            )
        prev_start_by_group[order_group] = s_line

        for ln in range(s_line, e_line + 1):
            coverage_counts[ln] += 1

        if leaf.get("has_override_metrics") or isinstance(leaf.get("override_metrics"), dict):
            override_nodes.append(nid)
            warnings.append(
                f"leaf {nid!r} uses override_metrics in {mode}; verify effective char conservation manually"
            )

    missing_lines = sorted(ln for ln, cnt in coverage_counts.items() if cnt == 0)
    if missing_lines:
        errors.append(f"uncovered lines ({len(missing_lines)}): first few={missing_lines[:10]}")

    overlap_lines = sorted(ln for ln, cnt in coverage_counts.items() if cnt > 1)
    if mode == "strict_line_partition" and overlap_lines:
        errors.append(f"overlapping lines in strict mode ({len(overlap_lines)}): first few={overlap_lines[:10]}")
    elif mode == "multicolumn_line_union" and overlap_lines and not override_nodes:
        warnings.append(
            f"multicolumn_line_union shares {len(overlap_lines)} line(s) without override_metrics char split"
        )

    return {
        "passed": len(errors) == 0,
        "mode": mode,
        "errors": errors,
        "warnings": warnings,
        "leaf_count": len(leaves),
        "total_lines": total_lines,
        "total_chars": int(doc_metrics["char_count"]),
        "override_nodes": override_nodes,
    }


def render_toc_markdown(toc_doc: dict[str, Any]) -> str:
    """Render a compact Markdown outline table for human inspection."""
    dm = toc_doc.get("doc_metrics") or {}
    lines = [
        f"# Document TOC: `{toc_doc.get('doc_id', '')}`",
        "",
        f"- Lines: `{dm.get('line_count', 0)}` | Chars: `{dm.get('char_count', 0)}` | Page markers: `{dm.get('page_marker_count', 0)}`",
        "",
        "| Node ID | Title | Lines | Chars | Extract |",
        "| --- | --- | --- | ---: | :---: |",
    ]
    for leaf in _iter_leaves(toc_doc.get("nodes") or []):
        m = leaf.get("metrics") or {}
        lines.append(
            f"| `{leaf.get('node_id', '')}` | {leaf.get('title', '')} | "
            f"{leaf.get('start_line', '?')}–{leaf.get('end_line', '?')} | "
            f"{m.get('char_count', 0)} | `{leaf.get('extract', 'yes')}` |"
        )
    return "\n".join(lines) + "\n"
