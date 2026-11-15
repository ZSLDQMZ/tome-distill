"""Pluggable LayoutAdapter interfaces and DoubleColumnDictAdapter for multi-column lexicons."""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from typing import Any, Sequence

from .profile import DomainProfile, default_generic_profile


class LayoutAdapter(ABC):
    """Abstract adapter turning positioned document lines/headers into structured lexicon drafts."""

    def __init__(self, profile: DomainProfile | None = None) -> None:
        self.profile = profile or default_generic_profile()

    @abstractmethod
    def split_columns(
        self,
        items: Sequence[dict[str, Any]],
        *,
        page_width: float = 1.0,
    ) -> list[list[dict[str, Any]]]:
        """Split a single page's positioned lines into ordered reading columns."""

    @abstractmethod
    def parse_header(self, header_text: str) -> dict[str, Any]:
        """Extract page number and start/end headword cues from a page header string."""

    @abstractmethod
    def build_draft_cards(
        self,
        entries: Sequence[dict[str, Any]],
        *,
        source_doc: str,
        extractor_id: str,
        doc_prefix: str = "DRAFT",
    ) -> list[dict[str, Any]]:
        """Convert parsed structural entries deterministically without LLM calls into draft dual-layer cards."""


_HEADER_PAGE_RE = re.compile(r"\bp\.?\s*0*(\d+)\b", re.IGNORECASE)
_HEADER_RANGE_RE = re.compile(r"\[([^\[\]\s/]+)\s*(?:\.{2,}|~|—|-|–|/)\s*([^\[\]\s/]+)\]")
_HEADER_CJK_SEPS_RE = re.compile(r"([^\s\[\]·/]+)\s*[·/|~—–-]{1,3}\s*([^\s\[\]·/]+)")


class DoubleColumnDictAdapter(LayoutAdapter):
    """Generic physical double-column dictionary adapter parameterized by DomainProfile."""

    def detect_gutter_x(
        self,
        items: Sequence[dict[str, Any]],
        *,
        page_width: float = 1.0,
    ) -> float | None:
        """Locate the central column-separating x coordinate if gap >= column_gutter_min_gap."""
        if not items or page_width <= 0:
            return None
        min_gap = self.profile.layout_adapter.column_gutter_min_gap
        norm_boxes: list[tuple[float, float]] = []
        for it in items:
            bbox = it.get("bbox")
            if not bbox or len(bbox) != 4:
                continue
            x0, _, x1, _ = (float(v) / page_width for v in bbox)
            # Ignore full-width titles or header banners spanning across the center.
            if x1 - x0 > 0.60:
                continue
            norm_boxes.append((max(0.0, min(1.0, x0)), max(0.0, min(1.0, x1))))

        if not norm_boxes:
            return None

        # Evaluate 100-bin occupancy across center band [0.35, 0.65].
        bins = 100
        occupied = [False] * bins
        for x0, x1 in norm_boxes:
            b0 = max(0, min(bins - 1, int(x0 * bins)))
            b1 = max(0, min(bins - 1, int(x1 * bins)))
            for b in range(b0, b1 + 1):
                occupied[b] = True

        best_start, best_len = -1, 0
        cur_start, cur_len = -1, 0
        for b in range(int(0.30 * bins), int(0.70 * bins)):
            if not occupied[b]:
                if cur_start == -1:
                    cur_start = b
                    cur_len = 1
                else:
                    cur_len += 1
                if cur_len > best_len:
                    best_start, best_len = cur_start, cur_len
            else:
                cur_start, cur_len = -1, 0

        gap_ratio = best_len / float(bins)
        if best_start == -1 or gap_ratio < min_gap:
            return None
        center_norm = (best_start + best_len / 2.0) / float(bins)
        return center_norm * page_width

    def split_columns(
        self,
        items: Sequence[dict[str, Any]],
        *,
        page_width: float = 1.0,
    ) -> list[list[dict[str, Any]]]:
        """Split page text boxes into left column then right column using gutter detection or midpoint."""
        gutter = self.detect_gutter_x(items, page_width=page_width)
        split_x = gutter if gutter is not None else (0.5 * page_width)
        left: list[dict[str, Any]] = []
        right: list[dict[str, Any]] = []
        for it in items:
            bbox = it.get("bbox")
            if not bbox or len(bbox) != 4:
                left.append(dict(it))
                continue
            x0, _, x1, _ = (float(v) for v in bbox)
            cx = 0.5 * (x0 + x1)
            if cx <= split_x:
                left.append(dict(it))
            else:
                right.append(dict(it))

        def _y_x_key(obj: dict[str, Any]) -> tuple[float, float]:
            bb = obj.get("bbox") or [0.0, 0.0, 0.0, 0.0]
            return (float(bb[1]), float(bb[0]))

        left.sort(key=_y_x_key)
        right.sort(key=_y_x_key)
        return [left, right]

    def parse_header(self, header_text: str) -> dict[str, Any]:
        """Extract page_num, start_headword, and end_headword from normalized header OCR/text."""
        txt = (header_text or "").strip()
        p_match = _HEADER_PAGE_RE.search(txt)
        page_num = int(p_match.group(1)) if p_match else None

        start_hw, end_hw = "", ""
        r_match = _HEADER_RANGE_RE.search(txt)
        if r_match:
            start_hw, end_hw = r_match.group(1).strip(), r_match.group(2).strip()
        else:
            cleaned = _HEADER_PAGE_RE.sub("", txt).strip()
            c_match = _HEADER_CJK_SEPS_RE.search(cleaned)
            if c_match:
                start_hw, end_hw = c_match.group(1).strip(), c_match.group(2).strip()

        return {
            "raw": txt,
            "page_num": page_num,
            "start_headword": start_hw,
            "end_headword": end_hw,
        }

    def check_onset_compatibility(self, section_key: str, headwords: Sequence[str]) -> list[str]:
        """Validate headwords against configured onset_compat_rules without dialect hardcoding."""
        rules = self.profile.layout_adapter.onset_compat_rules
        if not rules or section_key not in rules:
            return []
        allowed = set(rules[section_key])
        mismatches: list[str] = []
        for hw in headwords:
            lead = (hw or "").strip()[:1]
            if lead and allowed and lead not in allowed:
                mismatches.append(f"headword {hw!r} lead={lead!r} outside section {section_key!r} rules")
        return mismatches

    def audit_header_vs_index(
        self,
        header_ranges: Sequence[dict[str, Any]],
        index_entries: Sequence[dict[str, Any]],
    ) -> dict[str, Any]:
        """Zero-token physical closure check between page headers/body spans and book-end index."""
        by_page: dict[int, dict[str, Any]] = {}
        for hr in header_ranges:
            pno = hr.get("page_num")
            if isinstance(pno, int):
                by_page[pno] = dict(hr)

        discrepancies: list[dict[str, Any]] = []
        verified_count = 0
        section_hints = self.profile.layout_adapter.index_section_hints

        for idx_item in index_entries:
            hw = str(idx_item.get("headword") or "").strip()
            exp_page = idx_item.get("page")
            sec_hint = str(idx_item.get("section") or "").strip()
            if sec_hint and section_hints and sec_hint not in section_hints:
                discrepancies.append(
                    {
                        "headword": hw,
                        "page": exp_page,
                        "issue": f"unknown section hint {sec_hint!r}",
                        "crop_norm_box": self.profile.pdf_render.header_norm_box,
                    }
                )

            if not isinstance(exp_page, int) or exp_page not in by_page:
                discrepancies.append(
                    {
                        "headword": hw,
                        "page": exp_page,
                        "issue": "index page missing from header inventory",
                        "crop_norm_box": self.profile.pdf_render.preset_norm_boxes.get("full", [0.02, 0.02, 0.98, 0.98]),
                    }
                )
                continue

            page_info = by_page[exp_page]
            observed_hws = {str(x).strip() for x in (page_info.get("headwords") or []) if str(x).strip()}
            s_hw = str(page_info.get("start_headword") or "").strip()
            e_hw = str(page_info.get("end_headword") or "").strip()
            # Deterministic membership / boundary anchor match when explicit headwords are available.
            if observed_hws and hw not in observed_hws:
                col_hint = str(idx_item.get("column") or "full")
                crop_box = self.profile.pdf_render.preset_norm_boxes.get(
                    col_hint, self.profile.pdf_render.preset_norm_boxes.get("full", [0.02, 0.02, 0.98, 0.98])
                )
                discrepancies.append(
                    {
                        "headword": hw,
                        "page": exp_page,
                        "issue": f"headword absent on page {exp_page} (header=[{s_hw}..{e_hw}])",
                        "crop_norm_box": crop_box,
                    }
                )
            else:
                verified_count += 1

        return {
            "passed": len(discrepancies) == 0,
            "index_total": len(index_entries),
            "verified_count": verified_count,
            "discrepancy_count": len(discrepancies),
            "discrepancies": discrepancies,
        }

    def build_draft_cards(
        self,
        entries: Sequence[dict[str, Any]],
        *,
        source_doc: str,
        extractor_id: str,
        doc_prefix: str = "DRAFT",
    ) -> list[dict[str, Any]]:
        """Build structured dual-layer draft cards deterministically from parsed dictionary entries."""
        cards: list[dict[str, Any]] = []
        for idx, ent in enumerate(entries, start=1):
            forms = [str(f).strip() for f in (ent.get("forms") or []) if str(f).strip()]
            headword = str(ent.get("headword") or (forms[0] if forms else f"entry-{idx}")).strip()
            if not forms:
                forms = [headword]
            meaning = str(ent.get("meaning") or "").strip() or "(pending gloss)"
            quote = str(ent.get("quote") or ent.get("raw_text") or headword).strip()
            page = str(ent.get("page") or "?")
            card: dict[str, Any] = {
                "card_id": str(ent.get("card_id") or f"{doc_prefix}-{idx:04d}"),
                "judge_view": {
                    "card_type": str(ent.get("card_type") or "vocab"),
                    "topic": str(ent.get("topic") or headword),
                    "forms": forms,
                    "meaning": meaning,
                    "confidence": str(ent.get("confidence") or "medium"),
                },
                "provenance": {
                    "source_doc": source_doc,
                    "page": page,
                    "quote": quote,
                    "extracted_by": extractor_id,
                    "verified_by": str(ent.get("verified_by") or "PENDING_REVIEW"),
                },
            }
            for opt_j in ("usage", "standard_counterpart", "mandarin_counterpart", "misuse_or_confusable", "taxonomy_leaf", "examples", "related_cards", "tags"):
                if opt_j in ent and ent[opt_j] not in (None, "", []):
                    card["judge_view"][opt_j] = ent[opt_j]
            for opt_p in ("node_id", "section", "notes", "audit_status"):
                if opt_p in ent and ent[opt_p] not in (None, ""):
                    card["provenance"][opt_p] = ent[opt_p]
            cards.append(card)
        return cards
