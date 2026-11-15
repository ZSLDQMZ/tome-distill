"""End-to-end fictional toy pipeline demonstrating tome_distill from synthetic PDF to budgeted retrieval."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

# Allow running directly as `python examples/run_toy_pipeline.py` without editable install.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from tome_distill.adapters import DoubleColumnDictAdapter  # noqa: E402
from tome_distill.compiler import (  # noqa: E402
    compile_kb,
    sync_bidirectional_related_cards,
    verify_compiled_gates,
)
from tome_distill.field_diffs import apply_field_diffs, deterministic_precheck  # noqa: E402
from tome_distill.pdf_render import (  # noqa: E402
    render_header_sheets_capped,
    render_page_jpeg_capped,
)
from tome_distill.profile import DomainProfile, TextNormalizationConfig, load_profile  # noqa: E402
from tome_distill.retriever import SparseKBRetriever  # noqa: E402
from tome_distill.schema import inspect_batch, merge_per_doc_cards  # noqa: E402
from tome_distill.toc import build_doc_toc, render_toc_markdown, verify_document_closure  # noqa: E402
from tome_distill.zh_compounds import generate_common_compounds  # noqa: E402


def build_fictional_double_column_pdf(out_pdf: Path) -> dict[str, Any]:
    """Generate a 1-page completely fictional double-column Starlight Toy Lexicon PDF via Pillow."""
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1240, 1754
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    # Header bar placed squarely inside default header_norm_box [0.035, 0.030, 0.965, 0.102].
    draw.rectangle([48, 56, width - 48, 170], fill=(22, 22, 22), outline=(0, 0, 0), width=3)
    draw.text((80, 95), "STARLIGHT TOY LEXICON   TOY p.01 [Xingyan ... Qingniao]", fill=(255, 255, 255), font=font)

    # Fictional left-column items (x = 7%..41%) and right-column items (x = 59%..93%), wide blank gutter!
    left_entries = [
        {"bbox": [90, 260, 500, 340], "text": "【星言】虚构玩具词头，表示星光记号。", "headword": "星言", "page": 1},
        {"bbox": [90, 410, 500, 490], "text": "【云着】句末助词，表示虚构动作已然完成。", "headword": "云着", "page": 1},
        {"bbox": [90, 560, 500, 640], "text": "【月桥】夜色廊桥，参见 ⇨〖星侬〗。", "headword": "月桥", "page": 1},
    ]
    right_entries = [
        {"bbox": [740, 260, 1150, 340], "text": "【花廊】普通名物复合词：花卉长廊。", "headword": "花廊", "page": 1},
        {"bbox": [740, 410, 1150, 490], "text": "【露珠】清晨凝露（通用词汇）。", "headword": "露珠", "page": 1},
        {"bbox": [740, 560, 1150, 640], "text": "【青鸟】传说信使，亦作象征词。", "headword": "青鸟", "page": 1},
    ]

    for item in left_entries + right_entries:
        x0, y0, x1, y1 = item["bbox"]
        draw.rectangle([x0, y0, x1, y1], outline=(40, 40, 40), width=2)
        draw.text((x0 + 12, y0 + 28), item["headword"], fill=(0, 0, 0), font=font)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(out_pdf), "PDF", resolution=150.0)
    return {
        "pdf_path": str(out_pdf),
        "width": width,
        "height": height,
        "header_text": "TOY p.01 [星言 ... 青鸟]",
        "positioned_items": left_entries + right_entries,
        "index_entries": [
            {"headword": "星言", "page": 1},
            {"headword": "云着", "page": 1},
            {"headword": "月桥", "page": 1},
            {"headword": "花廊", "page": 1},
            {"headword": "露珠", "page": 1},
            {"headword": "青鸟", "page": 1},
        ],
    }


def make_fictional_toy_cards() -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    """Create 9 base fictional cards + 1 field_diffs patch reaching 10 clean cards."""
    base_cards: list[dict[str, Any]] = [
        {
            "card_id": "TOY-001",
            "judge_view": {
                "card_type": "grammar",
                "topic": "虚构体标记云着",
                "forms": ["云着"],
                "meaning": "虚构句子完成体助词，用于谓语后表示事件落定",
                "usage": "紧贴动词后；不可说*云着未，否则句式冲突",
                "taxonomy_leaf": "G3.1",
                "confidence": "high",
                "tags": ["regex:云着[了吗]"],
                "examples": [
                    {"dialect": "他买云着", "standard": "他买好了"},
                    {"dialect": "*他云着未买", "standard": "他还没买", "is_negative": True, "note": "否定病句红线"},
                ],
            },
            "provenance": {
                "source_doc": "toy_book_a",
                "node_id": "L1",
                "page": "1",
                "quote": "云着：虚构体助词，表动作已定，忌与否定词未同现。",
                "extracted_by": "extractor_a",
                "verified_by": "reviewer_b",
            },
        },
        {
            "card_id": "TOY-002",
            "judge_view": {
                "card_type": "grammar",
                "topic": "虚构人称星侬",
                "forms": ["星侬"],
                "meaning": "虚构第一人称复数代词，对应咱们；详见 ⇨〖月桥〗",
                "usage": "作主语或领属定语，常与体标记并见",
                "standard_counterpart": "咱们",
                "taxonomy_leaf": "G2.1",
                "confidence": "high",
                "examples": [{"dialect": "星侬去花廊", "standard": "咱们去花廊"}],
            },
            "provenance": {
                "source_doc": "toy_book_a",
                "node_id": "L1",
                "page": "1",
                "quote": "星侬：第一人称复数代词，相当于咱们，参见 ⇨〖月桥〗。",
                "extracted_by": "extractor_a",
                "verified_by": "reviewer_b",
            },
        },
        {
            "card_id": "TOY-003",
            "judge_view": {
                "card_type": "vocab",
                "topic": "月桥（源A）",
                "forms": ["月桥"],
                "meaning": "夜间观赏星月的虚构廊桥",
                "standard_counterpart": "夜桥",
                "confidence": "high",
            },
            "provenance": {
                "source_doc": "toy_book_a",
                "node_id": "L2",
                "page": "2",
                "quote": "月桥：夜间赏月之虚构廊桥，亦称夜桥。",
                "extracted_by": "extractor_a",
                "verified_by": "reviewer_b",
            },
        },
        {
            "card_id": "TOY-004",
            "judge_view": {
                "card_type": "vocab",
                "topic": "月桥（源B）",
                "forms": ["月桥"],
                "meaning": "跨越星河支流的拱形石桥；可参 ⇨〖星侬〗",
                "standard_counterpart": "夜桥",
                "confidence": "medium",
            },
            "provenance": {
                "source_doc": "toy_book_b",
                "node_id": "L3",
                "page": "3",
                "quote": "月桥：跨星河支流拱桥，旧俗夜游多聚此桥。",
                "extracted_by": "extractor_c",
                "verified_by": "reviewer_b",
            },
        },
        {
            "card_id": "TOY-005",
            "judge_view": {
                "card_type": "idiom",
                "topic": "风铃响水",
                "forms": ["风铃响水"],
                "meaning": "比喻微小的信号在水面引起连续回响",
                "standard_counterpart": "一石激浪",
                "confidence": "high",
            },
            "provenance": {
                "source_doc": "toy_book_b",
                "node_id": "L3",
                "page": "4",
                "quote": "风铃响水：喻微讯触发连环回响，犹言一石激浪。",
                "extracted_by": "extractor_c",
                "verified_by": "reviewer_b",
            },
        },
        {
            "card_id": "TOY-006",
            "judge_view": {
                "card_type": "orthography",
                "topic": "受保护古字龍燈",
                "forms": ["龍燈"],
                "meaning": "节庆巡游所执星彩灯架，首字受本字保护不改写",
                "standard_counterpart": "灯彩",
                "confidence": "high",
            },
            "provenance": {
                "source_doc": "toy_book_a",
                "node_id": "L2",
                "page": "5",
                "quote": "龍燈：节庆巡游彩灯，本字当作龍，不宜简并混同他词。",
                "extracted_by": "extractor_a",
                "verified_by": "reviewer_b",
            },
        },
        {
            "card_id": "TOY-007",
            "judge_view": {
                "card_type": "vocab",
                "topic": "花廊",
                "forms": ["花廊"],
                "meaning": "初步释义待修订",
                "standard_counterpart": "花棚",
                "confidence": "medium",
            },
            "provenance": {
                "source_doc": "toy_book_a",
                "node_id": "L2",
                "page": "6",
                "quote": "花廊：庭院搭架植蔓之处，作花棚解。",
                "extracted_by": "extractor_a",
                "verified_by": "extractor_a",
            },
        },
        {
            "card_id": "TOY-008",
            "judge_view": {
                "card_type": "grammar",
                "topic": "虚构叠词构式叠叠",
                "forms": ["叠叠"],
                "meaning": "表示轻微反复动作；重叠后禁止再附着着字",
                "usage": "单用即可；严禁叠加完成体",
                "misuse_or_confusable": "不可说*叠叠着",
                "taxonomy_leaf": "G1.1",
                "confidence": "high",
                "examples": [{"dialect": "*看叠叠着", "standard": "看了一会儿", "is_negative": True}],
            },
            "provenance": {
                "source_doc": "toy_book_b",
                "node_id": "L4",
                "page": "7",
                "quote": "叠叠：表轻量反复，叠用后严禁再带着字。",
                "extracted_by": "extractor_c",
                "verified_by": "reviewer_b",
            },
        },
        {
            "card_id": "TOY-009",
            "judge_view": {
                "card_type": "vocab",
                "topic": "露珠",
                "forms": ["露珠"],
                "meaning": "晨间叶面凝结的细小水珠",
                "standard_counterpart": "晨露",
                "confidence": "high",
            },
            "provenance": {
                "source_doc": "toy_book_b",
                "node_id": "L4",
                "page": "8",
                "quote": "露珠：黎明草木凝露。",
                "extracted_by": "extractor_c",
                "verified_by": "reviewer_b",
            },
        },
        {
            "card_id": "TOY-OBSOLETE",
            "judge_view": {
                "card_type": "vocab",
                "topic": "待删错录",
                "forms": ["错录"],
                "meaning": "错录条目待删除",
                "confidence": "low",
            },
            "provenance": {
                "source_doc": "toy_book_b",
                "node_id": "L4",
                "page": "8",
                "quote": "错录占位。",
                "extracted_by": "extractor_c",
                "verified_by": "reviewer_b",
            },
        },
    ]

    diff_patch = {
        "updates": [
            {
                "card_id": "TOY-007",
                "judge_view": {
                    "meaning": "庭院植蔓避荫的长廊，亦对应花棚",
                    "confidence": "high",
                },
            }
        ],
        "insert_after": [
            {
                "after_card_id": "TOY-009",
                "card": {
                    "card_id": "TOY-010",
                    "judge_view": {
                        "card_type": "vocab",
                        "topic": "水镜",
                        "forms": ["水镜"],
                        "meaning": "平静倒映星空的虚构池面",
                        "standard_counterpart": "静池",
                        "confidence": "high",
                    },
                    "provenance": {
                        "source_doc": "toy_book_b",
                        "node_id": "L4",
                        "page": "9",
                        "quote": "水镜：静水映星之池。",
                        "extracted_by": "extractor_c",
                        "verified_by": "reviewer_b",
                    },
                },
            }
        ],
        "delete_ids": ["TOY-OBSOLETE"],
        "visual_audit_notes": ["p.01 double-column gutter verified"],
    }

    conflict_specs = [
        {
            "conflict_id": "CONF-TOY-1",
            "topic": "月桥跨河与观赏义分歧",
            "variants": ["月桥"],
            "card_ids": ["TOY-003", "TOY-004"],
            "resolution_policy": "preserve_all_in_per_doc_kb_no_override",
        }
    ]
    regression_guards = [
        {"rule_id": "REG-TOY-1", "forms": ["云着"], "note": "aspect marker guard"},
    ]
    return base_cards, diff_patch, conflict_specs, regression_guards


def run_toy_pipeline(work_dir: Path) -> dict[str, Any]:
    """Execute the complete end-to-end fictional tome_distill workflow inside work_dir."""
    work_dir.mkdir(parents=True, exist_ok=True)
    profile_path = _REPO_ROOT / "profiles" / "generic_profile.yaml"
    profile: DomainProfile = load_profile(profile_path)
    # Add one fictional protected char '龍' solely for toy OpenCC span demonstration.
    profile.text_normalization = TextNormalizationConfig(
        opencc_config=profile.text_normalization.opencc_config,
        protected_chars=["龍"],
        compat_char_aliases=profile.text_normalization.compat_char_aliases,
        ocr_suspect_chars=profile.text_normalization.ocr_suspect_chars,
        inline_note_cues=profile.text_normalization.inline_note_cues,
        meta_leak_patterns=profile.text_normalization.meta_leak_patterns,
        crossref_arrow_pattern=profile.text_normalization.crossref_arrow_pattern,
        circled_chars=profile.text_normalization.circled_chars,
        etymon_note_pattern=profile.text_normalization.etymon_note_pattern,
        false_friend_pattern=profile.text_normalization.false_friend_pattern,
        hard_ban_pattern=profile.text_normalization.hard_ban_pattern,
    )

    # 1. Generate 1-page fictional double-column PDF + pypdfium2 capped JPEGs + column/header audit.
    pdf_info = build_fictional_double_column_pdf(work_dir / "toy_dict.pdf")
    render_dir = work_dir / "renders"
    _, full_meta = render_page_jpeg_capped(
        pdf_info["pdf_path"], 1, out_path=render_dir / "toy_p0001_full.jpg", cfg=profile.pdf_render, preset="full"
    )
    _, crop_meta = render_page_jpeg_capped(
        pdf_info["pdf_path"], 1, out_path=render_dir / "toy_p0001_lt.jpg", cfg=profile.pdf_render, preset="lt"
    )
    header_sheets = render_header_sheets_capped(
        pdf_info["pdf_path"], [1], render_dir / "headers", cfg=profile.pdf_render
    )

    adapter = DoubleColumnDictAdapter(profile)
    cols = adapter.split_columns(pdf_info["positioned_items"], page_width=pdf_info["width"])
    parsed_header = adapter.parse_header(pdf_info["header_text"])
    parsed_header["headwords"] = [it["headword"] for it in pdf_info["positioned_items"]]
    header_audit = adapter.audit_header_vs_index([parsed_header], pdf_info["index_entries"])
    draft_cards = adapter.build_draft_cards(
        [{"headword": it["headword"], "page": 1, "quote": it["text"]} for it in cols[0] + cols[1]],
        source_doc="toy_pdf_demo",
        extractor_id="layout_adapter",
        doc_prefix="PDF",
    )

    # 2. TOC construction & strict line partition closure.
    doc_text = "\n".join(
        [
            "-----",
            "第一节 语法与人称",
            "云着：完成体助词",
            "星侬：第一人称复数",
            "-----",
            "第二节 词汇与正字",
            "月桥：夜桥或拱桥",
            "龍燈：节庆彩灯",
            "花廊：花卉长廊",
            "-----",
            "第三节 熟语与补充",
            "风铃响水：微讯回响",
            "叠叠：轻微反复",
            "露珠：晨露",
            "水镜：静池",
        ]
    )
    toc_doc = build_doc_toc(
        "toy_corpus",
        [
            {"node_id": "L1", "title": "语法与人称", "start_line": 1, "end_line": 4, "extract": "yes"},
            {"node_id": "L2", "title": "词汇与正字", "start_line": 5, "end_line": 9, "extract": "yes"},
            {"node_id": "L3", "title": "跨源复核词", "start_line": 10, "end_line": 12, "extract": "yes"},
            {"node_id": "L4", "title": "熟语与补充", "start_line": 13, "end_line": 15, "extract": "yes"},
        ],
        doc_text,
    )
    toc_closure = verify_document_closure(toc_doc, doc_text, mode="strict_line_partition")
    (work_dir / "toc.md").write_text(render_toc_markdown(toc_doc), encoding="utf-8")

    # 3. Apply field_diffs patch (fixing TOY-007 self-review & inserting TOY-010, deleting TOY-OBSOLETE).
    raw_cards, diff_patch, conflict_specs, regression_guards = make_fictional_toy_cards()
    patched_cards, diff_summary = apply_field_diffs(raw_cards, diff_patch, reviewer_id="reviewer_b")

    coverage = {
        "node_to_cards": {
            "L1": ["TOY-001", "TOY-002"],
            "L2": ["TOY-003", "TOY-006", "TOY-007"],
            "L3": ["TOY-004", "TOY-005"],
            "L4": ["TOY-008", "TOY-009", "TOY-010"],
        },
        "empty_nodes": [],
    }
    batch_report = inspect_batch(
        patched_cards,
        profile=profile,
        coverage=coverage,
        expected_node_ids=["L1", "L2", "L3", "L4"],
    )
    merged = merge_per_doc_cards(
        [patched_cards],
        node_order=["L1", "L2", "L3", "L4"],
        coverage=coverage,
        profile=profile,
        out_jsonl=work_dir / "merged_cards.jsonl",
    )
    qa_precheck = deterministic_precheck(merged["cards"], source_texts={"toy_book_a": doc_text}, profile=profile)

    # 4. External compound mask generation -> KB compile -> release-gate verification.
    compounds_path = work_dir / "compounds.json"
    compounds_dict = generate_common_compounds(
        [("花廊", 650), ("露珠", 520), ("水镜", 410), ("龍燈", 500)],
        min_freq=200,
        protect_chars=profile.text_normalization.protected_chars,
        out_path=compounds_path,
    )
    synced_cards, sync_rep = sync_bidirectional_related_cards(
        merged["cards"], profile=profile
    )
    bundle = compile_kb(
        synced_cards,
        profile=profile,
        conflict_specs=conflict_specs,
        regression_guards=regression_guards,
        out_dir=work_dir / "compiled",
    )
    gate_report = verify_compiled_gates(bundle, profile=profile)

    # 5. Sparse retrieval showcase (budgeted & hybrid).
    retriever = SparseKBRetriever(
        work_dir / "compiled",
        profile=profile,
        external_compounds=compounds_path,
    )
    r3_res = retriever.retrieve(
        "星侬今日在花廊看叠叠着又买云着",
        mode="dialect_only",
        strategy="budgeted",
        max_cards=3,
        max_chars=260,
    )
    hybrid_res = retriever.retrieve(
        "星侬步月桥赏星",
        mode="hybrid",
        draft_standard="我们走过夜桥看花棚",
        strategy="budgeted",
    )

    return {
        "pdf_render": {
            "full_bytes": full_meta["bytes"],
            "crop_bytes": crop_meta["bytes"],
            "header_sheet_bytes": header_sheets[0]["bytes"],
            "col_counts": [len(cols[0]), len(cols[1])],
            "header_audit_passed": header_audit["passed"],
            "draft_card_count": len(draft_cards),
        },
        "toc_closure_passed": toc_closure["passed"],
        "diff_summary": diff_summary,
        "batch_gate_passed": batch_report["passed"],
        "qa_flagged_count": qa_precheck["flagged_count"],
        "compounds_generated": sorted(compounds_dict.keys()),
        "related_sync_added_backlinks": sync_rep["added_backlinks"],
        "compile_table_counts": bundle["manifest"]["table_counts"],
        "eight_gate_passed": gate_report["passed"],
        "r3_selected_ids": [h["entry_id"] for h in r3_res["selected_entries"]],
        "hybrid_selected_ids": [h["entry_id"] for h in hybrid_res["selected_entries"]],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run end-to-end fictional tome_distill toy pipeline.")
    parser.add_argument("--work-dir", type=Path, default=None, help="Output directory for pipeline artifacts.")
    args = parser.parse_args(argv)

    if args.work_dir is not None:
        summary = run_toy_pipeline(args.work_dir)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if (summary["eight_gate_passed"] and summary["batch_gate_passed"] and summary["toc_closure_passed"]) else 1

    with tempfile.TemporaryDirectory() as tmp:
        summary = run_toy_pipeline(Path(tmp))
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0 if (summary["eight_gate_passed"] and summary["batch_gate_passed"] and summary["toc_closure_passed"]) else 1


if __name__ == "__main__":
    raise SystemExit(main())
