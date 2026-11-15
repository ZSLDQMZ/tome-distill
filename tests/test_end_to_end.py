"""Fast (<10s) self-contained end-to-end stdlib unittest suite for tome_distill."""

from __future__ import annotations

import ast
import importlib.util
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

_TESTS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _TESTS_DIR.parent
_SRC_DIR = _REPO_ROOT / "src"
_EXAMPLES_DIR = _REPO_ROOT / "examples"

for p in (_SRC_DIR, _EXAMPLES_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from tome_distill.adapters import DoubleColumnDictAdapter  # noqa: E402
from tome_distill.compiler import (  # noqa: E402
    compile_kb,
    lint_regex_pattern,
    sync_bidirectional_related_cards,
    verify_compiled_gates,
)
from tome_distill.field_diffs import (  # noqa: E402
    apply_field_diffs,
    next_inserted_card_id,
    parse_page_nums,
)
from tome_distill.pdf_render import norm_box_to_pdfium_crop  # noqa: E402
from tome_distill.profile import DomainProfile, TextNormalizationConfig, load_profile  # noqa: E402
from tome_distill.retriever import SparseKBRetriever  # noqa: E402
from tome_distill.schema import inspect_batch, validate_card  # noqa: E402
from tome_distill.text_utils import TextNormalizer, has_bad_chars  # noqa: E402
from tome_distill.toc import build_doc_toc, verify_document_closure  # noqa: E402
import run_toy_pipeline  # noqa: E402


class TestLexiconKBEndToEnd(unittest.TestCase):
    def test_01_ast_no_fitz_and_no_domain_leakage(self) -> None:
        forbidden_imports = {"fitz", "PyMuPDF"}
        structural_leak_patterns = {
            "non_loopback_ipv4": re.compile(r"\b(?!127\.0\.0\.1\b|0\.0\.0\.0\b)(?:\d{1,3}\.){3}\d{1,3}\b"),
            "win_user_path": re.compile(r"[A-Za-z]:[/\\]{1,2}[Uu]sers[/\\]"),
            "posix_home_path": re.compile(r"(?<![A-Za-z0-9_])/home/[A-Za-z0-9_.-]+"),
            "cluster_disk_prefix": re.compile(r"/[A-Z]\d{5}\b"),
        }
        py_files = sorted(
            p for p in _REPO_ROOT.rglob("*.py") if "__pycache__" not in p.parts
        )
        self.assertGreaterEqual(len(py_files), 14)

        for py_file in py_files:
            src_text = py_file.read_text(encoding="utf-8")
            tree = ast.parse(src_text, filename=str(py_file))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".", 1)[0]
                        self.assertNotIn(top, forbidden_imports, f"Forbidden import in {py_file}")
                elif isinstance(node, ast.ImportFrom):
                    mod = (node.module or "").split(".", 1)[0]
                    self.assertNotIn(mod, forbidden_imports, f"Forbidden import-from in {py_file}")

        text_exts = {".py", ".yaml", ".yml", ".toml", ".md"}
        for doc_file in sorted(_REPO_ROOT.rglob("*")):
            if not doc_file.is_file() or "__pycache__" in doc_file.parts:
                continue
            if doc_file.suffix.lower() not in text_exts:
                continue
            content = doc_file.read_text(encoding="utf-8")
            for pname, pat in structural_leak_patterns.items():
                m = pat.search(content)
                if m is not None:
                    self.fail(f"Structural leak pattern {pname!r} matched {m.group(0)!r} in {doc_file}")

        # Verify neutral generic defaults with zero domain hardcoding.
        for prof in (
            load_profile(None),
            load_profile(_REPO_ROOT / "profiles" / "generic_profile.yaml"),
        ):
            self.assertEqual(prof.profile_meta.dialect, "generic")
            self.assertEqual(prof.sources.canonical_docs, [])
            self.assertEqual(prof.text_normalization.protected_chars, [])
            self.assertEqual(prof.text_normalization.compat_char_aliases, {})
            self.assertEqual(prof.layout_adapter.onset_compat_rules, {})
            self.assertEqual(prof.gates.bare_noun_blacklist, [])

    def test_02_core_imports_without_optional_dependencies(self) -> None:
        code = (
            "import sys\n"
            "class BlockOptional:\n"
            "    BLOCKED = {'pypdfium2', 'PIL', 'numpy', 'cv2', 'rapidocr_onnxruntime', 'jieba'}\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname.split('.')[0] in self.BLOCKED:\n"
            "            raise ImportError(f'blocked optional dep: {fullname}')\n"
            "sys.meta_path.insert(0, BlockOptional())\n"
            "import tome_distill\n"
            "from tome_distill import compile_kb, SparseKBRetriever, validate_card, load_profile\n"
            "prof = load_profile(None)\n"
            "assert prof.profile_meta.dialect == 'generic'\n"
            "print('CORE_OK')\n"
        )
        env = dict(os.environ)
        env["PYTHONPATH"] = str(_SRC_DIR)
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            env=env,
            check=True,
        )
        self.assertIn("CORE_OK", res.stdout)

    def test_03_pdfium_crop_formula_and_jpeg_cap(self) -> None:
        if not importlib.util.find_spec("pypdfium2") or not importlib.util.find_spec("PIL"):
            self.skipTest("optional [pdf] extras not installed")

        from PIL import Image
        import numpy as np
        from tome_distill.pdf_render import (
            render_header_sheets_capped,
            render_page_jpeg_capped,
            render_pdf_page_image,
        )

        profile = load_profile(_REPO_ROOT / "profiles" / "generic_profile.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            pdf_info = run_toy_pipeline.build_fictional_double_column_pdf(tmp_dir / "toy.pdf")

            # Direct formula unit assertion.
            crop_tuple = norm_box_to_pdfium_crop(1000.0, 2000.0, [0.1, 0.2, 0.8, 0.5])
            for got, exp in zip(crop_tuple, (100.0, 1000.0, 200.0, 400.0)):
                self.assertAlmostEqual(got, exp, places=6)

            # Render header band via pypdfium2 -> must catch dark header block (low mean brightness).
            hdr_img = render_pdf_page_image(
                pdf_info["pdf_path"],
                1,
                dpi=120,
                norm_box=profile.pdf_render.header_norm_box,
                bleach_threshold=None,
            )
            hdr_mean = float(np.asarray(hdr_img).mean())
            self.assertLess(hdr_mean, 180.0)

            # Render blank center gutter band -> must be pure white.
            gutter_img = render_pdf_page_image(
                pdf_info["pdf_path"],
                1,
                dpi=120,
                norm_box=[0.44, 0.15, 0.56, 0.85],
                bleach_threshold=None,
            )
            self.assertGreater(float(np.asarray(gutter_img).mean()), 250.0)

            _, full_meta = render_page_jpeg_capped(
                pdf_info["pdf_path"],
                1,
                out_path=tmp_dir / "full.jpg",
                cfg=profile.pdf_render,
                preset="full",
            )
            _, crop_meta = render_page_jpeg_capped(
                pdf_info["pdf_path"],
                1,
                out_path=tmp_dir / "lt.jpg",
                cfg=profile.pdf_render,
                preset="lt",
            )
            sheets = render_header_sheets_capped(
                pdf_info["pdf_path"], [1], tmp_dir / "sheets", cfg=profile.pdf_render
            )
            self.assertLessEqual(full_meta["bytes"], profile.pdf_render.max_img_bytes)
            self.assertLessEqual(crop_meta["bytes"], profile.pdf_render.max_crop_bytes)
            self.assertLessEqual(sheets[0]["bytes"], profile.pdf_render.max_img_bytes)
            self.assertTrue((tmp_dir / "full.jpg").is_file())
            # Verify generated image is readable grayscale JPEG.
            with Image.open(tmp_dir / "full.jpg") as opened:
                self.assertEqual(opened.format, "JPEG")

            adapter = DoubleColumnDictAdapter(profile)
            cols = adapter.split_columns(pdf_info["positioned_items"], page_width=pdf_info["width"])
            self.assertEqual([len(cols[0]), len(cols[1])], [3, 3])
            parsed_hdr = adapter.parse_header(pdf_info["header_text"])
            self.assertEqual(parsed_hdr["page_num"], 1)
            self.assertEqual(parsed_hdr["start_headword"], "星言")
            self.assertEqual(parsed_hdr["end_headword"], "青鸟")
            parsed_hdr["headwords"] = [x["headword"] for x in pdf_info["positioned_items"]]
            audit = adapter.audit_header_vs_index([parsed_hdr], pdf_info["index_entries"])
            self.assertTrue(audit["passed"])

    def test_04_control_char_pre_json_meta_leak_and_non_leading_star(self) -> None:
        profile = load_profile(_REPO_ROOT / "profiles" / "generic_profile.yaml")
        bad_card = {
            "card_id": "BAD-001",
            "judge_view": {
                "card_type": "vocab",
                "topic": "控制符与泄露测试",
                "forms": ["坏词*错"],
                "meaning": "含控制符\x07以及元话语原书未单列见p.12",
                "confidence": "high",
                "examples": [
                    {"dialect": "前缀之后*中置星号错句", "standard": "坏句"},
                ],
            },
            "provenance": {
                "source_doc": "toy_doc",
                "page": "1",
                "quote": "原始摘录",
                "extracted_by": "same_agent",
                "verified_by": "same_agent",
            },
        }
        self.assertTrue(any("0x07" in h for h in has_bad_chars("含控制符\x07")))
        errs = validate_card(bad_card, profile=profile)
        joined = "\n".join(errs)
        self.assertIn("bad_char", joined)
        self.assertIn("0x07", joined)
        self.assertIn("dual-blind violation", joined)
        self.assertIn("meta_leak", joined)
        self.assertIn("unannotated '*' marker in judge_view.forms", joined)
        self.assertIn("unmarked '*' in judge_view.examples[0].dialect", joined)

        batch_rep = inspect_batch([bad_card], profile=profile)
        self.assertFalse(batch_rep["passed"])

    def test_05_wide_page_range_and_over_25_inserts(self) -> None:
        pages = parse_page_nums("1-12, 20-25", max_page_span=50)
        self.assertEqual(pages, list(range(1, 13)) + list(range(20, 26)))
        with self.assertRaises(ValueError):
            parse_page_nums("1-99", max_page_span=20)

        anchor = [
            {
                "card_id": "TOY-001",
                "judge_view": {
                    "card_type": "vocab",
                    "topic": "锚点",
                    "forms": ["锚词"],
                    "meaning": "锚点词",
                    "confidence": "high",
                },
                "provenance": {
                    "source_doc": "toy",
                    "page": "1",
                    "quote": "锚词",
                    "extracted_by": "a",
                    "verified_by": "b",
                },
            }
        ]
        # Build 30 inserts after the same anchor (>25 single-letter B..Z range).
        inserts = [
            {
                "after_card_id": "TOY-001",
                "card": {
                    "card_id": "AUTO",
                    "judge_view": {
                        "card_type": "vocab",
                        "topic": f"插卡{i}",
                        "forms": [f"插词{i}"],
                        "meaning": "补充条目",
                        "confidence": "high",
                    },
                    "provenance": {
                        "source_doc": "toy",
                        "page": "1",
                        "quote": "插词",
                        "extracted_by": "a",
                    },
                },
            }
            for i in range(30)
        ]
        updated, summary = apply_field_diffs(anchor, {"insert_after": inserts}, reviewer_id="b")
        self.assertEqual(summary["inserted_count"], 30)
        ids = summary["inserted_ids"]
        self.assertEqual(ids[0], "TOY-001B")
        self.assertEqual(ids[24], "TOY-001Z")
        self.assertEqual(ids[25], "TOY-001AA")
        self.assertEqual(ids[29], "TOY-001AE")
        self.assertEqual(len(updated), 31)
        self.assertEqual(next_inserted_card_id("TOY-001", set(ids) | {"TOY-001"}), "TOY-001AF")

    def test_06_protected_opencc_1to1_span_invariant(self) -> None:
        norm = TextNormalizer(TextNormalizationConfig(opencc_config="t2s", protected_chars=["龍"]))
        src = "傳統龍燈詞"
        out = norm.to_simp_keep_protected(src, preserve_length=True)
        self.assertEqual(out, "传统龍灯词")
        self.assertEqual(len(out), len(src))

    def test_07_toc_closure_modes_and_override_metrics_warning(self) -> None:
        text = "\n".join([f"line {i}" for i in range(1, 7)])
        strict_toc = build_doc_toc(
            "d1",
            [
                {"node_id": "n1", "title": "A", "start_line": 1, "end_line": 3, "extract": "yes"},
                {"node_id": "n2", "title": "B", "start_line": 4, "end_line": 6, "extract": "yes"},
            ],
            text,
        )
        self.assertTrue(verify_document_closure(strict_toc, text, mode="strict_line_partition")["passed"])

        # Shared line span fails strict_line_partition, succeeds multicolumn_line_union with explicit audit warning.
        multi_toc = build_doc_toc(
            "d2",
            [
                {
                    "node_id": "c1_1",
                    "column_id": "left",
                    "title": "Col1",
                    "start_line": 1,
                    "end_line": 6,
                    "override_metrics": {"char_count": 15},
                },
                {
                    "node_id": "c2_1",
                    "column_id": "right",
                    "title": "Col2",
                    "start_line": 1,
                    "end_line": 6,
                    "override_metrics": {"char_count": 15},
                },
            ],
            text,
        )
        self.assertFalse(verify_document_closure(multi_toc, text, mode="strict_line_partition")["passed"])
        multi_rep = verify_document_closure(multi_toc, text, mode="multicolumn_line_union")
        self.assertTrue(multi_rep["passed"])
        self.assertTrue(any("override_metrics" in w for w in multi_rep["warnings"]))

    def test_08_full_10_card_pipeline_compile_gates_and_retrieval(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_toy_pipeline.run_toy_pipeline(Path(tmp))

        self.assertTrue(summary["pdf_render"]["header_audit_passed"])
        self.assertEqual(summary["pdf_render"]["col_counts"], [3, 3])
        self.assertTrue(summary["toc_closure_passed"])
        self.assertTrue(summary["batch_gate_passed"])
        self.assertTrue(summary["eight_gate_passed"])
        self.assertEqual(summary["diff_summary"]["output_count"], 10)
        self.assertEqual(sum(summary["compile_table_counts"].values()), 9)  # TOY-003 + TOY-004 cluster into 1
        self.assertIn("花廊", summary["compounds_generated"])
        # Protected char '龍' stays un-simplified while surrounding '燈' converts to '灯'.
        self.assertNotIn("龍燈", summary["compounds_generated"])

        # Verify regex linter blocks unbounded wildcards and ambiguity logging works cleanly.
        self.assertTrue(any("wildcard" in e for e in lint_regex_pattern("云着.*吗")))
        self.assertEqual(lint_regex_pattern("云着[了吗]"), [])

        # Verify ambiguity callback on duplicate headword cross-ref.
        cards, _, _, _ = run_toy_pipeline.make_fictional_toy_cards()
        amb_cards = [
            cards[1],  # 星侬 -> ⇨〖月桥〗
            cards[2],  # 月桥 (source A)
            cards[3],  # 月桥 (source B)
        ]
        seen_amb: list[tuple[str, str, list[str]]] = []
        _, srep = sync_bidirectional_related_cards(
            amb_cards,
            ambiguity_strategy="log_and_skip",
            on_ambiguous=lambda cid, tgt, cands: seen_amb.append((cid, tgt, cands)),
        )
        self.assertEqual(len(seen_amb), 1)
        self.assertEqual(len(srep["ambiguities"]), 1)

        # Verify compiled release gates + budgeted priority + compound masking directly.
        profile = DomainProfile(
            profile_meta=load_profile(None).profile_meta,
            text_normalization=TextNormalizationConfig(protected_chars=["龍"]),
        )
        raw_cards, diff_patch, conflict_specs, regression_guards = run_toy_pipeline.make_fictional_toy_cards()
        patched, _ = apply_field_diffs(raw_cards, diff_patch, reviewer_id="reviewer_b")
        bundle = compile_kb(patched, profile=profile, conflict_specs=conflict_specs, regression_guards=regression_guards)
        gates = verify_compiled_gates(bundle, profile=profile)
        self.assertTrue(gates["passed"])
        self.assertTrue(gates["gates"]["GATE-8_blind_audit"]["small_corpus_scaled"])
        # Protected char index key uses '龍灯' (龍 protected, 燈->灯 1:1).
        self.assertIn("龍灯", bundle["compiled_lookup_index"]["dialect_t2s_to_entries"])

        retriever = SparseKBRetriever(bundle, profile=profile, external_compounds={"花廊", "云烟"})
        res_r3 = retriever.retrieve("看云烟和叠叠着也买云着", mode="dialect_only", strategy="budgeted", max_cards=2, max_chars=280)
        selected_forms = [f for h in res_r3["selected_entries"] for f in h["meta"]["forms"]]
        self.assertIn("云着", selected_forms)
        self.assertIn("叠叠", selected_forms)

        res_anchor = retriever.retrieve("去静池赏夜桥", mode="standard_anchor", strategy="full")
        anchor_forms = [f for h in res_anchor["selected_entries"] for f in h["meta"]["forms"]]
        self.assertIn("月桥", anchor_forms)
        self.assertIn("水镜", anchor_forms)


if __name__ == "__main__":
    unittest.main()
