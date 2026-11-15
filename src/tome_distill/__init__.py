"""tome_distill: Dual-layer lexicon knowledge-base authoring, compilation, and sparse retrieval."""

from .adapters import DoubleColumnDictAdapter, LayoutAdapter
from .compiler import (
    UnionFind,
    build_conflict_records,
    classify_misuse_card,
    compile_kb,
    extract_card_star_redlines,
    lint_regex_pattern,
    sync_bidirectional_related_cards,
    verify_compiled_gates,
)
from .field_diffs import (
    apply_field_diffs,
    deterministic_precheck,
    find_best_quote_window,
    next_inserted_card_id,
    parse_page_nums,
    stratified_blind_sample,
)
from .profile import DomainProfile, default_generic_profile, load_profile
from .retriever import SparseKBRetriever
from .schema import inspect_batch, merge_per_doc_cards, validate_card
from .text_utils import (
    TextNormalizer,
    clean_latex_fragments,
    find_bad_chars_in_obj,
    get_attested_whole_forms,
    has_bad_chars,
    is_whole_form_attested,
    scrub_bad_chars,
)
from .toc import build_doc_toc, calc_text_metrics, render_toc_markdown, verify_document_closure

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DomainProfile",
    "DoubleColumnDictAdapter",
    "LayoutAdapter",
    "SparseKBRetriever",
    "TextNormalizer",
    "UnionFind",
    "apply_field_diffs",
    "build_conflict_records",
    "build_doc_toc",
    "calc_text_metrics",
    "classify_misuse_card",
    "clean_latex_fragments",
    "compile_kb",
    "default_generic_profile",
    "deterministic_precheck",
    "extract_card_star_redlines",
    "find_bad_chars_in_obj",
    "find_best_quote_window",
    "get_attested_whole_forms",
    "has_bad_chars",
    "inspect_batch",
    "is_whole_form_attested",
    "lint_regex_pattern",
    "load_profile",
    "merge_per_doc_cards",
    "next_inserted_card_id",
    "parse_page_nums",
    "render_toc_markdown",
    "scrub_bad_chars",
    "stratified_blind_sample",
    "sync_bidirectional_related_cards",
    "validate_card",
    "verify_compiled_gates",
    "verify_document_closure",
]
