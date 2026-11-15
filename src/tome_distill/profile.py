"""DomainProfile configuration dataclasses and YAML loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


def _default_meta_leak_patterns() -> list[dict[str, Any]]:
    return [
        {"name": "page_ref_latin", "pattern": r"\bp\.\s*\d+", "ignore_case": True},
        {"name": "page_ref_cjk", "pattern": r"第\s*\d+\s*页"},
        {"name": "meta_this_card", "pattern": r"本卡"},
        {"name": "meta_this_section", "pattern": r"本节"},
        {"name": "meta_not_listed_separately", "pattern": r"原书未单列"},
        {"name": "meta_original_book", "pattern": r"原书"},
        {"name": "meta_unicode_codepoint", "pattern": r"U\+[0-9A-Fa-f]{4,6}"},
    ]


def _default_preset_boxes() -> dict[str, tuple[float, float, float, float]]:
    return {
        "full": (0.02, 0.02, 0.98, 0.98),
        "left": (0.01, 0.02, 0.53, 0.98),
        "right": (0.47, 0.02, 0.99, 0.98),
        "top": (0.02, 0.02, 0.98, 0.54),
        "bot": (0.02, 0.46, 0.98, 0.98),
        "lt": (0.01, 0.02, 0.53, 0.55),
        "lb": (0.01, 0.45, 0.53, 0.98),
        "rt": (0.47, 0.02, 0.99, 0.55),
        "rb": (0.47, 0.45, 0.99, 0.98),
    }


def _default_taxonomy() -> dict[str, Any]:
    return {
        "G1": {
            "name": "Morphology & Affixes",
            "children": {
                "G1.1": {"name": "Reduplication & Derivation", "scope": "Morphology"},
            },
        },
        "G2": {
            "name": "Nominals & Reference",
            "children": {
                "G2.1": {"name": "Pronouns & Demonstratives", "scope": "Nominals"},
            },
        },
        "G3": {
            "name": "Predicate, Aspect & Modality",
            "children": {
                "G3.1": {"name": "Aspect & Sentence-Final Particles", "scope": "Syntax"},
            },
        },
    }


@dataclass
class ProfileMetaConfig:
    dialect: str = "generic"
    schema_version: str = "1.0"
    target_tables: tuple[str, ...] = (
        "grammar_and_function_words",
        "lexicon_and_orthography",
        "idioms_and_proverbs",
    )
    taxonomy_tree: dict[str, Any] = field(default_factory=_default_taxonomy)

    def valid_leaf_ids(self) -> set[str]:
        leaves: set[str] = set()
        for branch in self.taxonomy_tree.values():
            children = branch.get("children", {}) if isinstance(branch, dict) else {}
            leaves.update(children.keys())
        return leaves


@dataclass
class SourcesConfig:
    canonical_docs: list[str] = field(default_factory=list)
    pdf_glob_patterns: dict[str, str] = field(default_factory=dict)
    reprocessed_files: dict[str, str] = field(default_factory=dict)
    isolated_eval_paths: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TextNormalizationConfig:
    opencc_config: str = "t2s"
    protected_chars: list[str] = field(default_factory=list)
    compat_char_aliases: dict[str, str] = field(default_factory=dict)
    ocr_suspect_chars: list[str] = field(default_factory=list)
    inline_note_cues: list[dict[str, str]] = field(
        default_factory=lambda: [
            {"name": "inline_gloss_parens", "pattern": r"[（(〔\[].{1,25}[）)〕\]]"},
            {"name": "inline_gloss_metaphor", "pattern": r"比喻"},
        ]
    )
    meta_leak_patterns: list[dict[str, Any]] = field(default_factory=_default_meta_leak_patterns)
    crossref_arrow_pattern: str = r"⇨\s*〖([^〗]+)〗"
    circled_chars: str = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"
    etymon_note_pattern: str = (
        r"(?:本字(?:当|应|宜|实)?(?:作|为|是)|正字(?:当|应)?作|俗(?:常)?(?:作|写|误作)|借字|训读字|同音借用)"
    )
    false_friend_pattern: str = r"(?:≠|不等于|望文生义|不同于|字面.*误解)"
    hard_ban_pattern: str = r"(?:不可说|不能说|严禁|切忌直译|忌用|不合语法|病句|误说|\*)"


@dataclass
class PdfRenderConfig:
    dpi: int = 150
    header_dpi: int = 160
    crop_dpi: int = 165
    max_img_bytes: int = 215_000
    max_crop_bytes: int = 145_000
    bleach_threshold: int = 224
    header_norm_box: tuple[float, float, float, float] = (0.035, 0.030, 0.965, 0.102)
    preset_norm_boxes: dict[str, tuple[float, float, float, float]] = field(default_factory=_default_preset_boxes)
    strip_width: int = 1040
    tag_width: int = 108
    max_strips_per_sheet: int = 15
    scale_steps: tuple[float, ...] = (1.0, 0.90, 0.80, 0.72, 0.64, 0.56)
    jpeg_qualities: tuple[int, ...] = (78, 70, 62, 56, 48, 42)
    min_dim: int = 200
    font_paths: list[str] = field(default_factory=list)


@dataclass
class LayoutAdapterConfig:
    adapter_type: str = "double_column_dict"
    page_delimiter: str = "-----"
    column_gutter_min_gap: float = 0.05
    index_section_hints: dict[str, Any] = field(default_factory=dict)
    onset_compat_rules: dict[str, list[str]] = field(default_factory=dict)
    max_page_range_span: int = 50


@dataclass
class CompilerConfig:
    stage2a_max_meaning_len: int = 28
    stage2a_max_usage_len: int = 20
    cross_source_min_jaccard: float = 0.35
    compact_tier_limits: dict[str, int] = field(
        default_factory=lambda: {
            "T1_general_lexicon": 50,
            "T1_median": 35,
            "T2_special_or_idiom": 100,
            "T3_grammar_and_redline": 260,
            "T3_hub_max": 300,
            "hub_min_source_cards": 6,
        }
    )
    dual_class_overrides: dict[str, Any] = field(default_factory=dict)
    extra_fw_overrides: dict[str, str] = field(default_factory=dict)
    vocab_merge_targets: dict[str, str] = field(default_factory=dict)
    conflict_specs_path: str = ""
    regression_guards_path: str = ""
    category_labels: dict[str, str] = field(
        default_factory=lambda: {
            "noun": "名词",
            "verb": "动词",
            "adjective": "形容词",
            "time": "时间词",
            "place": "处所词",
            "idiom": "熟语",
            "proverb": "俗谚",
        }
    )
    polysemy_regex: str = r"[①②③④⑤⑥⑦⑧⑨⑩⑴⑵⑶⑷⑸]|(?<!\d)[1234]\s*[.、．]"
    variant_prose_regex: str = r"(?:亦作|又作|或作|又写|亦写|本字|正字|俗作|俗写|借字|训读|同音借|参见)"
    encyclopedic_regex: str = r"(?:植物|动物|昆虫|草本|木本|学名|科属|症状|病症|传统小吃)"
    prosody_contrast_regex: str = r"(?:轻读|重读|轻音|重音|连调|变调不同|文读.*白读|白读.*文读)"
    special_register_regex: str = r"(?:詈词|骂人|讳称|忌讳|婉称|黑话|隐语|谑称|贬称)"


@dataclass
class GatesConfig:
    min_blind_audit_count: int = 600
    min_blind_audit_ratio: float = 0.05
    target_sample_count: int = 660
    target_sample_ratio: float = 0.055
    qa_spotcheck_ratio: float = 0.05
    qa_min_per_doc: int = 3
    closed_class_regex_chars: str = ""
    bare_noun_blacklist: list[str] = field(default_factory=list)
    probe_suite_path: str = ""
    max_regex_open_char_class: int = 3
    max_regex_open_alts: int = 4
    visual_trigger_docs: list[str] = field(default_factory=list)
    visual_trigger_markers: tuple[str, ...] = ("△", "‖", "□")
    speculative_drift_terms: list[str] = field(default_factory=list)


@dataclass
class RetrieverConfig:
    strategy: str = "budgeted"
    max_cards: int = 12
    max_chars_compact: int = 1800
    max_chars_json: int = 3500
    lex_reserve: int = 3
    per_lex_est_compact: int = 70
    per_lex_est_json: int = 250
    max_slots_per_fw: int = 2
    max_slots_per_leaf: int = 3
    open_class_1char_stop: list[str] = field(default_factory=list)
    target_2char_stopwords: list[str] = field(default_factory=list)
    defer_1char_gram: list[str] = field(default_factory=list)
    enclitic_tail_exempt: list[str] = field(default_factory=list)


@dataclass
class DomainProfile:
    """Complete externalized domain profile replacing all hardcoded dialect/book constants."""

    profile_meta: ProfileMetaConfig = field(default_factory=ProfileMetaConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    text_normalization: TextNormalizationConfig = field(default_factory=TextNormalizationConfig)
    pdf_render: PdfRenderConfig = field(default_factory=PdfRenderConfig)
    layout_adapter: LayoutAdapterConfig = field(default_factory=LayoutAdapterConfig)
    compiler: CompilerConfig = field(default_factory=CompilerConfig)
    gates: GatesConfig = field(default_factory=GatesConfig)
    retriever: RetrieverConfig = field(default_factory=RetrieverConfig)


def _merge_section(cls: type, data: dict[str, Any] | None, tuple_fields: set[str] | None = None) -> Any:
    defaults = cls()
    if not data:
        return defaults
    kwargs: dict[str, Any] = {}
    for f_name in defaults.__dataclass_fields__:
        if f_name not in data:
            continue
        val = data[f_name]
        if f_name == "preset_norm_boxes" and isinstance(val, dict):
            val = {k: tuple(float(x) for x in v) for k, v in val.items()}
        elif tuple_fields and f_name in tuple_fields and isinstance(val, list):
            val = tuple(val)
        kwargs[f_name] = val
    return cls(**kwargs)


def default_generic_profile() -> DomainProfile:
    """Return an in-memory neutral DomainProfile without needing a YAML file."""
    return DomainProfile()


def load_profile(path: str | Path | None) -> DomainProfile:
    """Load and validate a DomainProfile from a YAML file (or return neutral defaults if None)."""
    if path is None:
        return default_generic_profile()
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"Profile root must be a YAML mapping: {path}")

    compiler_raw = dict(raw.get("compiler") or {})
    if "v0_guards_path" in compiler_raw and "regression_guards_path" not in compiler_raw:
        compiler_raw["regression_guards_path"] = compiler_raw.pop("v0_guards_path")

    return DomainProfile(
        profile_meta=_merge_section(ProfileMetaConfig, raw.get("profile_meta"), {"target_tables"}),
        sources=_merge_section(SourcesConfig, raw.get("sources")),
        text_normalization=_merge_section(TextNormalizationConfig, raw.get("text_normalization")),
        pdf_render=_merge_section(
            PdfRenderConfig,
            raw.get("pdf_render"),
            {"header_norm_box", "scale_steps", "jpeg_qualities"},
        ),
        layout_adapter=_merge_section(LayoutAdapterConfig, raw.get("layout_adapter")),
        compiler=_merge_section(CompilerConfig, compiler_raw),
        gates=_merge_section(GatesConfig, raw.get("gates"), {"visual_trigger_markers"}),
        retriever=_merge_section(RetrieverConfig, raw.get("retriever")),
    )
