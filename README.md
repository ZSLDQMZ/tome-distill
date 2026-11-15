[English](./README.md) | [简体中文](./README_zh.md)

# tome-distill

**tome-distill** is an **Apache-2.0** licensed toolkit for **post-OCR structural verification, incremental proofreading, multi-source compilation, and budget-constrained retrieval** over complex, entry- and rule-structured reference works—such as multi-column dictionaries, grammar and rule manuals, regional gazetteers, and domain glossaries.

> **Scope Boundary**: `tome-distill` **does not replace** page-level document parsers such as `MinerU`, `olmOCR`, or `Docling`. Instead, it sits downstream of initial OCR output and source PDFs to solve the specific failure modes of dense reference works: column reading-order mix-ups, silent entry omissions, rare character corruption, fragile full-JSON rewrites during proofreading, cross-book scholarly disagreements, and noisy single-character false hits during RAG retrieval.

---

## Why `tome-distill` (Problems Solved & Use Cases)

While general OCR handles single-column papers well, turning scanned **two-column dictionaries, structured rule manuals, glossaries, and grammatical reference books** into reliable LLM knowledge bases hits four recurring bottlenecks:

1. **Cross-column reading drift & silent entry drops**: OCR tools frequently read horizontally across columns or silently drop paragraphs. Looking only at extracted JSON makes it nearly impossible to know which pages or headwords went missing.
2. **Full-page VLM re-extraction is costly and brittle**: Sending large uncompressed page scans to vision-language models often hits payload timeouts, while asking an LLM to regenerate full JSON cards during proofreading frequently breaks valid existing fields. In addition, relying on AGPL PDF rendering libraries creates licensing friction for permissive projects.
3. **Rare characters, orthographic variants & conflicting sources**: Standard Traditional-to-Simplified conversion can merge distinct lexical/etymological characters needed for disambiguation and alter string lengths. Meanwhile, when two authoritative reference books disagree on a usage rule, naive majority voting erases legitimate evidence.
4. **Single-character substring collisions flood LLM context**: Naïve keyword matching triggers dozens of function-word entries whenever common characters appear inside ordinary two-character words, exhausting prompt budgets with irrelevant context.

**Common use cases**:
- Building grounded, citation-traceable lexical and grammatical knowledge bases for translation, linguistic evaluation, or domain RAG;
- Running completeness audits and targeted crop-based visual proofreading on digitized reference books;
- Deterministic sparse retrieval under strict card and character budgets with automatic compound-word masking.

---

## Pipeline Overview & Implementation Design

```text
Inputs: Initial OCR Text/Boxes + Optional Source PDF + YAML DomainProfile
   │
   ├─► [1] TOC Partitioning & Multi-Column Layout Audit (toc.py, adapters.py)
   │        └─ Line-span closure checks, column coordinate sorting, header vs. index audit
   │
   ├─► [2] Targeted Strip Cropping & Collage Generation [Optional] (pdf_render.py, ocr_helper.py)
   │        └─ Permissive pypdfium2 cropping of suspect regions into compact grayscale JPEG strips
   │
   ├─► [3] Dual-Layer Schema & Field-Level Incremental Patching (schema.py, field_diffs.py)
   │        └─ Inference vs. provenance isolation, meta-leak guards, independent patch review
   │
   ├─► [4] Character-Protected Normalization & Compilation (text_utils.py, compiler.py)
   │        └─ 1:1 protected OpenCC conversion, Union-Find merging, conflict preservation, lint checks
   │
   └─► [5] Compound-Masked & Budgeted Retrieval (retriever.py, zh_compounds.py)
            └─ Common compound masking + prioritized slot/character budget -> LLM prompt context
```

### Stage-by-Stage Implementation

#### 1. Deterministic Structural & Layout Verification (`toc.py`, `adapters.py`)
- **Line-range closure (`toc.py`)**: Before running any LLM calls, checks that extracted TOC/section line spans cover the document cleanly without gaps or accidental overlaps (supporting both strict single-column partitions and multi-column line-union checks).
- **Column re-ordering & Header × Index cross-checking (`DoubleColumnDictAdapter`)**: Uses column-gutter coordinates to re-sort interleaved text blocks into strict column-first reading order (left column top-to-bottom, then right column). It then cross-checks extracted headwords against running page-header spans (`[first ... last]`) and back-of-book indexes to pinpoint missing or misordered pages at zero token cost. Custom layouts can subclass `LayoutAdapter`.

#### 2. Targeted Strip Cropping for Visual Proofreading (`pdf_render.py`, `ocr_helper.py`, Optional)
- **Permissive-license PDF rasterization**: Built exclusively on **Apache-2.0 / BSD `pypdfium2`** rather than AGPL PDF bindings, keeping downstream distribution permissive.
- **Compact JPEG strip collages**: Renders and crops only flagged pages, header regions, or entry bounding boxes from the source PDF by normalized coordinates, applies background bleaching, labels each strip, and steps down resolution/quality into compact grayscale JPEG sheets (capped at `215 KB` by default). This keeps both human spot-checking and multimodal VLM verification fast and payload-safe. Optional `RapidOCR` support assists header checks.

#### 3. Dual-Layer Entry Schema & Field-Level Patching (`schema.py`, `field_diffs.py`)
- **Clean inference view vs. source provenance**: Every entry isolates:
  - `judge_view`: clean definitions, variants, usage rules, and parallel examples intended for downstream LLM prompts—validated to reject editorial/citation meta-text (such as page numbers or `U+XXXX` strings) from leaking into inference;
  - `provenance`: source book ID, page numbers, verbatim OCR quote, and extraction/review attribution;
  - Scans raw string leaves for hidden control characters prior to JSON serialization and verifies record-count conservation across merges.
- **Incremental field patches (`field_diffs.py`)**: Enforces reviewer separation (`extractor != reviewer`) and applies compact patches (`updates`, `insert_after` with sequential ID assignment beyond `Z -> AA, AB...`, and `delete_ids`) rather than full-card rewrites, alongside stratified blind sampling for QA checks.

#### 4. Character-Protected Normalization & Multi-Source Compilation (`text_utils.py`, `compiler.py`)
- **Length-preserving protected conversion (`text_utils.py`)**: Runs OpenCC Traditional/Simplified normalization while protecting every character listed in `protected_chars` and preserving an exact **1:1 string length mapping** so character match spans never drift.
- **Cross-source clustering & explicit conflict linking (`compiler.py`)**: Uses Union-Find clustering to merge shared entries across reference books, links bidirectional cross-references (`⇨〖target〗`), and preserves contradictory scholarly claims side-by-side rather than overwriting them. Lints custom trigger regexes against unbounded wildcards (`.*`) and compiles tiered compact summaries plus multi-key indexes.

#### 5. Compound-Masked & Budgeted Retrieval (`retriever.py`, `zh_compounds.py`)
- **Preventing substring collisions (`retriever.py`, `zh_compounds.py`)**: Masks out common multi-character compounds (supplied externally or built via `zh_compounds.py`) before scanning single-character grammatical entries—eliminating false triggers when common words contain functional morphemes, while supporting configurable word-final exceptions.
- **Multi-mode anchors & prioritized budgeting**: Supports direct lexical lookup (`dialect_only`), standard-language gloss anchor matching (`standard_anchor`), and draft-edit span boosting (`hybrid`). Under the default `budgeted` strategy, allocates critical grammatical constraints first, caps repeated slots per function word or taxonomy leaf, and reserves space for high-relevance vocabulary and idioms (`longest` and `full` strategies are also supported).

---

## Installation & Quickstart

### 1. Install

Core engine (CPU-only, requires Python >= 3.11, `PyYAML`, and `opencc-python-reimplemented`; zero PDF/OCR dependencies):

```bash
pip install -e .
```

Install optional extras as needed:

```bash
pip install -e ".[pdf]"        # pypdfium2 + Pillow + NumPy for crop rendering & JPEG sheets
pip install -e ".[ocr]"        # RapidOCR + OpenCV helpers for header / narrow-column OCR
pip install -e ".[compounds]"  # jieba for building general compound-word mask lists
pip install -e ".[all]"        # Install all optional extras (pdf + ocr + compounds)
```

### 2. Run the Synthetic Toy Dictionary Pipeline

The example script programmatically generates a 1-page fictional double-column PDF ("Starlight Toy Lexicon") and 10 synthetic entries to walk through PDF rendering, JPEG strip cropping, TOC closure, `field_diffs` patching, compilation checks, and budgeted retrieval (requires `.[pdf]`; on Windows PowerShell run `$env:PYTHONPATH="src"` first):

```bash
# 1. Run end-to-end fictional toy dictionary pipeline
PYTHONPATH=src python examples/run_toy_pipeline.py --work-dir demo_out

# 2. Run the full stdlib unittest suite (AST zero-AGPL check, schema guards, retrieval tests)
PYTHONPATH=src python -B -m unittest discover -s tests -v
```

---

## Configuration & Domain Adaptation (`DomainProfile`)

`tome-distill` separates all domain vocabulary, layout coordinates, regex guards, and retrieval budgets into a single neutral template: [`profiles/generic_profile.yaml`](./profiles/generic_profile.yaml). Copy and edit it to adapt to a new language or book collection without modifying engine code:

- **Sources & taxonomy (`profile_meta`, `sources`)**: Target language tag, output tables, custom grammatical/lexical `taxonomy_tree`, canonical document list, and optional held-out evaluation paths.
- **Text normalization & safety (`text_normalization`)**: OpenCC mode, `protected_chars` list, character aliases, inline-note heuristics, and regex patterns preventing editorial meta-text from leaking into `judge_view`.
- **PDF cropping (`pdf_render`)**: Rasterization DPI, header/column normalized boxes (`header_norm_box`, `preset_norm_boxes`), strip width, background bleaching threshold, and JPEG payload cap (`max_img_bytes`).
- **Layout & compilation (`layout_adapter`, `compiler`, `gates`)**: Column gutter thresholds, cross-source similarity cutoffs, tiered compact character limits, blind QA ratios, and special markers flagging visual inspection.
- **Retrieval budget (`retriever`)**: Allocation strategy (`budgeted` | `longest` | `full`), max cards, max character caps (`max_chars_compact` / `max_chars_json`), lexical reserves, and slot limits per entry/category.

For non-two-column layouts (e.g., three-column tables or digital volumes), subclass `LayoutAdapter` in [`src/tome_distill/adapters.py`](./src/tome_distill/adapters.py).

---

## Repository Layout & Scope Notes

```text
.
├── pyproject.toml                  # Package metadata (tome-distill / tome_distill) & optional extras
├── LICENSE                         # Apache License 2.0
├── THIRD_PARTY_NOTICES.md          # Permissive PDF stack & dependency notices
├── README.md                       # English documentation
├── README_zh.md                    # Simplified Chinese documentation
├── profiles/
│   └── generic_profile.yaml        # Neutral generic DomainProfile YAML template
├── src/tome_distill/
│   ├── __init__.py
│   ├── profile.py                  # DomainProfile dataclasses & YAML loader
│   ├── toc.py                      # TOC parsing & line-span closure verification
│   ├── adapters.py                 # LayoutAdapter ABC & two-column dictionary adapter
│   ├── pdf_render.py               # [pdf] pypdfium2 cropping & compact JPEG strip sheets
│   ├── ocr_helper.py               # [ocr] Optional RapidOCR + OpenCV header helpers
│   ├── schema.py                   # Dual-layer validation, leak guards & count-conserving merge
│   ├── field_diffs.py              # Reviewer isolation, field_diffs patching & stratified QA
│   ├── text_utils.py               # 1:1 character-protected OpenCC normalization & control scan
│   ├── compiler.py                 # Union-Find merging, conflict links, regex lint & indexing
│   ├── retriever.py                # Compound-masked, anchor-aware budgeted retriever
│   └── zh_compounds.py             # [compounds] Optional general compound mask builder
├── examples/
│   └── run_toy_pipeline.py         # End-to-end demo on a fictional 1-page toy dictionary PDF
└── tests/
    └── test_end_to_end.py          # Stdlib unittest suite
```

**Scope Notes**:
- Designed strictly as a **post-OCR curation, verification, compilation, and retrieval layer**; skewed scans or handwritten manuscripts require upstream deskewing and OCR first.
- Precision on specialized corpora depends on populating `protected_chars`, editorial leak patterns, and compound masks in your `DomainProfile` YAML.
- Licensed under [Apache-2.0](./LICENSE); contains zero copyrighted scans, copied third-party code, or model weights.
