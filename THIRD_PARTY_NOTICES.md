# Third-Party Notices

## Original Source & Zero Bundled Assets Declaration

This repository (**`tome-distill`** / package `tome_distill`) contains **100% original source code** authored for post-OCR structural verification, dual-blind micro-patch curation, cross-source compilation, and budgeted knowledge retrieval.

- **Zero copied third-party source code**: No third-party OCR engines, PDF libraries, or NLP toolkits are copied or vendored into this repository.
- **Zero bundled model weights**: No OCR models, ONNX weights, or neural checkpoints are included.
- **Zero copyrighted reference works**: No published dictionaries, gazetteers, scanned book pages, or proprietary corpus cards are included. All examples and tests run on a programmatically drawn, 100% fictional single-page toy dictionary PDF and synthetic toy cards.

All original code in this repository is licensed under the **Apache License, Version 2.0** (see [`LICENSE`](./LICENSE)).

---

## Pure Permissive-License PDF Rasterization (`pypdfium2` Replaces AGPL)

To keep `tome-distill` 100% compatible with permissive Apache-2.0 distribution and avoid AGPL copyleft contagion:

- **Zero AGPL-licensed PDF dependencies**: No AGPL PDF tooling (`PyMuPDF` / `fitz`) is used anywhere in the codebase (enforced by AST tests).
- Optional PDF page rasterization and margin-calibrated strip cropping (`[pdf]`) are implemented exclusively via **`pypdfium2`** (**Apache-2.0 / BSD-3-Clause**), enabling pure permissive-license PDF rendering.
- Core schema validation, TOC line-partition verification, `field_diffs` patching, compilation, and retrieval require **zero PDF or OCR dependencies**.

---

## Third-Party Components & Acknowledgments

| Component | Role in `tome-distill` | License | Distribution Group |
| --- | --- | --- | --- |
| **PyYAML** | `DomainProfile` YAML schema loading & validation | **MIT** | Core dependency |
| **OpenCC** | Protected-character 1:1 Traditional-to-Simplified normalization | **Apache-2.0** | Core dependency |
| **pypdfium2** | Pure permissive-license PDF page rasterization & calibrated crop rendering (replaces AGPL `fitz`) | **Apache-2.0 / BSD-3-Clause** | Optional (`[pdf]`) |
| **Pillow** | Grayscale JPEG strip collage assembly, step-down compression & toy PDF rendering | **HPND** | Optional (`[pdf]`) |
| **NumPy** | Pixel array thresholding/bleaching & column-gutter projection helpers | **BSD-3-Clause** | Optional (`[pdf]`) |
| **RapidOCR** (`rapidocr_onnxruntime`) | Optional header & narrow-column OCR cross-verification helper | **Apache-2.0** | Optional (`[ocr]`) |
| **OpenCV** (`opencv-python-headless`) | Optional image preprocessing for header/column OCR helpers | **Apache-2.0** | Optional (`[ocr]`) |
| **jieba** | Optional non-grammatical common compound mask generation | **MIT** | Optional (`[compounds]`) |

All dependencies listed above are installed independently by the user from PyPI or their respective upstream channels under their own license terms.
