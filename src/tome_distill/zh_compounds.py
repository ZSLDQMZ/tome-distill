"""Optional [compounds] external compound-mask table builder with jieba or word-freq file support."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .text_utils import is_cjk_char


def _load_freq_pairs_from_file(path: Path) -> list[tuple[str, int]]:
    pairs: list[tuple[str, int]] = []
    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        word = parts[0].strip()
        try:
            freq = int(float(parts[1]))
        except ValueError:
            continue
        pairs.append((word, freq))
    return pairs


def _load_jieba_dict_pairs() -> list[tuple[str, int]]:
    try:
        import jieba  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Generating compounds without an external word-frequency file requires jieba: "
            "pip install 'tome-distill[compounds]'"
        ) from exc

    dict_file = jieba.get_dict_file()
    try:
        raw_bytes = dict_file.read()
    finally:
        dict_file.close()

    pairs: list[tuple[str, int]] = []
    for raw in raw_bytes.decode("utf-8", errors="replace").splitlines():
        parts = raw.strip().split()
        if len(parts) < 2:
            continue
        try:
            pairs.append((parts[0], int(parts[1])))
        except ValueError:
            continue
    return pairs


def generate_common_compounds(
    freq_source: str | Path | Iterable[tuple[str, int]] | None = None,
    *,
    min_freq: int = 200,
    min_len: int = 2,
    max_len: int = 4,
    protect_chars: Iterable[str] = (),
    exclude_words: Iterable[str] = (),
    out_path: str | Path | None = None,
) -> dict[str, int]:
    """Build a pure-CJK common compound mask table excluding protected dialect chars/words.

    When freq_source is supplied (a file path or iterable of (word, freq) pairs), no jieba
    dependency is imported. Otherwise lazily falls back to jieba's dictionary.
    """
    if isinstance(freq_source, (str, Path)):
        raw_pairs = _load_freq_pairs_from_file(Path(freq_source))
    elif freq_source is not None:
        raw_pairs = list(freq_source)
    else:
        raw_pairs = _load_jieba_dict_pairs()

    protected = {ch for ch in protect_chars if ch}
    excluded = {str(w).strip() for w in exclude_words if str(w).strip()}
    out: dict[str, int] = {}

    for word, freq in raw_pairs:
        w = str(word or "").strip()
        if not (min_len <= len(w) <= max_len):
            continue
        if freq < min_freq:
            continue
        if w in excluded:
            continue
        if any(ch in protected for ch in w):
            continue
        if not all(is_cjk_char(ch) for ch in w):
            continue
        prev = out.get(w, 0)
        if freq > prev:
            out[w] = freq

    ordered = dict(sorted(out.items(), key=lambda kv: (-kv[1], kv[0])))
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta": {
                "min_freq": min_freq,
                "min_len": min_len,
                "max_len": max_len,
                "count": len(ordered),
            },
            "compounds": ordered,
        }
        p.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    return ordered
