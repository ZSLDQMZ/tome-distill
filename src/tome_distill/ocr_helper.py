"""Optional [ocr] pluggable OCR wrapper with zero top-level rapidocr/cv2 dependency."""

from __future__ import annotations

from typing import Any, Callable, Sequence


class HeaderOcrEngine:
    """Lazy-loading RapidOCR wrapper that supports external OCR function injection for testing."""

    def __init__(self, ocr_fn: Callable[[Any], list[dict[str, Any]]] | None = None) -> None:
        self._custom_fn = ocr_fn
        self._engine: Any = None

    def _ensure_engine(self) -> Any:
        if self._engine is not None:
            return self._engine
        try:
            from rapidocr_onnxruntime import RapidOCR  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "OCR requires optional extras: pip install 'tome-distill[ocr]'"
            ) from exc
        self._engine = RapidOCR()
        return self._engine

    def run(self, image_input: Any) -> list[dict[str, Any]]:
        """Run OCR on an image path, bytes, numpy array, or PIL Image."""
        if self._custom_fn is not None:
            return list(self._custom_fn(image_input))

        engine = self._ensure_engine()
        payload = image_input
        if hasattr(image_input, "convert") and hasattr(image_input, "size"):
            try:
                import numpy as np  # type: ignore

                payload = np.asarray(image_input.convert("RGB"))
            except ImportError as exc:
                raise ImportError(
                    "Converting PIL Image for RapidOCR requires numpy: pip install 'tome-distill[pdf,ocr]'"
                ) from exc

        raw_result, _ = engine(payload)
        items: list[dict[str, Any]] = []
        for entry in raw_result or []:
            if len(entry) >= 3:
                box, text, score = entry[0], str(entry[1]), float(entry[2])
                items.append({"box": box, "text": text, "score": score})
        return items


def ocr_image_text(
    image_input: Any,
    engine: HeaderOcrEngine | None = None,
    *,
    suspect_chars: Sequence[str] = (),
) -> dict[str, Any]:
    """Extract concatenated OCR lines and flag any configured domain suspect characters."""
    eng = engine or HeaderOcrEngine()
    detections = eng.run(image_input)
    text = "\n".join(item.get("text", "") for item in detections if item.get("text"))
    scores = [float(item["score"]) for item in detections if "score" in item]
    suspect_set = {ch for ch in suspect_chars if ch}
    flagged = sorted(ch for ch in suspect_set if ch in text)
    return {
        "text": text,
        "detections": detections,
        "mean_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
        "suspect_chars_hit": flagged,
    }
