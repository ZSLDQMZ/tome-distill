"""Optional [pdf] rasterization, coordinate-safe cropping, and <=215KB JPEG compression using pypdfium2."""

from __future__ import annotations

import io
import os
import sys
from pathlib import Path
from typing import Any, Sequence

from .profile import PdfRenderConfig


def _require_pdfium():
    try:
        import pypdfium2 as pdfium  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "PDF rasterization requires optional extras: pip install 'tome-distill[pdf]'"
        ) from exc
    return pdfium


def _require_pil():
    try:
        from PIL import Image, ImageDraw, ImageFont  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Image processing requires Pillow: pip install 'tome-distill[pdf]'"
        ) from exc
    return Image, ImageDraw, ImageFont


def norm_box_to_pdfium_crop(
    page_width: float,
    page_height: float,
    norm_box: Sequence[float],
) -> tuple[float, float, float, float]:
    """Convert top-left normalized box [fx0, fy0, fx1, fy1] to pypdfium2 inward crop margins.

    pypdfium2's PdfPage.render(..., crop=(left, bottom, right, top)) expects points trimmed inward
    from each edge in PDF bottom-left coordinates.
    """
    if len(norm_box) != 4:
        raise ValueError(f"norm_box must have 4 elements [fx0, fy0, fx1, fy1], got {norm_box!r}")
    fx0, fy0, fx1, fy1 = (float(v) for v in norm_box)
    if not (0.0 <= fx0 < fx1 <= 1.0 and 0.0 <= fy0 < fy1 <= 1.0):
        raise ValueError(f"norm_box coordinates out of [0, 1] order: {norm_box!r}")
    left = page_width * fx0
    bottom = page_height * (1.0 - fy1)
    right = page_width * (1.0 - fx1)
    top = page_height * fy0
    return (left, bottom, right, top)


def bleach_grayscale_image(pil_img: Any, threshold: int = 224) -> Any:
    """Whiten scanned paper backgrounds above threshold while keeping ink crisp."""
    img = pil_img.convert("L") if pil_img.mode != "L" else pil_img
    if threshold >= 255:
        return img
    try:
        import numpy as np  # type: ignore

        arr = np.asarray(img, dtype=np.uint8)
        out = arr.copy()
        out[out >= threshold] = 255
        if threshold > 32:
            dark_mask = out < threshold
            scaled = np.clip((out[dark_mask].astype(np.float32) * (245.0 / threshold)), 0, 245).astype(np.uint8)
            out[dark_mask] = scaled
        Image, _, _ = _require_pil()
        return Image.fromarray(out)
    except ImportError:
        lut = [255 if i >= threshold else min(245, int(i * (245.0 / max(1, threshold)))) for i in range(256)]
        return img.point(lut)


def encode_jpeg_under_cap(
    pil_img: Any,
    *,
    max_bytes: int = 215_000,
    scale_steps: Sequence[float] = (1.0, 0.90, 0.80, 0.72, 0.64, 0.56),
    jpeg_qualities: Sequence[int] = (78, 70, 62, 56, 48, 42),
    min_dim: int = 200,
) -> tuple[bytes, dict[str, Any]]:
    """Compress an image to grayscale JPEG strictly under max_bytes via a scale/quality ladder."""
    Image, _, _ = _require_pil()
    base = pil_img.convert("L") if pil_img.mode != "L" else pil_img
    orig_w, orig_h = base.size

    def _to_jpeg(img_obj: Any, q: int) -> bytes:
        buf = io.BytesIO()
        img_obj.save(buf, format="JPEG", quality=q, optimize=True)
        return buf.getvalue()

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    smallest_data: bytes = b""
    smallest_meta: dict[str, Any] = {}

    for scale in scale_steps:
        tw = max(1, int(round(orig_w * scale)))
        th = max(1, int(round(orig_h * scale)))
        if scale < 1.0 and min(tw, th) < min_dim and min(orig_w, orig_h) >= min_dim:
            continue
        cur = base if scale >= 0.999 else base.resize((tw, th), resampling)
        for q in jpeg_qualities:
            data = _to_jpeg(cur, int(q))
            meta = {"bytes": len(data), "width": cur.width, "height": cur.height, "scale": scale, "quality": int(q)}
            if not smallest_data or len(data) < len(smallest_data):
                smallest_data = data
                smallest_meta = meta
            if len(data) <= max_bytes:
                return data, meta

    # Emergency downscale + low quality fallback if still over cap.
    cur = base
    cur_scale = float(scale_steps[-1]) if scale_steps else 0.5
    while len(smallest_data) > max_bytes and min(cur.width, cur.height) > 64:
        cur_scale *= 0.8
        cur = base.resize(
            (max(32, int(round(orig_w * cur_scale))), max(32, int(round(orig_h * cur_scale)))),
            resampling,
        )
        for q in (40, 32, 24):
            data = _to_jpeg(cur, q)
            meta = {"bytes": len(data), "width": cur.width, "height": cur.height, "scale": round(cur_scale, 3), "quality": q}
            if len(data) < len(smallest_data):
                smallest_data = data
                smallest_meta = meta
            if len(data) <= max_bytes:
                return data, meta

    return smallest_data, smallest_meta


def _open_pdf_document(pdf_source: str | Path | bytes):
    pdfium = _require_pdfium()
    if isinstance(pdf_source, (str, Path)):
        return pdfium.PdfDocument(str(pdf_source))
    return pdfium.PdfDocument(pdf_source)


def get_pdf_page_count(pdf_source: str | Path | bytes) -> int:
    doc = _open_pdf_document(pdf_source)
    try:
        return len(doc)
    finally:
        doc.close()


def render_pdf_page_image(
    pdf_source: str | Path | bytes,
    page_num: int,
    *,
    dpi: int = 150,
    norm_box: Sequence[float] | None = None,
    bleach_threshold: int | None = 224,
) -> Any:
    """Render a 1-based PDF page (or normalized sub-box) via pypdfium2 into a grayscale PIL Image."""
    doc = _open_pdf_document(pdf_source)
    try:
        if page_num < 1 or page_num > len(doc):
            raise IndexError(f"page_num={page_num} out of bounds 1..{len(doc)}")
        page = doc[page_num - 1]
        render_kwargs: dict[str, Any] = {"scale": dpi / 72.0, "grayscale": True}
        if norm_box is not None:
            w, h = page.get_size()
            render_kwargs["crop"] = norm_box_to_pdfium_crop(w, h, norm_box)
        bitmap = page.render(**render_kwargs)
        pil_im = bitmap.to_pil()
        if bleach_threshold is not None:
            pil_im = bleach_grayscale_image(pil_im, threshold=bleach_threshold)
        return pil_im
    finally:
        doc.close()


def render_page_jpeg_capped(
    pdf_source: str | Path | bytes,
    page_num: int,
    out_path: str | Path | None = None,
    *,
    cfg: PdfRenderConfig | None = None,
    preset: str | None = None,
    norm_box: Sequence[float] | None = None,
    is_crop: bool = False,
) -> tuple[bytes, dict[str, Any]]:
    """Render a PDF page or crop box and compress under max_img_bytes / max_crop_bytes."""
    conf = cfg or PdfRenderConfig()
    box = norm_box
    if box is None and preset:
        if preset not in conf.preset_norm_boxes:
            raise KeyError(f"Unknown render preset {preset!r}; available={sorted(conf.preset_norm_boxes)}")
        box = conf.preset_norm_boxes[preset]
        if preset != "full":
            is_crop = True

    dpi = conf.crop_dpi if is_crop else conf.dpi
    cap = conf.max_crop_bytes if is_crop else conf.max_img_bytes
    im = render_pdf_page_image(
        pdf_source,
        page_num,
        dpi=dpi,
        norm_box=box,
        bleach_threshold=conf.bleach_threshold,
    )
    data, meta = encode_jpeg_under_cap(
        im,
        max_bytes=cap,
        scale_steps=conf.scale_steps,
        jpeg_qualities=conf.jpeg_qualities,
        min_dim=conf.min_dim,
    )
    meta["page_num"] = page_num
    meta["preset"] = preset or ("crop" if box else "full")
    if out_path is not None:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        meta["path"] = str(p)
    return data, meta


def render_pdf_pages(
    pdf_source: str | Path | bytes,
    pages: Sequence[int],
    out_dir: str | Path,
    *,
    cfg: PdfRenderConfig | None = None,
    prefix: str = "page",
) -> list[dict[str, Any]]:
    """Batch-render a sequence of 1-based pages to capped JPEG files in out_dir."""
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for pno in pages:
        out_file = out_root / f"{prefix}_p{pno:04d}.jpg"
        _, meta = render_page_jpeg_capped(pdf_source, pno, out_path=out_file, cfg=cfg)
        results.append(meta)
    return results


def _load_label_font(font_paths: Sequence[str], size: int = 20) -> Any:
    _, _, ImageFont = _require_pil()
    candidates = list(font_paths)
    if sys.platform.startswith("win"):
        windir = os.environ.get("WINDIR", r"C:\Windows")
        candidates.extend(
            [
                os.path.join(windir, "Fonts", "arial.ttf"),
                os.path.join(windir, "Fonts", "msyh.ttc"),
                os.path.join(windir, "Fonts", "segoeui.ttf"),
            ]
        )
    elif sys.platform == "darwin":
        candidates.extend(
            [
                "/System/Library/Fonts/Helvetica.ttc",
                "/System/Library/Fonts/Supplemental/Arial.ttf",
            ]
        )
    else:
        candidates.extend(
            [
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
            ]
        )
    for fp in candidates:
        if fp and os.path.isfile(fp):
            try:
                return ImageFont.truetype(fp, size=size)
            except OSError:
                continue
    return ImageFont.load_default()


def render_header_sheets_capped(
    pdf_source: str | Path | bytes,
    pages: Sequence[int],
    out_dir: str | Path,
    *,
    cfg: PdfRenderConfig | None = None,
    norm_box: Sequence[float] | None = None,
    prefix: str = "headers",
) -> list[dict[str, Any]]:
    """Render paginated header contact sheets (grayscale JPEG, strictly <= max_img_bytes)."""
    Image, ImageDraw, _ = _require_pil()
    conf = cfg or PdfRenderConfig()
    box = norm_box or conf.header_norm_box
    out_root = Path(out_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    resampling = getattr(Image, "Resampling", Image).LANCZOS
    font = _load_label_font(conf.font_paths, size=20)
    per_sheet = max(1, conf.max_strips_per_sheet)
    strip_w = max(128, conf.strip_width)
    tag_w = max(48, conf.tag_width)
    doc = _open_pdf_document(pdf_source)
    sheet_metas: list[dict[str, Any]] = []

    try:
        for sheet_idx in range(0, len(pages), per_sheet):
            chunk = pages[sheet_idx : sheet_idx + per_sheet]
            strips: list[tuple[int, Any]] = []
            for pno in chunk:
                if pno < 1 or pno > len(doc):
                    raise IndexError(f"page_num={pno} out of bounds 1..{len(doc)}")
                page = doc[pno - 1]
                w, h = page.get_size()
                crop = norm_box_to_pdfium_crop(w, h, box)
                im = page.render(scale=conf.header_dpi / 72.0, crop=crop, grayscale=True).to_pil()
                im = bleach_grayscale_image(im, threshold=conf.bleach_threshold)
                scaled_h = max(16, int(round(im.height * (strip_w / max(1, im.width)))))
                im_resized = im.resize((strip_w, scaled_h), resampling)
                strips.append((pno, im_resized))

            pad = 4
            canvas_w = tag_w + strip_w
            canvas_h = sum(st.height + pad for _, st in strips) + pad
            canvas = Image.new("L", (canvas_w, max(32, canvas_h)), color=255)
            draw = ImageDraw.Draw(canvas)
            y_cursor = pad
            for pno, st in strips:
                draw.text((8, y_cursor + max(2, st.height // 4)), f"p.{pno}", fill=0, font=font)
                canvas.paste(st, (tag_w, y_cursor))
                draw.line([(0, y_cursor + st.height + 1), (canvas_w, y_cursor + st.height + 1)], fill=200, width=1)
                y_cursor += st.height + pad

            data, meta = encode_jpeg_under_cap(
                canvas,
                max_bytes=conf.max_img_bytes,
                scale_steps=conf.scale_steps,
                jpeg_qualities=conf.jpeg_qualities,
                min_dim=conf.min_dim,
            )
            sheet_file = out_root / f"{prefix}_sheet_{sheet_idx // per_sheet + 1:03d}.jpg"
            sheet_file.write_bytes(data)
            meta["path"] = str(sheet_file)
            meta["pages"] = list(chunk)
            sheet_metas.append(meta)
    finally:
        doc.close()

    return sheet_metas
