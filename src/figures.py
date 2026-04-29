"""
Render figure crops out of the source PDF and upload them as PNGs.

DI gives us figure bounding-boxes (page + polygon in inches). We re-open the
local PDF with pypdfium2, render the relevant page once at 2x scale (≈144 DPI),
crop each figure on that page with Pillow, and upload to a public-read blob
container so Chainlit can render the resulting URL inline as `cl.Image`.

Kept out of `extract.py` so the extractor stays pure (no I/O beyond DI) and
out of `ingest.py` so the orchestrator doesn't bloat with rendering details.
"""

from __future__ import annotations

import base64
import os
from io import BytesIO
from pathlib import Path

from .extract import FigureRef

FIGURES_CONTAINER = "kb-figures"
RENDER_SCALE = 2.0   # 2x = ~144 DPI; balance between crop fidelity and PNG size

CAPTION_SYSTEM_PROMPT = (
    "You describe a single figure cropped from a product manual. Reply in "
    "≤80 words, plain prose. If the figure has numbered or lettered callouts "
    "(① ② ③, A B C, 1 2 3), name what each one points to in order. If the "
    "figure is purely decorative or you can't tell, reply exactly: "
    "'Decorative figure; no callouts.'"
)


def figures_container_name() -> str:
    return os.environ.get("AZURE_FIGURES_CONTAINER", FIGURES_CONTAINER)


def _polygon_pixel_box(polygon: list[float], scale: float) -> tuple[int, int, int, int]:
    """DI polygon = `[x1,y1,...,x4,y4]` in inches. Convert to a pixel
    (left, top, right, bottom) box at `scale` (where scale=N means 72*N DPI)."""
    xs = polygon[0::2]
    ys = polygon[1::2]
    ppi = 72.0 * scale
    return (
        int(min(xs) * ppi),
        int(min(ys) * ppi),
        int(max(xs) * ppi),
        int(max(ys) * ppi),
    )


def _captions_enabled() -> bool:
    """Vision captioning is opt-in: each figure costs ~1 vision call (~$0.01).
    Set ENABLE_FIGURE_CAPTIONS=1 in .env to turn on."""
    return os.environ.get("ENABLE_FIGURE_CAPTIONS", "").lower() in {"1", "true", "yes"}


def _caption_image(png_bytes: bytes) -> str:
    """gpt-4o (or whichever AZURE_OPENAI_VISION_DEPLOYMENT points at) over a
    base64 image. Returns the description text, or empty string on failure."""
    from .embed import get_azure_openai_client

    client = get_azure_openai_client()
    deployment = os.environ.get("AZURE_OPENAI_VISION_DEPLOYMENT") or os.environ["AZURE_OPENAI_CHAT_DEPLOYMENT"]
    b64 = base64.b64encode(png_bytes).decode("ascii")
    try:
        resp = client.chat.completions.create(
            model=deployment,
            messages=[
                {"role": "system", "content": CAPTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this figure."},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/png;base64,{b64}"},
                        },
                    ],
                },
            ],
            temperature=0,
            max_tokens=200,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"[figures] caption failed: {type(e).__name__}: {e}")
        return ""


def render_and_upload_figures(
    local_pdf: Path,
    figures: list[FigureRef],
    blob_stem: str,
    container_client,
    *,
    scale: float = RENDER_SCALE,
) -> None:
    """Crop every FigureRef out of `local_pdf`, upload PNG to
    `{container}/{blob_stem}/{figure_id}.png`, and set `figure.blob_url` in place.

    Pages are rendered at most once each — we batch all figures per page.
    If ENABLE_FIGURE_CAPTIONS is set, also calls gpt-4o-vision on each crop
    and writes the result back to `figure.caption`."""
    if not figures or local_pdf.suffix.lower() != ".pdf":
        return

    import pypdfium2 as pdfium

    captions_on = _captions_enabled()
    by_page: dict[int, list[FigureRef]] = {}
    for fig in figures:
        by_page.setdefault(fig.page, []).append(fig)

    pdf = pdfium.PdfDocument(str(local_pdf))
    try:
        for page_no, page_figs in by_page.items():
            idx = page_no - 1
            if idx < 0 or idx >= len(pdf):
                continue
            page = pdf[idx]
            bitmap = page.render(scale=scale)  # type: ignore[arg-type]
            pil = bitmap.to_pil()
            w, h = pil.size
            for fig in page_figs:
                left, top, right, bottom = _polygon_pixel_box(fig.polygon, scale)
                # Clamp + skip degenerate boxes.
                left, right = max(0, min(left, w)), max(0, min(right, w))
                top, bottom = max(0, min(top, h)), max(0, min(bottom, h))
                if right - left < 4 or bottom - top < 4:
                    continue
                buf = BytesIO()
                pil.crop((left, top, right, bottom)).save(buf, format="PNG", optimize=True)
                png_bytes = buf.getvalue()
                blob_path = f"{blob_stem}/{fig.figure_id}.png"
                container_client.upload_blob(name=blob_path, data=png_bytes, overwrite=True)
                fig.blob_url = f"{container_client.url}/{blob_path}"
                if captions_on:
                    fig.caption = _caption_image(png_bytes)
    finally:
        pdf.close()


def splice_captions_into_markdown(markdown: str, figures: list[FigureRef]) -> str:
    """Insert each figure's caption as visible text just before the closing
    `</figure>` tag in the raw DI markdown. Splices in DESCENDING offset order
    so earlier offsets stay valid as we mutate the string.

    Result: when chunk.py later finds the `<figure>` HTML in the chunk body,
    the caption rides along inside it — searchable via BM25, visible to the
    LLM at prompt time."""
    captioned = [
        f for f in figures
        if f.caption and f.markdown_length > 0
        and 0 <= f.markdown_offset < len(markdown)
    ]
    if not captioned:
        return markdown
    out = markdown
    for fig in sorted(captioned, key=lambda f: f.markdown_offset, reverse=True):
        end = fig.markdown_offset + fig.markdown_length
        if end > len(out):
            end = len(out)
        span = out[fig.markdown_offset:end]
        injection = f"\n\nFigure description (page {fig.page}): {fig.caption}\n"
        if "</figure>" in span:
            new_span = span.replace("</figure>", f"{injection}</figure>", 1)
        else:
            new_span = span + injection
        out = out[:fig.markdown_offset] + new_span + out[end:]
    return out


def ensure_figures_container(blob_service_client) -> None:
    """Create the figures container with anonymous-blob read access if missing.
    Public read is acceptable for this demo — figures came from docs we shipped.
    For production, swap to per-URL SAS generation in app.py instead."""
    name = figures_container_name()
    container = blob_service_client.get_container_client(name)
    try:
        container.create_container(public_access="blob")
    except Exception as e:
        # Already exists, or RBAC doesn't grant container creation rights.
        # Either way, downstream uploads will surface a clearer error.
        if "ContainerAlreadyExists" not in str(e):
            print(f"[figures] create_container({name}) note: {type(e).__name__}: {e}")
