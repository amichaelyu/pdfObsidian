import argparse
import os

import pymupdf as fitz  # PyMuPDF (modern import name)

PDF_PATH = "ph81_2025_26_q3_chapter_4.pdf"
OUTPUT_DIR = "outputs"  # everything this script writes goes here
DPI = 200  # higher = sharper text, slower / more memory
MODEL = "mlx-community/dots.ocr-bf16"

# Ollama model for topic extraction. gemma3:4b is fast; gemma3:12b is sharper.
TOPICS_MODEL = "gemma3:4b"

# dots.ocr expects an explicit OCR instruction, not an empty prompt.
PROMPT = "Extract the text from this page as Markdown."

TOPICS_PROMPT = (
    "You are given the text of one page from a document. "
    "List the key topics covered on this page as concise bullet points. "
    "Output only the bullets, no preamble.\n\n--- PAGE TEXT ---\n{page}"
)


def render_pages(pdf_path, out_dir=OUTPUT_DIR, dpi=DPI):
    """Rasterize each PDF page to a PNG and yield (page_num, image_path)."""
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72  # PDF base resolution is 72 DPI
    matrix = fitz.Matrix(zoom, zoom)
    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(matrix=matrix)
        out = os.path.join(out_dir, f"page_{page_num:03d}.png")
        pix.save(out)
        yield page_num, out


def ocr_pages(pdf_path):
    """Run the VLM OCR over every rendered page. Returns list of page texts."""
    from mlx_vlm import load, generate
    from mlx_vlm.prompt_utils import apply_chat_template
    from mlx_vlm.utils import load_config

    model, processor = load(MODEL)
    config = load_config(MODEL)

    pages_text = []
    for page_num, image_path in render_pages(pdf_path):
        formatted_prompt = apply_chat_template(
            processor, config, PROMPT, num_images=1
        )
        output = generate(
            model, processor, formatted_prompt, [image_path], max_tokens=10000
        )
        print(f"--- page {page_num} ---")
        print(output.text)
        pages_text.append(output.text)
    return pages_text


def extract_topics(pages_text, model=TOPICS_MODEL):
    """Ask an Ollama model for the key topics on each page. Returns list of strings."""
    import ollama

    topics = []
    for page_num, page in enumerate(pages_text):
        if not page.strip():
            topics.append("")
            continue
        resp = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": TOPICS_PROMPT.format(page=page)}],
        )
        text = resp["message"]["content"].strip()
        print(f"=== topics: page {page_num} ===")
        print(text)
        topics.append(text)
    return topics


def main():
    parser = argparse.ArgumentParser(description="OCR a PDF to Markdown and extract key topics.")
    parser.add_argument("pdf", nargs="?", default=PDF_PATH, help="Path to the PDF.")
    parser.add_argument(
        "--topics-model", default=TOPICS_MODEL,
        help="Ollama model for topic extraction (e.g. gemma3:4b, gemma3:12b).",
    )
    parser.add_argument(
        "--no-topics", action="store_true", help="Skip topic extraction.",
    )
    args = parser.parse_args()

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pages_text = ocr_pages(args.pdf)
    output_md = os.path.join(OUTPUT_DIR, "output.md")
    with open(output_md, "w") as f:
        f.write("\n\n".join(pages_text))
    print(f"Wrote {output_md}")

    if not args.no_topics:
        topics = extract_topics(pages_text, model=args.topics_model)
        topics_md = os.path.join(OUTPUT_DIR, "topics.md")
        with open(topics_md, "w") as f:
            for page_num, t in enumerate(topics):
                f.write(f"## Page {page_num}\n\n{t}\n\n")
        print(f"Wrote {topics_md}")


if __name__ == "__main__":
    main()
