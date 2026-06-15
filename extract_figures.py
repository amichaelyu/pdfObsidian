#!/usr/bin/env python3
"""
Extract figures from PDFs whose diagrams are VECTOR drawings
(not embedded raster images). Standard tools like `pdfimages` or
PyMuPDF's get_images() return nothing for these.

Why not just use page.cluster_drawings()? Page-wide rule lines and the
colored accent bars beside callout boxes act as "bridges": a single long
path can touch both a figure and the header, so the clusterer merges them
and the crop ends up containing surrounding text. This script removes those
bridge elements FIRST, then clusters the remaining stroke fragments with a
fast spatial union-find, so each figure comes out isolated.

Usage:  python extract_figures.py input.pdf [output_dir]
Requires: pymupdf  (pip install pymupdf)
"""
import sys, os
from collections import defaultdict
import pymupdf as fitz  # PyMuPDF (modern import name)


def is_bridge(r):
    """Long horizontal page rules and thin vertical accent bars."""
    if r.width > 300 and r.height < 6:   # page-wide rule
        return True
    if r.width < 12 and r.height > 40:   # vertical accent bar
        return True
    return False


def is_figure(r):
    """A surviving cluster big enough to be a real diagram."""
    return r.width >= 45 and r.height >= 3 and not (r.width > 300 and r.height < 6)


def absorb_labels(fig, words, margin=12):
    """Grow a figure box to include text labels (e.g. vector/Greek symbols)
    that sit just outside the strokes. Labels are font glyphs, so they're
    invisible to get_drawings() and would otherwise be clipped. Only words
    that fall FULLY inside the figure inflated by `margin` are absorbed, so
    body-paragraph lines (which extend well beyond the figure) are ignored."""
    grown = fig + (-margin, -margin, margin, margin)
    box = fitz.Rect(fig)
    for w in words:
        wb = fitz.Rect(w[:4])
        if wb in grown:
            box |= wb
    return box


def cluster(rects, tol=16):
    """Union-find clustering of rects whose inflated boxes overlap.
    Spatial-grid binning keeps it near-linear even with thousands of
    tiny stroke fragments."""
    parent = list(range(len(rects)))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    cell = max(tol, 4)
    grid = defaultdict(list)
    for i, r in enumerate(rects):
        gx0, gx1 = int((r.x0 - tol) // cell), int((r.x1 + tol) // cell)
        gy0, gy1 = int((r.y0 - tol) // cell), int((r.y1 + tol) // cell)
        for gx in range(gx0, gx1 + 1):
            for gy in range(gy0, gy1 + 1):
                for j in grid[(gx, gy)]:
                    ra, rb = find(i), find(j)
                    if ra != rb:
                        parent[ra] = rb
                grid[(gx, gy)].append(i)

    groups = defaultdict(list)
    for i in range(len(rects)):
        groups[find(i)].append(rects[i])

    out = []
    for g in groups.values():
        b = fitz.Rect(g[0])
        for r in g[1:]:
            b |= r
        out.append(b)
    return out


def extract(pdf_path, out_dir="figures", dpi=300, tol=16, pad=8):
    os.makedirs(out_dir, exist_ok=True)
    doc = fitz.open(pdf_path)
    zoom = dpi / 72
    n = 0

    for i, page in enumerate(doc):
        # 1) genuine embedded raster images, if any
        for img in page.get_images(full=True):
            xref = img[0]
            pix = fitz.Pixmap(doc, xref)
            if pix.n - pix.alpha > 3:        # CMYK -> RGB
                pix = fitz.Pixmap(fitz.csRGB, pix)
            pix.save(os.path.join(out_dir, f"p{i+1}_raster_{xref}.png"))
            n += 1

        # 2) vector figures: drop bridges, cluster the rest, render each
        words = page.get_text("words")
        rects = [d["rect"] for d in page.get_drawings() if not is_bridge(d["rect"])]
        for c in cluster(rects, tol=tol):
            if not is_figure(c):
                continue
            c = absorb_labels(c, words)        # pull in clipped text labels
            clip = (c + (-pad, -pad, pad, pad)) & page.rect
            pix = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom), clip=clip)
            pix.save(os.path.join(out_dir, f"p{i+1}_y{int(c.y0):03d}.png"))
            n += 1

    print(f"Extracted {n} figure(s) to {out_dir}/")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python extract_figures.py input.pdf [output_dir]")
    extract(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "figures")
