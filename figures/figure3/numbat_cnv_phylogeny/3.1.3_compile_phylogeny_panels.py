"""
Compile all *_phylogeny.png files from metacell Numbat output folders
into a single multi-panel figure with a color legend in the blank space.

Usage:
    python 3.1.3_compile_phylogeny_panels.py [--mode patient|sample|patient-timepoint]
                                             [--n-cols N]

    --mode patient            (default) one PNG per patient, cell-type colors
    --mode sample             one PNG per sample (grouped by patient/timepoint),
                              cell-type colors
    --mode patient-timepoint  one PNG per multi-timepoint patient, timepoint colors
    --n-cols N                number of columns (default: 6)
"""
import argparse
import math
import os
import csv
from PIL import Image, ImageDraw, ImageFont

BASE      = "/data1/chanj3/HTA.lung.NE_plasticity.120122/Numbat_All"
OUT_DIR   = "/data1/chanj3/wangm10/HTA.lung.NE_plasticity_scDeBussy/figures/figure3/Numbat/figures"
FONT_PATH = "/usr/share/fonts/dejavu/DejaVuSans.ttf"

CELLTYPE_LEGEND = [
    ("SCLC-A",              "#d62728"),
    ("SCLC-N",              "#17becf"),
    ("SCLC-P",              "#1f77b4"),
    ("NonNE SCLC",          "#4B0082"),
    ("EMT",                 "#673147"),
    ("FOXI1+ NonNE SCLC-N", "#2ca02c"),
    ("LUAD",                "#ffd700"),
    ("LUSC",                "#FFA500"),
    ("Other / non-tumor",   "#AAAAAA"),
]

# Timepoint legend: fixed discrete ranks shared across all patients
TIMEPOINT_LEGEND = [
    ("Timepoint 1 (earliest)", "#4575B4"),
    ("Timepoint 2",            "#91BFDB"),
    ("Timepoint 3",            "#FC8D59"),
    ("Timepoint 4 (latest)",   "#D73027"),
    ("Non-tumor",              "#AAAAAA"),
]


def _hex_to_rgb(h):
    return int(h[1:3], 16), int(h[3:5], 16), int(h[5:7], 16)


EXCLUDE   = {"RU1646"}
TITLE_H   = 30   # px reserved above each tree for the patient label


def draw_legend(canvas, legend_entries, title, w, h, n_cols, n_rows, pad, n_images):
    """Draw a legend in the first blank cell of the last row."""
    n_blank = n_cols * n_rows - n_images
    if n_blank == 0:
        return
    first_blank_col = n_images % n_cols
    leg_x0 = pad + first_blank_col * (w + pad)
    leg_y0 = pad + (n_rows - 1) * (h + pad)

    draw = ImageDraw.Draw(canvas)
    try:
        font_label = ImageFont.truetype(FONT_PATH, size=22)
        font_title = ImageFont.truetype(FONT_PATH.replace(".ttf", "-Bold.ttf"), size=24)
    except Exception:
        font_label = ImageFont.load_default()
        font_title = font_label

    swatch      = 22
    row_gap     = 34
    left_margin = 20
    title_gap   = 14

    block_h = 28 + title_gap + len(legend_entries) * row_gap
    start_y = leg_y0 + (h - block_h) // 2

    draw.text((leg_x0 + left_margin, start_y), title,
              font=font_title, fill=(40, 40, 40))
    entry_y = start_y + 28 + title_gap

    for label, hex_color in legend_entries:
        r, g, b = _hex_to_rgb(hex_color)
        sx, sy = leg_x0 + left_margin, entry_y
        draw.rectangle([sx, sy, sx + swatch, sy + swatch],
                       fill=(r, g, b), outline=(80, 80, 80), width=1)
        draw.text((sx + swatch + 10, sy + 1), label,
                  font=font_label, fill=(40, 40, 40))
        entry_y += row_gap


def compile(png_paths, out_path, legend_entries, legend_title, n_cols=6, titles=None):
    print(f"Found {len(png_paths)} phylogeny PNGs")
    if not png_paths:
        print("Nothing to compile.")
        return

    images   = [Image.open(p) for p in png_paths]
    w, h     = images[0].size
    cell_h   = h + TITLE_H          # each cell reserves space for the label
    n_images = len(images)
    n_rows   = math.ceil(n_images / n_cols)
    if n_images % n_cols == 0:   # last row full → add row for legend
        n_rows += 1

    pad      = 10
    canvas_w = n_cols * w + (n_cols + 1) * pad
    canvas_h = n_rows * cell_h + (n_rows + 1) * pad
    canvas   = Image.new("RGB", (canvas_w, canvas_h), color=(255, 255, 255))

    try:
        font_panel = ImageFont.truetype(FONT_PATH, size=20)
    except Exception:
        font_panel = ImageFont.load_default()
    draw = ImageDraw.Draw(canvas)

    for idx, img in enumerate(images):
        row, col = idx // n_cols, idx % n_cols
        x0 = pad + col * (w + pad)
        y0 = pad + row * (cell_h + pad)

        if titles and idx < len(titles):
            label = titles[idx]
            bbox  = draw.textbbox((0, 0), label, font=font_panel)
            tx    = x0 + (w - (bbox[2] - bbox[0])) // 2
            ty    = y0 + (TITLE_H - (bbox[3] - bbox[1])) // 2
            draw.text((tx, ty), label, font=font_panel, fill=(40, 40, 40))

        if img.mode in ("RGBA", "LA", "P"):
            white = Image.new("RGBA", img.size, (255, 255, 255, 255))
            white.paste(img.convert("RGBA"), mask=img.convert("RGBA").split()[3])
            img = white.convert("RGB")
        canvas.paste(img, (x0, y0 + TITLE_H))

    draw_legend(canvas, legend_entries, legend_title, w, cell_h,
                n_cols, n_rows, pad, n_images)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path, dpi=(150, 150))
    print(f"Saved: {out_path}  ({canvas_w}×{canvas_h} px, {n_cols} cols × {n_rows} rows)")


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_sample_timepoints(script_dir):
    """Return dict sample_id → {patient_id, date_rank, n_distinct_tp}."""
    tp_file = os.path.join(script_dir, "sample_timepoints.tsv")
    tp = {}
    with open(tp_file) as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            tp[row["sample_id"]] = {
                "patient_id":    row["patient_id"],
                "date_rank":     int(row["date_rank"]),
                "n_distinct_tp": int(row["n_distinct_tp"]),
            }
    return tp


def _collect_patient_pngs(ids, script_dir):
    """Collect patient-level PNGs (with fallback to sample-level for single-sample patients).

    Returns (png_paths, titles) — two parallel lists.
    """
    png_paths, titles = [], []
    for pid in ids:
        patient = pid.split("_")[0]
        if patient in EXCLUDE:
            print(f"  Excluding {pid}")
            continue
        p = os.path.join(BASE, patient,
                         "numbat_output_FACETS_HISENS_metacell",
                         f"{patient}_phylogeny.png")
        if os.path.exists(p):
            png_paths.append(p)
            titles.append(patient)
        elif pid != patient:
            p_sample = os.path.join(BASE, patient,
                                    "numbat_output_FACETS_HISENS_metacell",
                                    pid, f"{pid}_phylogeny.png")
            if os.path.exists(p_sample):
                print(f"  Note: using sample-level PNG for {pid}")
                png_paths.append(p_sample)
                titles.append(pid)
            else:
                print(f"  Warning: PNG not found for {pid}")
        else:
            print(f"  Warning: PNG not found for {pid}")
    return png_paths, titles


# ── main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",
                        choices=["patient", "sample", "patient-timepoint"],
                        default="patient")
    parser.add_argument("--n-cols", type=int, default=6,
                        help="Number of columns in the compiled figure (default: 6)")
    args = parser.parse_args()

    script_dir = os.path.dirname(os.path.abspath(__file__))

    if args.mode == "patient":
        list_file = os.path.join(script_dir, "patient_list.txt")
        out_file  = os.path.join(OUT_DIR, "all_phylogeny_metacell.png")
        with open(list_file) as f:
            ids = [l.strip() for l in f if l.strip()]
        png_paths, titles = _collect_patient_pngs(ids, script_dir)
        compile(png_paths, out_file, CELLTYPE_LEGEND, "Cell type",
                n_cols=args.n_cols, titles=titles)

    elif args.mode == "sample":
        # sample_list.txt is already ordered by patient then timepoint
        list_file = os.path.join(script_dir, "sample_list.txt")
        out_file  = os.path.join(OUT_DIR, "all_phylogeny_metacell_per_sample.png")
        with open(list_file) as f:
            ids = [l.strip() for l in f if l.strip()]
        png_paths, titles = [], []
        for sid in ids:
            patient = sid.split("_")[0]
            if patient in EXCLUDE:
                print(f"  Excluding {sid}")
                continue
            p = os.path.join(BASE, patient,
                             "numbat_output_FACETS_HISENS_metacell",
                             sid, f"{sid}_phylogeny.png")
            if os.path.exists(p):
                png_paths.append(p)
                titles.append(sid)
            else:
                print(f"  Warning: PNG not found for {sid} → {p}")
        compile(png_paths, out_file, CELLTYPE_LEGEND, "Cell type",
                n_cols=args.n_cols, titles=titles)

    elif args.mode == "patient-timepoint":
        list_file = os.path.join(script_dir, "patient_timepoint_list.txt")
        out_file  = os.path.join(OUT_DIR, "all_phylogeny_metacell_timepoint.png")
        with open(list_file) as f:
            ids = [l.strip() for l in f if l.strip()]
        png_paths, titles = [], []
        for patient in ids:
            if patient in EXCLUDE:
                print(f"  Excluding {patient}")
                continue
            p = os.path.join(BASE, patient,
                             "numbat_output_FACETS_HISENS_metacell",
                             f"{patient}_phylogeny_timepoint.png")
            if os.path.exists(p):
                png_paths.append(p)
                titles.append(patient)
            else:
                print(f"  Warning: timepoint PNG not found for {patient} → {p}")
        compile(png_paths, out_file, TIMEPOINT_LEGEND, "Timepoint",
                n_cols=args.n_cols, titles=titles)
