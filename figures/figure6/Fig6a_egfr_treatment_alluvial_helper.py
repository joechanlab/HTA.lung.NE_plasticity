"""
Helper for the EGFR status x prior-treatment alluvial plot (T-SCLC patients).

Three stages:
1. EGFR status (Mutant vs Wild-type) -- from figures/figure7/Supplementary_Table_3
   FINAL.xlsx ("Supp Table 3"), via the Cohort field (EGFR-mut / EGFR-wt T-SCLC),
   restricted to those two cohorts.
2. Prior treatment status (Treated vs Untreated) -- from metadata_for_limma.csv
   (per-sample boolean chemo/IO/TKI flags), "ever treated" across all of a
   patient's samples: "Treated" if any sample has chemo, IO, or TKI == True,
   else "Untreated". Every sample has a definitive True/False for all three
   flags, so there is no "Unknown" category.
3. Treatment type (TKI / Chemo / IO) -- for Treated patients only. A patient
   can appear under more than one type (e.g. someone who got both TKI and
   chemo counts fully in both bars), so these three bars are an *overlapping*
   breakdown, not a partition: their counts do not have to sum to the Treated
   total. Untreated patients are not branched further.
"""
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import numpy as np
import pandas as pd

SUPP_TABLE_PATH = "../figure7/Supplementary_Table_3 FINAL.xlsx"
LIMMA_METADATA_PATH = "/data1/chanj3/HTA.lung.NE_plasticity.120122/ref/metadata_for_limma.csv"

TX_COLS = ["chemo", "IO", "TKI"]
TX_LABELS = {"chemo": "Chemo", "IO": "IO", "TKI": "TKI"}
TX_TYPE_ORDER = ["TKI", "Chemo", "IO"]

EGFR_COLORS = {"Mutant": "#2a78d6", "Wild-type": "#eb6834"}
TX_COLORS = {"TKI": "#1baf7a", "Chemo": "#eda100", "IO": "#4a3aa7"}
STAGE2_ORDER = ["Treated", "Untreated"]
NODE_COLOR = "#c3c2b7"  # muted ink, Chart chrome & ink / Baseline-axis role
GAP = 0.03  # fractional gap between stacked node segments


def load_patient_table(supp_table_path=SUPP_TABLE_PATH, limma_metadata_path=LIMMA_METADATA_PATH):
    df = pd.read_excel(supp_table_path, sheet_name="Supp Table 3")
    tsclc = df[df["Cohort"].isin(["EGFR-mut T-SCLC", "EGFR-wt T-SCLC"])].copy()
    egfr_by_patient = (
        tsclc[["Patient ID", "Cohort"]]
        .drop_duplicates()
        .assign(**{
            "EGFR status": lambda d: np.where(
                d["Cohort"] == "EGFR-mut T-SCLC", "Mutant", "Wild-type"
            )
        })
        .set_index("Patient ID")["EGFR status"]
    )

    meta = pd.read_csv(limma_metadata_path)
    meta = meta[meta["patient"].isin(egfr_by_patient.index)]
    ever = meta.groupby("patient")[TX_COLS].any()
    ever.columns = [TX_LABELS[c] for c in ever.columns]
    treated_by_patient = ever.any(axis=1).map({True: "Treated", False: "Untreated"})

    patient_df = pd.concat(
        [
            egfr_by_patient.rename("EGFR status"),
            treated_by_patient.rename("Prior treatment status"),
            ever,
        ],
        axis=1,
    ).dropna(subset=["EGFR status"]).reset_index(names="Patient ID")
    return patient_df


def _stack_nodes(counts, gap=GAP):
    """counts: dict label -> n, in display order (top to bottom). Returns dict
    label -> (y0, y1) spanning [0, 1] with `gap` fractional whitespace between."""
    total = sum(counts.values())
    n = len(counts)
    usable = 1 - gap * (n - 1)
    spans = {}
    y = 1.0
    for label, c in counts.items():
        h = usable * c / total
        spans[label] = (y - h, y)
        y = y - h - gap
    return spans


def _flow_path(x0, y0_lo, y0_hi, x1, y1_lo, y1_hi):
    """Smooth ribbon between a vertical segment at x0 and one at x1."""
    xm = (x0 + x1) / 2
    verts = [
        (x0, y0_lo), (xm, y0_lo), (xm, y1_lo), (x1, y1_lo),
        (x1, y1_hi), (xm, y1_hi), (xm, y0_hi), (x0, y0_hi),
        (x0, y0_lo),
    ]
    codes = [
        mpath.Path.MOVETO, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
        mpath.Path.LINETO, mpath.Path.CURVE4, mpath.Path.CURVE4, mpath.Path.CURVE4,
        mpath.Path.CLOSEPOLY,
    ]
    return mpath.Path(verts, codes)


def _draw_flow(ax, x_a, a_lo, a_hi, x_b, b_lo, b_hi, color):
    path = _flow_path(x_a, a_lo, a_hi, x_b, b_lo, b_hi)
    ax.add_patch(mpatches.PathPatch(path, facecolor=color, edgecolor="none", alpha=0.55, zorder=1))


def _draw_node(ax, x, y0, y1, node_width, color, label, ha, fontsize=9):
    rect = mpatches.FancyBboxPatch(
        (x, y0), node_width, y1 - y0,
        boxstyle="round,pad=0,rounding_size=0.006",
        facecolor=color, edgecolor="white", linewidth=0.6, zorder=2,
    )
    ax.add_patch(rect)
    if ha == "right":
        text_x, va, ty = x - 0.02, "center", (y0 + y1) / 2
    elif ha == "left":
        text_x, va, ty = x + node_width + 0.015, "center", (y0 + y1) / 2
    else:
        text_x, va, ty = x + node_width / 2, ha, y0
    ax.text(text_x, ty, label, ha="center" if ha not in ("left", "right") else ha,
             va=va, fontsize=fontsize, color="#0b0b0b")


def draw_alluvial(ax, patient_df, node_width=0.045, x0=0.14, x1=0.48, x2=0.86):
    stage1_order = [c for c in ["Mutant", "Wild-type"] if c in patient_df["EGFR status"].unique()]
    stage2_order = [c for c in STAGE2_ORDER if c in patient_df["Prior treatment status"].unique()]

    # --- Stage 1 -> Stage 2: EGFR status -> Treated/Untreated (conservative partition) ---
    counts1 = {k: int((patient_df["EGFR status"] == k).sum()) for k in stage1_order}
    counts2 = {k: int((patient_df["Prior treatment status"] == k).sum()) for k in stage2_order}
    spans1 = _stack_nodes(counts1)
    spans2 = _stack_nodes(counts2)

    # Precompute the Treated node's treatment-pattern grouping (and its vertical
    # slots) up front, before any Stage1->Stage2 flow is drawn. Stage2->Stage3
    # carves the Treated node into these same pattern-group slots, so the
    # Stage1->Stage2 ribbon has to land on the identical slots (subdivided by
    # EGFR status within each one) or the two columns disagree about which
    # patients sit where inside "Treated".
    group_sizes = {}
    group_order = []
    pattern_slots = {}
    if "Treated" in spans2:
        treated_sub = patient_df[patient_df["Prior treatment status"] == "Treated"].copy()
        treated_sub["_pattern"] = treated_sub.apply(
            lambda r: tuple(t for t in TX_TYPE_ORDER if r[t]), axis=1
        )
        group_sizes = treated_sub.groupby("_pattern").size().to_dict()
        group_order = sorted(
            group_sizes, key=lambda p: (len(p), [TX_TYPE_ORDER.index(t) for t in p])
        )
        treated_y0, treated_y1 = spans2["Treated"]
        treated_height = treated_y1 - treated_y0
        n_treated = len(treated_sub)
        cur = treated_y1
        for pattern in group_order:
            group_h = (group_sizes[pattern] / n_treated) * treated_height
            pattern_slots[pattern] = (cur, cur - group_h)
            cur -= group_h

    cursor1 = {k: spans1[k][1] for k in spans1}

    # --- Stage 1 -> Stage 2: Treated, sliced by treatment-pattern group (same
    # group_order/pattern_slots that Stage 2 -> Stage 3 uses below), and within
    # each group split by EGFR status. This keeps a given patient's vertical
    # position inside the Treated node consistent across both ribbon columns.
    # Drawn first so it claims the top portion of each EGFR node's cursor --
    # Treated sits above Untreated in stage 2, so its ribbons should come off
    # the top of Mutant/Wild-type too, leaving Untreated's ribbons the bottom
    # portion below them instead of crossing over them. ---
    if "Treated" in spans2:
        for pattern in group_order:
            slot_top, slot_bot = pattern_slots[pattern]
            slot_h = slot_top - slot_bot
            group_patients = treated_sub[treated_sub["_pattern"] == pattern]
            group_n = len(group_patients)
            sub_cursor = slot_top
            for label1 in stage1_order:
                n = int((group_patients["EGFR status"] == label1).sum())
                if n == 0:
                    continue
                a_hi = cursor1[label1]
                a_lo = a_hi - (n / counts1[label1]) * (spans1[label1][1] - spans1[label1][0])
                cursor1[label1] = a_lo
                b_hi = sub_cursor
                b_lo = b_hi - (n / group_n) * slot_h
                sub_cursor = b_lo
                _draw_flow(ax, x0 + node_width, a_lo, a_hi, x1 - node_width / 2, b_lo, b_hi, EGFR_COLORS[label1])

    # --- Stage 1 -> Stage 2: Untreated (simple per-EGFR-status block; Untreated
    # has no Stage 3, so there's no ordering downstream to match). Takes what's
    # left of cursor1 below the Treated slice drawn above. ---
    if "Untreated" in spans2:
        cursor_u = spans2["Untreated"][1]
        for label1 in stage1_order:
            n = int((
                (patient_df["EGFR status"] == label1)
                & (patient_df["Prior treatment status"] == "Untreated")
            ).sum())
            if n == 0:
                continue
            a_hi = cursor1[label1]
            a_lo = a_hi - (n / counts1[label1]) * (spans1[label1][1] - spans1[label1][0])
            cursor1[label1] = a_lo
            b_hi = cursor_u
            b_lo = b_hi - (n / counts2["Untreated"]) * (spans2["Untreated"][1] - spans2["Untreated"][0])
            cursor_u = b_lo
            _draw_flow(ax, x0 + node_width, a_lo, a_hi, x1 - node_width / 2, b_lo, b_hi, EGFR_COLORS[label1])

    for label, (y0, y1) in spans1.items():
        _draw_node(ax, x0, y0, y1, node_width, EGFR_COLORS[label], f"EGFR\n{label}\n(n={counts1[label]})", "right", fontsize=10)

    stage2_labels = list(spans2.keys())
    for i, (label, (y0, y1)) in enumerate(spans2.items()):
        rect = mpatches.FancyBboxPatch(
            (x1 - node_width / 2, y0), node_width, y1 - y0,
            boxstyle="round,pad=0,rounding_size=0.006",
            facecolor=NODE_COLOR, edgecolor="white", linewidth=0.6, zorder=2,
        )
        ax.add_patch(rect)
        if i == 0:
            ax.text(x1, y1 + 0.015, f"{label} (n={counts2[label]})", ha="center", va="bottom", fontsize=9, color="#0b0b0b")
        elif i == len(stage2_labels) - 1:
            ax.text(x1, y0 - 0.015, f"{label} (n={counts2[label]})", ha="center", va="top", fontsize=9, color="#0b0b0b")
        else:
            ax.text(x1, (y0 + y1) / 2, f"{label}\n(n={counts2[label]})", ha="center", va="center", fontsize=8, color="#0b0b0b")

    # --- Stage 2 -> Stage 3: Treated patients only, fan into TKI / Chemo / IO. ---
    # Patients are grouped by their treatment pattern (same combination of
    # TKI/Chemo/IO together, singles before combinations) and drawn as one block
    # per pattern -- not one sliver per patient -- so adjacent flows share a
    # common source edge instead of each patient carving its own tiny notch
    # (which produced a sawtooth "teeth" artifact along the fan-out). A pattern
    # with more than one modality sends one ribbon per modality, and those
    # ribbons all start from the same block (i.e. they overlap at the source,
    # alpha-blended) rather than being tiled side by side, since those patients
    # are on all of them at once, not split between them. Ribbons are colored by
    # treatment type. This is an overlapping breakdown, not a partition -- bar
    # counts need not sum to the Treated total. Untreated is a dead end -- no
    # outgoing flow to stage 3.
    if "Treated" in spans2:
        type_counts = {t: int(treated_sub[t].sum()) for t in TX_TYPE_ORDER if treated_sub[t].sum() > 0}
        right_spans = _stack_nodes(type_counts)
        right_cursor = {t: right_spans[t][1] for t in right_spans}

        # group_sizes/group_order/pattern_slots computed above, shared with the
        # Stage1->Stage2 Treated flow so the two ribbon columns agree on layout.
        for pattern in group_order:
            count = group_sizes[pattern]
            slot_top, slot_bot = pattern_slots[pattern]
            for t in pattern:
                r_h = (right_spans[t][1] - right_spans[t][0]) * (count / type_counts[t])
                r_top = right_cursor[t]
                r_bot = r_top - r_h
                right_cursor[t] = r_bot
                _draw_flow(
                    ax, x1 + node_width / 2, slot_bot, slot_top,
                    x2 - node_width, r_bot, r_top, TX_COLORS[t],
                )

        for t, (y0, y1) in right_spans.items():
            _draw_node(ax, x2 - node_width, y0, y1, node_width, TX_COLORS[t], f"{t}\n(n={type_counts[t]})", "left")

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.05, 1.05)
    ax.axis("off")
