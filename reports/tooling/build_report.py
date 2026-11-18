"""Build the PathSnap final technical report (PDF + Markdown + DOCX layout JSON) from recorded evidence.

  reports/tooling/.venv/bin/python reports/tooling/build_report.py     (repo root)
Every number is read from a file under experiments/ or reports/data/; nothing is typed in.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import matplotlib
import matplotlib.image as mpimg
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
E, FE, ML, FIN = ROOT / "experiments", ROOT / "experiments/final_evaluation", ROOT / "experiments/ml", ROOT / "experiments/final"
R, FIG, DATA = ROOT / "reports", ROOT / "reports/figures", ROOT / "reports/data"
NICE = {"logreg": "Logistic Regression", "decision_tree": "Decision Tree", "random_forest": "Random Forest",
        "extra_trees": "Extra Trees", "hist_gb": "HistGradientBoosting"}
ORDER = ["logreg", "decision_tree", "random_forest", "extra_trees", "hist_gb"]
FAMILY = {"logreg": "Linear / probabilistic", "decision_tree": "Tree-based", "random_forest": "Ensemble (bagging)",
          "extra_trees": "Ensemble (randomised trees)", "hist_gb": "Boosting"}


def f4(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.4f}"


def f3(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


# ============================================================== load evidence
cv = pd.read_csv(ML / "cv_model_comparison.csv").set_index("model")
cvf = pd.read_csv(ML / "cv_fold_metrics.csv")
st = pd.read_csv(ML / "student_test_model_report.csv").set_index("model")
ohmm = pd.read_csv(ML / "oof_hmm_comparison.csv")
egrid = pd.read_csv(ML / "emission_grid_random_forest.csv")
dec = pd.read_csv(E / "optuna_trials.csv")
rft = pd.read_csv(ML / "rf_optuna_trials_partial.csv")
ab = pd.read_csv(FIN / "ablation_summary.csv")
tvc = pd.read_csv(FIN / "train_vs_cv_metrics.csv").set_index("model")
cmx = pd.read_csv(FIN / "confusion_matrices_oof.csv").set_index("model")
summ = json.load(open(FE / "evaluation_summary.json"))
man = json.load(open(FE / "evaluation_manifest.json"))
ev = pd.read_csv(FE / "evaluation_per_trip.csv", dtype={"trip_id": str})
err = pd.read_csv(FE / "route_error_analysis.csv", dtype={"trip_id": str})
comp = pd.read_csv(FE / "comparison_with_previous_splits.csv")
pf = json.load(open(FE / "post_freeze_checks/post_freeze_check.json"))
dov = pd.read_csv(DATA / "dataset_overview_summary.csv").set_index("split")
dpt = pd.read_csv(DATA / "dataset_overview_per_trip.csv", dtype={"trip_id": str})
conn = pd.read_csv(DATA / "network_connectivity_summary.csv").set_index("split")
tds = json.load(open(DATA / "training_dataset_stats.json"))
tests = json.load(open(DATA / "test_run.json"))
rcon = pd.read_csv(DATA / "reference_route_continuity.csv", dtype={"trip_id": str})
api = json.load(open(DATA / "api_example.json"))
paired = pd.read_csv(DATA / "paired_tests.csv").set_index("comparison")
dtu = pd.read_csv(DATA / "decoder_tuned_vs_untuned.csv")
mapchk = pd.read_csv(DATA / "route_map_f1_check.csv", dtype={"trip_id": str})
alq = pd.read_csv(ML / "alignment_quality.csv", dtype={"trip_id": str})
lab = pd.read_csv(FIN / "label_verification.csv", dtype={"trip_id": str})
part = json.load(open(E / "train_partition.json"))
cfg = json.load(open(ROOT / "configs/ml_v3.json"))
cfgv1 = json.load(open(ROOT / "configs/baseline_v1.json"))
freeze = {l.split("==")[0].lower(): l.split("==")[1] for l in open(FIN / "pip_freeze.txt").read().split() if "==" in l}
fit_tvc = pd.read_csv(FIN / "train_vs_cv_metrics.csv")


def ab_get(es, var, model, col="arc_f1"):
    return float(ab[(ab.eval_set == es) & (ab.variant == var) & (ab.model == model)][col].iloc[0])


def vstat(v, k, s="mean"):
    return summ["variants"][v][k][s]


dbest = dec.loc[dec.value.idxmax()]
dtrial0 = float(dec.loc[dec.number == 0, "value"].iloc[0])
rfc = rft[rft.state == "COMPLETE"]
rbest = rfc.loc[rfc.value.idxmin()]
r0 = float(rft.loc[rft.number == 0, "value"].iloc[0])
geo_oof = ohmm[ohmm.model.str.startswith("tuned_v2")].iloc[0]
oof_by = ohmm.set_index("model")
eg = egrid.set_index(["emission", "ml_weight"])
dep, geo = "rf80_hmm_deployed", "geometric_hmm_tuned_v2"
dvg = summ["deployed_vs_geometric_per_trip"]
e_dep = ev[ev.variant == dep].set_index("trip_id")
e_geo = ev[ev.variant == geo].set_index("trip_id")
er_dep = err[err.variant == dep]
er_geo = err[err.variant == geo]
make_models_src = (ROOT / "src/ml_train.py").read_text()

# ============================================================== content model
B: list[dict] = []
FIGN = [0]
TABN = [0]


def h1(t): B.append({"t": "h1", "text": t})
def h2(t): B.append({"t": "h2", "text": t})
def h3(t): B.append({"t": "h3", "text": t})
def p(t): B.append({"t": "p", "text": t})
def bl(items): B.append({"t": "bullets", "items": items})
def code(t, width=96):
    out = []
    for line in t.split("\n"):
        while len(line) > width:
            cut = line.rfind(", ", 0, width)
            cut = width if cut < 20 else cut + 2
            out.append(line[:cut]); line = "      " + line[cut:]
        out.append(line)
    B.append({"t": "code", "text": "\n".join(out)})
def pb(): B.append({"t": "pagebreak"})


def note(t): B.append({"t": "note", "text": t})


def fig(path, caption, width=1.0):
    FIGN[0] += 1
    pth = Path(path)
    img = mpimg.imread(pth)
    B.append({"t": "fig", "path": str(pth), "caption": f"Figure {FIGN[0]}. {caption}", "width": width,
              "px": [int(img.shape[1]), int(img.shape[0])]})
    return FIGN[0]


def tab(header, rows, caption, widths=None, small=False):
    TABN[0] += 1
    B.append({"t": "table", "header": header, "rows": [[str(c) for c in r] for r in rows],
              "caption": f"Table {TABN[0]}. {caption}", "widths": widths, "small": small})
    return TABN[0]


# ---- content (v2): see reports/tooling/report_content.py
exec(compile(open(R / "tooling" / "report_content.py").read(), "report_content.py", "exec"))
def _sub(x):
    if isinstance(x, str):
        for k_, v_ in TOK.items():
            x = x.replace(k_, v_)
        return x
    if isinstance(x, list):
        return [_sub(i) for i in x]
    if isinstance(x, dict):
        return {k_: _sub(v_) for k_, v_ in x.items()}
    return x
B[:] = [_sub(b) for b in B]
json.dump(B, open(R / "tooling" / "report_blocks.json", "w"), indent=1, ensure_ascii=False)
print(f"blocks: {len(B)}, figures: {FIGN[0]}, tables: {TABN[0]}")

# ============================================================== Markdown
def md_inline(s):
    return s


md = []
for b in B:
    t = b["t"]
    if t == "cover":
        md += [f"# {b['title']}", "", f"**{b['subtitle']}**", ""] + [f"{l}  " for l in b["lines"]] + [""]
        md += [f"**Team {b['team']['Team']}**", "", "| SRN | Name |", "|---|---|"]
        md += [f"| {a} | {n} |" for a, n in b["team"]["members"]] + [""] + [f"{l}  " for l in b["foot"]] + [""]
    elif t == "toc":
        md += ["*Table of contents: see the PDF/DOCX versions.*", ""]
    elif t in ("h1", "h2", "h3"):
        md += ["#" * {"h1": 2, "h2": 3, "h3": 4}[t] + " " + b["text"], ""]
    elif t == "p":
        md += [b["text"], ""]
    elif t == "note":
        md += ["> " + b["text"], ""]
    elif t == "eq":
        md += ["", "    " + b["text"], ""]
    elif t == "bullets":
        md += [f"- {i}" for i in b["items"]] + [""]
    elif t == "code":
        md += ["```", b["text"], "```", ""]
    elif t == "refs":
        md += [f"{i + 1}. {r}" for i, r in enumerate(b["items"])] + [""]
    elif t == "table":
        md += ["| " + " | ".join(b["header"]) + " |", "|" + "---|" * len(b["header"])]
        md += ["| " + " | ".join(c.replace("|", "\\|") for c in r) + " |" for r in b["rows"]]
        md += ["", f"*{b['caption']}*", ""]
    elif t == "figrow":
        for pth, lb in zip(b["paths"], b["labels"]):
            md += [f"![{lb}]({Path(pth).relative_to(R)})", "", f"*{lb}*", ""]
        md += [f"*{b['caption']}*", ""]
    elif t == "fig":
        rel = Path(b["path"]).relative_to(R)
        md += [f"![{b['caption']}]({rel})", "", f"*{b['caption']}*", ""]
(R / "PathSnap_Final_Technical_Report.md").write_text("\n".join(md))

# ============================================================== PDF (reportlab)
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, PageBreak, PageTemplate, Paragraph,
                                Preformatted, Spacer, Table, TableStyle, ListFlowable, ListItem, NextPageTemplate)
from reportlab.platypus.tableofcontents import TableOfContents

fdir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(fdir / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(fdir / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(fdir / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", str(fdir / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(fdir / "DejaVuSansMono.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

ACC = colors.HexColor("#2a78d6")
INK = colors.HexColor("#1b1b1a")
INK2 = colors.HexColor("#52514e")
LINE = colors.HexColor("#d9d8d3")
ss = getSampleStyleSheet()
S = {
    "body": ParagraphStyle("body", fontName="DV", fontSize=9.4, leading=13.2, textColor=INK, spaceAfter=6, alignment=4),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=15, leading=19, textColor=ACC, spaceBefore=14, spaceAfter=8, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=11.5, leading=15, textColor=INK, spaceBefore=9, spaceAfter=5, keepWithNext=1),
    "h3": ParagraphStyle("h3", fontName="DV-B", fontSize=10, leading=13, textColor=INK, spaceBefore=6, spaceAfter=4, keepWithNext=1),
    "cap": ParagraphStyle("cap", fontName="DV-I", fontSize=8, leading=10.5, textColor=INK2, spaceBefore=3, spaceAfter=10),
    "cell": ParagraphStyle("cell", fontName="DV", fontSize=7.8, leading=9.6, textColor=INK),
    "cells": ParagraphStyle("cells", fontName="DV", fontSize=6.9, leading=8.4, textColor=INK),
    "head": ParagraphStyle("head", fontName="DV-B", fontSize=7.8, leading=9.6, textColor=colors.white),
    "heads": ParagraphStyle("heads", fontName="DV-B", fontSize=6.9, leading=8.4, textColor=colors.white),
    "code": ParagraphStyle("code", fontName="DVM", fontSize=7.0, leading=8.9, textColor=INK, backColor=colors.HexColor("#f3f2ee"),
                           borderPadding=5, spaceBefore=4, spaceAfter=8),
    "note": ParagraphStyle("note", fontName="DV", fontSize=8.8, leading=12, textColor=INK, backColor=colors.HexColor("#fdf1ea"),
                           borderColor=colors.HexColor("#eb6834"), borderWidth=0.8, borderPadding=6, spaceBefore=6, spaceAfter=10),
    "ref": ParagraphStyle("ref", fontName="DV", fontSize=8.6, leading=11.5, textColor=INK, leftIndent=14, firstLineIndent=-14, spaceAfter=4),
    "bul": ParagraphStyle("bul", fontName="DV", fontSize=9.2, leading=12.8, textColor=INK, alignment=0),
}


def rl(s):
    s = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"`(.+?)`", r"<font name='DVM'>\1</font>", s)
    return s


class Doc(BaseDocTemplate):
    def afterFlowable(self, fl):
        if isinstance(fl, Paragraph) and getattr(fl, "_toc", None):
            lvl, txt = fl._toc
            key = f"h{self.seq.nextf('h')}"
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(txt, key, level=lvl, closed=lvl > 0)
            self.notify("TOCEntry", (lvl, txt, self.page, key))


W, H = A4
LM = RM = 2.0 * cm


def on_page(c, d):
    c.saveState()
    c.setFont("DV", 7.5); c.setFillColor(INK2)
    c.drawString(LM, H - 1.25 * cm, "PathSnap — Final Technical Report")
    c.drawRightString(W - RM, H - 1.25 * cm, "UE24CS352A Machine Learning")
    c.setStrokeColor(LINE); c.line(LM, H - 1.4 * cm, W - RM, H - 1.4 * cm)
    c.drawCentredString(W / 2, 1.1 * cm, f"Page {d.page}")
    c.restoreState()


doc = Doc(str(R / "PathSnap_Final_Technical_Report.pdf"), pagesize=A4, leftMargin=LM, rightMargin=RM,
          topMargin=2.0 * cm, bottomMargin=1.8 * cm, title="PathSnap: ML-Assisted GPS Map Matching and Route Reconstruction",
          author="PathSnap team", subject="Final technical report")
frame = Frame(LM, 1.8 * cm, W - LM - RM, H - 3.8 * cm, id="f")
doc.addPageTemplates([PageTemplate("cover", frames=[frame]), PageTemplate("main", frames=[frame], onPage=on_page)])
TW = W - LM - RM
story = []
for b in B:
    t = b["t"]
    if t == "cover":
        story += [Spacer(1, 3.6 * cm),
                  Paragraph(rl(b["title"]), ParagraphStyle("t", fontName="DV-B", fontSize=24, leading=30, textColor=ACC, alignment=TA_CENTER)),
                  Spacer(1, 0.5 * cm),
                  Paragraph(rl(b["subtitle"]), ParagraphStyle("st", fontName="DV", fontSize=14, leading=18, textColor=INK, alignment=TA_CENTER)),
                  Spacer(1, 1.4 * cm)]
        for l in b["lines"]:
            story.append(Paragraph(rl(l), ParagraphStyle("cl", fontName="DV", fontSize=10, leading=14, textColor=INK2,
                                                         alignment=TA_CENTER, spaceAfter=4)))
        story.append(Spacer(1, 1.2 * cm))
        story.append(Paragraph(f"Team {b['team']['Team']}", ParagraphStyle("tm", fontName="DV-B", fontSize=13, leading=17,
                                                                        textColor=INK, alignment=TA_CENTER, spaceAfter=8)))
        tt = Table([[Paragraph("<b>SRN</b>", S["cell"]), Paragraph("<b>Name</b>", S["cell"])]] +
                   [[Paragraph(a, ParagraphStyle("c1", fontName="DV", fontSize=10, leading=13)),
                     Paragraph(n, ParagraphStyle("c2", fontName="DV", fontSize=10, leading=13))] for a, n in b["team"]["members"]],
                   colWidths=[4.6 * cm, 5.6 * cm])
        tt.setStyle(TableStyle([("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("BOX", (0, 0), (-1, -1), 0.6, LINE),
                                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3fb")),
                                ("TOPPADDING", (0, 0), (-1, -1), 5), ("BOTTOMPADDING", (0, 0), (-1, -1), 5)]))
        story += [tt, Spacer(1, 2.2 * cm)]
        for l in b["foot"]:
            story.append(Paragraph(rl(l), ParagraphStyle("cf", fontName="DV", fontSize=8.5, leading=12, textColor=INK2,
                                                         alignment=TA_CENTER)))
        story += [NextPageTemplate("main"), PageBreak()]
    elif t == "toc":
        toc = TableOfContents()
        toc.levelStyles = [ParagraphStyle("t0", fontName="DV-B", fontSize=9.2, leading=11.2, leftIndent=0, spaceBefore=1.5, spaceAfter=0),
                           ParagraphStyle("t1", fontName="DV", fontSize=8.2, leading=9.8, leftIndent=14, spaceBefore=0, spaceAfter=0)]
        toc.dotsMinLevel = 0
        story += [Paragraph("Contents", S["h1"]), toc, PageBreak()]
    elif t in ("h1", "h2", "h3"):
        para = Paragraph(rl(b["text"]), S[t])
        para._toc = (0 if t == "h1" else 1, b["text"]) if t != "h3" else None
        story.append(para)
    elif t == "p":
        story.append(Paragraph(rl(b["text"]), S["body"]))
    elif t == "note":
        story.append(KeepTogether([Paragraph(rl(b["text"]), S["note"])]))
    elif t == "eq":
        story.append(Paragraph(rl(b["text"]), ParagraphStyle("eq", fontName="DV-I", fontSize=9.6, leading=13, textColor=INK,
                                                              alignment=TA_CENTER, spaceBefore=3, spaceAfter=7)))
    elif t == "bullets":
        def _lf(items, after):
            return ListFlowable([ListItem(Paragraph(rl(i), S["bul"]), leftIndent=12, value="•") for i in items],
                                bulletType="bullet", start="•", leftIndent=12, bulletFontName="DV", bulletFontSize=8,
                                spaceAfter=after)
        if story and isinstance(story[-1], Paragraph) and getattr(story[-1], "_toc", None) is not None and len(b["items"]) > 1:
            # keep the heading with the first item only, so long lists can flow across pages
            story.append(KeepTogether([story.pop(), _lf(b["items"][:1], 0)]))
            story.append(_lf(b["items"][1:], 6))
        elif story and isinstance(story[-1], Paragraph) and getattr(story[-1], "_toc", None) is not None:
            story.append(KeepTogether([story.pop(), _lf(b["items"], 6)]))
        else:
            story.append(_lf(b["items"], 6))
        story.append(Spacer(1, 4))
    elif t == "code":
        pre = Preformatted(b["text"], S["code"])
        if story and isinstance(story[-1], Paragraph) and getattr(story[-1], "_toc", None) is not None:
            story.append(KeepTogether([story.pop(), pre]))
        else:
            story.append(pre)
    elif t == "refs":
        for i, r in enumerate(b["items"]):
            story.append(Paragraph(f"[{i + 1}]&nbsp;&nbsp;" + rl(r), S["ref"]))
    elif t == "table":
        sm = b["small"]
        cs, hs = (S["cells"], S["heads"]) if sm else (S["cell"], S["head"])
        data = [[Paragraph(rl(h), hs) for h in b["header"]]] + [[Paragraph(rl(c), cs) for c in r] for r in b["rows"]]
        n = len(b["header"])
        fs = 6.9 if sm else 7.8
        def tw(txt, bold=False):
            return pdfmetrics.stringWidth(txt, "DV-B" if bold else "DV", fs)
        allrows = [b["header"]] + b["rows"]
        mins = np.array([max(max((tw(tok, j == 0) for tok in str(r[i]).split() or [""]), default=0)
                             for j, r in enumerate(allrows)) + 9 for i in range(n)])
        want = np.array([min(max(tw(str(r[i]), j == 0) for j, r in enumerate(allrows)) + 9, 200) for i in range(n)])
        if b["widths"]:
            want = np.maximum(np.array(b["widths"]) * TW, mins)
        want = np.maximum(want, mins)
        if want.sum() <= TW:
            cw = want * TW / want.sum()
        else:
            extra = TW - mins.sum()
            cw = mins + (want - mins) / max((want - mins).sum(), 1e-9) * max(extra, 0) if extra > 0 else mins * TW / mins.sum()
        cw = list(cw)
        tb = Table(data, colWidths=cw, repeatRows=1)
        tb.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), ACC), ("VALIGN", (0, 0), (-1, -1), "TOP"),
                                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f5f5f2")]),
                                ("LINEBELOW", (0, 0), (-1, -1), 0.3, LINE), ("BOX", (0, 0), (-1, -1), 0.5, LINE),
                                ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                                ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]))
        cap = Paragraph(rl(b["caption"]), S["cap"])
        grp = [tb, cap]
        if story and isinstance(story[-1], Paragraph) and getattr(story[-1], "_toc", None) is not None:
            grp.insert(0, story.pop())
        if len(b["rows"]) <= 14:
            story.append(KeepTogether(grp))
        else:
            story += grp
    elif t == "fig":
        wpx, hpx = b["px"]
        wmax = TW * b["width"]
        hmax = H - 6.5 * cm
        w_ = wmax
        h_ = w_ * hpx / wpx
        if h_ > hmax:
            h_ = hmax; w_ = h_ * wpx / hpx
        grp = [Image(b["path"], width=w_, height=h_), Paragraph(rl(b["caption"]), S["cap"])]
        if story and isinstance(story[-1], Paragraph) and getattr(story[-1], "_toc", None) is not None:
            grp.insert(0, story.pop())
        story.append(KeepTogether(grp))
    elif t == "figrow":
        n_ = len(b["paths"]); gap = 0.3 * cm
        cw_ = (TW - gap * (n_ - 1)) / n_
        cells, labs = [], []
        for pth, lb in zip(b["paths"], b["labels"]):
            im_ = mpimg.imread(pth); hh = cw_ * im_.shape[0] / im_.shape[1]
            cells.append(Image(pth, width=cw_, height=hh))
            labs.append(Paragraph(rl(lb), ParagraphStyle("fl", fontName="DV", fontSize=7.6, leading=9.5, textColor=INK2)))
        tb_ = Table([cells, labs], colWidths=[cw_ + (gap if i < n_ - 1 else 0) for i in range(n_)])
        tb_.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "TOP"), ("LEFTPADDING", (0, 0), (-1, -1), 0),
                                 ("RIGHTPADDING", (0, 0), (-1, -1), 0), ("TOPPADDING", (0, 1), (-1, 1), 3)]))
        grp = [tb_, Paragraph(rl(b["caption"]), S["cap"])]
        if story and isinstance(story[-1], Paragraph) and getattr(story[-1], "_toc", None) is not None:
            grp.insert(0, story.pop())
        story.append(KeepTogether(grp))
    elif t == "pagebreak":
        story.append(PageBreak())
    elif t == "pagebreak_if_low":
        from reportlab.platypus.doctemplate import CondPageBreak
        story.append(CondPageBreak(b["min_cm"] * cm))
doc.multiBuild(story)
print("PDF written")
