#!/usr/bin/env python3
"""Generate extended XS vector figures and 300 dpi review PNGs.

Inputs are compact exports produced by export_xs_frozen_evidence.py plus the
frozen final VLA CSV/JSON. No model, simulator, video, or rollout code is used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas


W = 7.2 * 72
BLUE = "#0072B2"
ORANGE = "#E69F00"
GREEN = "#009E73"
VERMILLION = "#D55E00"
SKY = "#56B4E9"
GREY = "#B8BDC5"
LIGHT_GREY = "#EEF1F4"
DARK = "#20252B"
MID = "#667085"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def pdf_text(c: canvas.Canvas, x: float, y: float, value: str, size: float = 9,
             *, bold: bool = False, colour: str = DARK, align: str = "left") -> None:
    c.setFillColor(HexColor(colour))
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def pdf_box(c: canvas.Canvas, x: float, y: float, w: float, h: float,
            title: str, lines: Iterable[str], *, fill: str, stroke: str) -> None:
    c.setFillColor(HexColor(fill))
    c.setStrokeColor(HexColor(stroke))
    c.setLineWidth(1.2)
    c.roundRect(x, y, w, h, 7, fill=1, stroke=1)
    pdf_text(c, x + w / 2, y + h - 16, title, 9.2, bold=True, align="center")
    for index, line in enumerate(lines):
        pdf_text(c, x + w / 2, y + h - 31 - 12 * index, line, 8.2, align="center")


def pdf_arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float,
              colour: str = MID) -> None:
    c.setStrokeColor(HexColor(colour))
    c.setFillColor(HexColor(colour))
    c.setLineWidth(1.4)
    c.line(x1, y1, x2, y2)
    direction = 1 if x2 >= x1 else -1
    c.line(x2, y2, x2 - 6 * direction, y2 + 3)
    c.line(x2, y2, x2 - 6 * direction, y2 - 3)


def draw_lifecycle_pdf(path: Path) -> None:
    height = 3.15 * 72
    c = canvas.Canvas(str(path), pagesize=(W, height), invariant=1, pageCompression=1)
    c.setTitle("XS action lifecycle accounting")
    pdf_text(c, 18, height - 23, "Action lifecycle and conservation", 12, bold=True)
    pdf_text(c, W - 18, height - 23, "C=50 fixed in the formal horizon study", 8.5,
             colour=BLUE, align="right")

    y = height - 100
    box_w, box_h = 102, 55
    xs = [18, 141, 264, 387]
    pdf_box(c, xs[0], y, box_w, box_h, "Observe + invoke", ["one model call", "creates C actions"], fill="#DCEFF7", stroke=BLUE)
    pdf_box(c, xs[1], y, box_w, box_h, "Generate", ["M calls", "G = C x M"], fill="#DCEFF7", stroke=BLUE)
    pdf_box(c, xs[2], y, box_w, box_h, "Execute", ["A env.step calls", "native actions"], fill="#DDF2EC", stroke=GREEN)
    pdf_box(c, xs[3], y, box_w + 8, box_h, "Account unused", ["U = G - A", "three disjoint causes"], fill="#FFF2CC", stroke=ORANGE)
    for left, right in zip(xs, xs[1:]):
        pdf_arrow(c, left + box_w, y + box_h / 2, right - 5, y + box_h / 2)

    pdf_text(c, 18, 63, "Call finalization reason", 9.2, bold=True)
    labels = [
        ("horizon", "C - E unused after the planned prefix", BLUE),
        ("trigger", "unexecuted active-horizon tail", VERMILLION),
        ("terminal", "unexecuted prefix at success/cap", MID),
    ]
    x = 18
    for title, desc, colour in labels:
        c.setFillColor(HexColor(colour))
        c.roundRect(x, 28, 8, 24, 2, fill=1, stroke=0)
        pdf_text(c, x + 14, 43, title, 8.5, bold=True, colour=colour)
        pdf_text(c, x + 14, 31, desc, 7.5, colour=DARK)
        x += 166
    pdf_text(c, W / 2, 8, "U = U_horizon + U_trigger + U_terminal; utilization = A / G; mean_actual_horizon is the mean realized actions per finalized call",
             7.5, bold=True, align="center")
    c.showPage()
    c.save()


def font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def draw_lifecycle_png(path: Path) -> None:
    scale = 300 / 72
    width, height = round(W * scale), round(3.15 * 72 * scale)
    image = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(image)
    def xy(value: float) -> int: return round(value * scale)
    d.text((xy(18), xy(14)), "Action lifecycle and conservation", font=font(xy(12), True), fill=DARK)
    d.text((width - xy(18), xy(16)), "C=50 fixed in the formal horizon study", font=font(xy(8.5)), fill=BLUE, anchor="ra")
    y, bw, bh = xy(70), xy(102), xy(55)
    boxes = [
        (18, "Observe + invoke", ["one model call", "creates C actions"], "#DCEFF7", BLUE),
        (141, "Generate", ["M calls", "G = C x M"], "#DCEFF7", BLUE),
        (264, "Execute", ["A env.step calls", "native actions"], "#DDF2EC", GREEN),
        (387, "Account unused", ["U = G - A", "three disjoint causes"], "#FFF2CC", ORANGE),
    ]
    for x0, title, lines, fill, stroke in boxes:
        x = xy(x0); extra = xy(8) if x0 == 387 else 0
        d.rounded_rectangle((x, y, x + bw + extra, y + bh), radius=xy(7), fill=fill, outline=stroke, width=xy(1.2))
        d.text((x + (bw + extra)/2, y + xy(9)), title, font=font(xy(9.2), True), fill=DARK, anchor="ma")
        for i, line in enumerate(lines):
            d.text((x + (bw + extra)/2, y + xy(27 + 12*i)), line, font=font(xy(8.2)), fill=DARK, anchor="ma")
    for x1, x2 in ((120, 136), (243, 259), (366, 382)):
        d.line((xy(x1), y + bh/2, xy(x2), y + bh/2), fill=MID, width=xy(1.4))
        d.polygon(((xy(x2), y+bh/2), (xy(x2-6), y+bh/2-xy(3)), (xy(x2-6), y+bh/2+xy(3))), fill=MID)
    d.text((xy(18), xy(146)), "Call finalization reason", font=font(xy(9.2), True), fill=DARK)
    labels = [("horizon", "C - E unused after planned prefix", BLUE), ("trigger", "unexecuted active-horizon tail", VERMILLION), ("terminal", "unexecuted prefix at success/cap", MID)]
    for i, (title, desc, colour) in enumerate(labels):
        x = xy(18 + 166*i)
        d.rounded_rectangle((x, xy(164), x+xy(8), xy(188)), radius=xy(2), fill=colour)
        d.text((x+xy(14), xy(163)), title, font=font(xy(8.5), True), fill=colour)
        d.text((x+xy(14), xy(176)), desc, font=font(xy(7.3)), fill=DARK)
    footer = "U = U_horizon + U_trigger + U_terminal; utilization = A / G; mean_actual_horizon = mean realized actions per finalized call"
    d.text((width/2, height-xy(14)), footer, font=font(xy(7.4), True), fill=DARK, anchor="ms")
    image.save(path, dpi=(300, 300), optimize=True)


HEAT_COLORS = {
    "both_success": BLUE,
    "h1_only": ORANGE,
    "h20_only": GREEN,
    "both_fail": "#D9DDE3",
}
HEAT_LABELS = {"both_success": "B", "h1_only": "1", "h20_only": "20", "both_fail": "F"}


def draw_heatmap_pdf(rows: list[dict[str, str]], path: Path) -> None:
    height = 3.15 * 72
    c = canvas.Canvas(str(path), pagesize=(W, height), invariant=1, pageCompression=1)
    c.setTitle("XS Static-H1 versus Static-H20 task-state outcomes")
    pdf_text(c, 18, height - 23, "Static-H1 vs Static-H20 outcome complementarity", 12, bold=True)
    grid = {(int(r["task_id"]), int(r["initial_state_id"])): r["outcome_category"] for r in rows}
    left, top, cw, ch = 89, height - 54, 55, 14
    for state in range(5):
        pdf_text(c, left + state*cw + cw/2, top + 5, f"state {state}", 8, bold=True, align="center")
    for task in range(10):
        y = top - (task + 1)*ch
        pdf_text(c, left - 9, y + 3.5, f"task {task}", 8, align="right")
        for state in range(5):
            category = grid[(task, state)]
            c.setFillColor(HexColor(HEAT_COLORS[category]))
            c.setStrokeColor(white)
            c.rect(left + state*cw, y, cw, ch, fill=1, stroke=1)
            pdf_text(c, left + state*cw + cw/2, y + 3.2, HEAT_LABELS[category], 7.5,
                     bold=True, colour="#FFFFFF" if category != "both_fail" else DARK, align="center")
    legend = [("B", "both success (26)", BLUE), ("1", "H1 only (7)", ORANGE), ("20", "H20 only (6)", GREEN), ("F", "both fail (11)", "#D9DDE3")]
    x = 18
    for code, label, colour in legend:
        c.setFillColor(HexColor(colour)); c.rect(x, 12, 14, 12, fill=1, stroke=0)
        pdf_text(c, x+7, 15, code, 6.8, bold=True, colour="#FFFFFF" if code != "F" else DARK, align="center")
        pdf_text(c, x+19, 15, label, 7.5)
        x += 124
    c.showPage(); c.save()


def draw_heatmap_png(rows: list[dict[str, str]], path: Path) -> None:
    scale = 300/72; width, height = round(W*scale), round(3.15*72*scale)
    image = Image.new("RGB", (width, height), "white"); d = ImageDraw.Draw(image)
    def xy(v: float) -> int: return round(v*scale)
    d.text((xy(18), xy(14)), "Static-H1 vs Static-H20 outcome complementarity", font=font(xy(12), True), fill=DARK)
    grid = {(int(r["task_id"]), int(r["initial_state_id"])): r["outcome_category"] for r in rows}
    left, top, cw, ch = xy(89), xy(54), xy(55), xy(14)
    for state in range(5): d.text((left+state*cw+cw/2, top-xy(5)), f"state {state}", font=font(xy(8), True), fill=DARK, anchor="ms")
    for task in range(10):
        y=top+task*ch; d.text((left-xy(9), y+ch/2), f"task {task}", font=font(xy(8)), fill=DARK, anchor="rm")
        for state in range(5):
            cat=grid[(task,state)]; x=left+state*cw
            d.rectangle((x,y,x+cw,y+ch), fill=HEAT_COLORS[cat], outline="white", width=xy(0.8))
            d.text((x+cw/2,y+ch/2), HEAT_LABELS[cat], font=font(xy(7.5), True), fill="white" if cat!="both_fail" else DARK, anchor="mm")
    legend=[("B","both success (26)",BLUE),("1","H1 only (7)",ORANGE),("20","H20 only (6)",GREEN),("F","both fail (11)","#D9DDE3")]
    for i,(code,label,colour) in enumerate(legend):
        x=xy(18+124*i); y=height-xy(27); d.rectangle((x,y,x+xy(14),y+xy(12)),fill=colour)
        d.text((x+xy(7),y+xy(6)),code,font=font(xy(6.8),True),fill="white" if code!="F" else DARK,anchor="mm")
        d.text((x+xy(19),y+xy(6)),label,font=font(xy(7.5)),fill=DARK,anchor="lm")
    image.save(path,dpi=(300,300),optimize=True)


CONDITION_ORDER = [
    ("development", "Static-H1"), ("development", "Static-H10"),
    ("development", "Static-H20"), ("development", "Adaptive-v1"),
    ("held-out", "Static-H20"), ("held-out", "Adaptive-v2a"),
]
SEGMENTS = [
    ("executed_env_steps", "executed", BLUE),
    ("horizon_tail_discarded_actions", "horizon-unused", SKY),
    ("trigger_tail_discarded_actions", "trigger-unused", VERMILLION),
    ("terminal_tail_unused_actions", "terminal-unused", GREY),
]


def aggregate_accounting(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows: grouped[(row["cohort"], row["condition"])].append(row)
    result=[]
    for key in CONDITION_ORDER:
        items=grouped[key]
        generated=sum(int(x["generated_actions"]) for x in items)
        values={field:sum(int(x[field]) for x in items) for field,_,_ in SEGMENTS}
        if generated != sum(values.values()): raise ValueError(f"aggregate accounting fails: {key}")
        result.append({"cohort":key[0],"condition":key[1],"generated_mean":generated/len(items),"utilization":values["executed_env_steps"]/generated,"parts":values})
    return result


def draw_accounting_pdf(data: list[dict[str, Any]], path: Path) -> None:
    height=3.55*72; c=canvas.Canvas(str(path),pagesize=(W,height),invariant=1,pageCompression=1)
    c.setTitle("XS generated executed unused action accounting")
    pdf_text(c,18,height-23,"Generated, executed, and unused action composition",12,bold=True)
    pdf_text(c,18,height-38,"Bars are normalized within condition; G/ep labels preserve the absolute generated scale.",7.5,colour=MID)
    left,right=116,W-80; bar_w=right-left; y=height-58
    for index,item in enumerate(data):
        if index==4:
            c.setStrokeColor(HexColor("#CBD0D7")); c.line(18,y+5,W-18,y+5)
            pdf_text(c,18,y-8,"untouched held-out",7.5,bold=True,colour=MID); y-=25
        elif index==0:
            pdf_text(c,18,y+13,"parity-corrected development",7.5,bold=True,colour=MID); y-=12
        pdf_text(c,left-8,y+3,item["condition"],8.2,bold=item["condition"].startswith("Adaptive"),align="right")
        x=left; total=sum(item["parts"].values())
        for field,_,colour in SEGMENTS:
            width=bar_w*item["parts"][field]/total
            if width>0:
                c.setFillColor(HexColor(colour)); c.rect(x,y,width,13,fill=1,stroke=0)
            x+=width
        pdf_text(c,right+6,y+3,f"G/ep={item['generated_mean']:.0f}; util={100*item['utilization']:.1f}%",7.4)
        y-=25
    x=18
    for _,label,colour in SEGMENTS:
        c.setFillColor(HexColor(colour)); c.rect(x,12,12,9,fill=1,stroke=0)
        pdf_text(c,x+16,13,label,7.2); x+=115
    c.showPage(); c.save()


def draw_accounting_png(data: list[dict[str, Any]], path: Path) -> None:
    scale=300/72; width,height=round(W*scale),round(3.55*72*scale)
    image=Image.new("RGB",(width,height),"white"); d=ImageDraw.Draw(image)
    def xy(v:float)->int:return round(v*scale)
    d.text((xy(18),xy(14)),"Generated, executed, and unused action composition",font=font(xy(12),True),fill=DARK)
    d.text((xy(18),xy(31)),"Bars are normalized within condition; G/ep labels preserve the absolute generated scale.",font=font(xy(7.5)),fill=MID)
    left,right=xy(116),width-xy(80); bw=right-left; y=xy(58)
    for index,item in enumerate(data):
        if index==4:
            d.line((xy(18),y-xy(5),width-xy(18),y-xy(5)),fill="#CBD0D7",width=xy(0.8)); d.text((xy(18),y+xy(8)),"untouched held-out",font=font(xy(7.5),True),fill=MID,anchor="ls"); y+=xy(25)
        elif index==0:
            d.text((xy(18),y-xy(13)),"parity-corrected development",font=font(xy(7.5),True),fill=MID); y+=xy(12)
        d.text((left-xy(8),y+xy(6.5)),item["condition"],font=font(xy(8.2),item["condition"].startswith("Adaptive")),fill=DARK,anchor="rm")
        x=left; total=sum(item["parts"].values())
        for field,_,colour in SEGMENTS:
            seg=round(bw*item["parts"][field]/total)
            if seg>0:d.rectangle((x,y,x+seg,y+xy(13)),fill=colour)
            x+=seg
        d.text((right+xy(6),y+xy(6.5)),f"G/ep={item['generated_mean']:.0f}; util={100*item['utilization']:.1f}%",font=font(xy(7.4)),fill=DARK,anchor="lm")
        y+=xy(25)
    for i,(_,label,colour) in enumerate(SEGMENTS):
        x=xy(18+115*i); y=height-xy(21); d.rectangle((x,y,x+xy(12),y+xy(9)),fill=colour); d.text((x+xy(16),y+xy(4.5)),label,font=font(xy(7.2)),fill=DARK,anchor="lm")
    image.save(path,dpi=(300,300),optimize=True)


def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    root=Path(__file__).resolve().parents[2]
    parser.add_argument("--source-dir",type=Path,default=root/"figures/xs_adaptive_extended/source")
    parser.add_argument("--output-dir",type=Path,default=root/"figures/xs_adaptive_extended")
    args=parser.parse_args(); source=args.source_dir.resolve(); out=args.output_dir.resolve(); out.mkdir(parents=True,exist_ok=True)
    episodes=read_csv(source/"xs_episode_metrics.csv"); outcomes=read_csv(source/"xs_h1_h20_task_state_outcomes.csv")
    outputs=[
        out/"xs_action_lifecycle_accounting.pdf",out/"xs_action_lifecycle_accounting.png",
        out/"xs_task_state_outcome_heatmap.pdf",out/"xs_task_state_outcome_heatmap.png",
        out/"xs_action_accounting_stacked.pdf",out/"xs_action_accounting_stacked.png",
    ]
    draw_lifecycle_pdf(outputs[0]); draw_lifecycle_png(outputs[1])
    draw_heatmap_pdf(outcomes,outputs[2]); draw_heatmap_png(outcomes,outputs[3])
    accounting=aggregate_accounting(episodes); draw_accounting_pdf(accounting,outputs[4]); draw_accounting_png(accounting,outputs[5])
    inputs=sorted(source.glob("xs_*"))
    provenance={
        "schema_version":1,"frozen_data_only":True,"model_loaded":False,"rollout_executed":False,
        "palette":"Okabe-Ito","png_dpi":300,"inputs":[{"path":str(p.relative_to(root)).replace('\\','/'),"sha256":sha256(p)} for p in inputs],
        "outputs":[{"path":p.name,"sha256":sha256(p)} for p in outputs],
    }
    (out/"xs_extended_figure_provenance.json").write_text(json.dumps(provenance,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({"status":"ok","outputs":[p.name for p in outputs]}))


if __name__ == "__main__":
    main()
