#!/usr/bin/env python3
"""Generate the XS adaptive-control method and success--compute figures.

The quantitative figure reads only the two frozen final-result artifacts:

* analysis/final_vla_results.csv
* analysis/final_vla_statistics.json

No rollout, model, LIBERO, or early parity-result directory is imported or
opened.  The script cross-checks the CSV against the JSON before drawing.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.colors import HexColor, white
from reportlab.pdfgen import canvas


PAGE_W = 7.2 * 72
TRADEOFF_H = 3.65 * 72
MECHANISM_H = 4.25 * 72
STATIC_BLUE = "#0072B2"       # Okabe--Ito blue
ADAPTIVE_ORANGE = "#D55E00"   # Okabe--Ito vermillion
TEAL = "#009E73"              # Okabe--Ito bluish green
LIGHT_BLUE = "#DCEFF7"
LIGHT_ORANGE = "#FBE8DE"
LIGHT_TEAL = "#DDF2EC"
LIGHT_GREY = "#F1F3F5"
DARK = "#20252B"
MID = "#6B7280"


@dataclass(frozen=True)
class ResultPoint:
    cohort: str
    condition: str
    episodes: int
    successes: int
    success_rate: float
    wilson_low: float
    wilson_high: float
    model_calls_mean: float


EXPECTED = {
    "development": (
        "Static-H1",
        "Static-H10",
        "Static-H20",
        "Adaptive-v1-H20→H1",
    ),
    "confirmatory": (
        "Static-H20",
        "Adaptive-v2a-H20→H1",
    ),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_points(results_csv: Path, statistics_json: Path) -> list[ResultPoint]:
    with results_csv.open(newline="", encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    with statistics_json.open(encoding="utf-8") as stream:
        statistics = json.load(stream)

    separation = statistics.get("cohort_separation", {})
    if separation.get("strict") is not True:
        raise ValueError("frozen statistics do not enforce strict cohort separation")
    if separation.get("cross_cohort_pairing_or_pooling") is not False:
        raise ValueError("cross-cohort pairing/pooling must be false")

    selected_rows: dict[tuple[str, str], dict[str, str]] = {}
    for row in rows:
        role = row["cohort_role"]
        condition = row["condition"]
        if role in EXPECTED and condition in EXPECTED[role]:
            key = (role, condition)
            if key in selected_rows:
                raise ValueError(f"duplicate frozen result row: {key}")
            selected_rows[key] = row

    expected_keys = {
        (cohort, condition)
        for cohort, conditions in EXPECTED.items()
        for condition in conditions
    }
    if set(selected_rows) != expected_keys:
        raise ValueError(
            f"frozen result selection mismatch; missing={sorted(expected_keys-set(selected_rows))}, "
            f"extra={sorted(set(selected_rows)-expected_keys)}"
        )

    json_rows = {
        (item["cohort"], item["condition"]): item
        for item in statistics.get("condition_results", [])
    }
    points: list[ResultPoint] = []
    for cohort, conditions in EXPECTED.items():
        for condition in conditions:
            csv_row = selected_rows[(cohort, condition)]
            json_row = json_rows[(cohort, condition)]
            csv_values = (
                int(csv_row["episodes"]),
                int(csv_row["successes"]),
                float(csv_row["success_rate"]),
                float(csv_row["wilson95_low"]),
                float(csv_row["wilson95_high"]),
                float(csv_row["model_calls_mean"]),
            )
            json_values = (
                int(json_row["episodes"]),
                int(json_row["successes"]),
                float(json_row["success_rate"]),
                float(json_row["wilson_95_ci"][0]),
                float(json_row["wilson_95_ci"][1]),
                float(json_row["model_calls_mean"]),
            )
            if any(not math.isclose(a, b, rel_tol=0.0, abs_tol=1e-12) for a, b in zip(csv_values, json_values)):
                raise ValueError(f"CSV/JSON frozen-value mismatch for {(cohort, condition)}")
            points.append(ResultPoint(cohort, condition, *csv_values))

    excluded = statistics.get("exclusions", [])
    if not excluded or any(item.get("included_in_final_performance_results") is not False for item in excluded):
        raise ValueError("the frozen early-parity exclusion is missing or not enforced")
    return points


def _short_name(condition: str) -> str:
    return (
        condition.replace("Static-", "")
        .replace("Adaptive-v1-H20→H1", "Adaptive-v1")
        .replace("Adaptive-v2a-H20→H1", "Adaptive-v2a")
    )


def _pdf_text(c: canvas.Canvas, x: float, y: float, text: str, size: float = 9, *,
              bold: bool = False, colour: str = DARK, align: str = "left") -> None:
    c.setFillColor(HexColor(colour))
    c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
    if align == "center":
        c.drawCentredString(x, y, text)
    elif align == "right":
        c.drawRightString(x, y, text)
    else:
        c.drawString(x, y, text)


def _pdf_wrapped(c: canvas.Canvas, x: float, y: float, lines: Sequence[str], *,
                 size: float = 9, leading: float = 11, bold_first: bool = False,
                 colour: str = DARK, align: str = "center") -> None:
    for index, line in enumerate(lines):
        _pdf_text(c, x, y - index * leading, line, size,
                  bold=bold_first and index == 0, colour=colour, align=align)


def _rounded_box(c: canvas.Canvas, x: float, y: float, w: float, h: float, *,
                 fill: str, stroke: str, lines: Sequence[str], size: float = 9) -> None:
    c.setFillColor(HexColor(fill))
    c.setStrokeColor(HexColor(stroke))
    c.setLineWidth(1.1)
    c.roundRect(x, y, w, h, 7, fill=1, stroke=1)
    total = (len(lines) - 1) * (size + 2)
    _pdf_wrapped(c, x + w / 2, y + h / 2 + total / 2 - size * 0.35,
                 lines, size=size, leading=size + 2, bold_first=True)


def _arrow(c: canvas.Canvas, x1: float, y1: float, x2: float, y2: float, *, colour: str = MID) -> None:
    c.setStrokeColor(HexColor(colour))
    c.setFillColor(HexColor(colour))
    c.setLineWidth(1.4)
    c.line(x1, y1, x2, y2)
    angle = math.atan2(y2 - y1, x2 - x1)
    head = 5.5
    spread = 0.55
    points = [
        (x2, y2),
        (x2 - head * math.cos(angle - spread), y2 - head * math.sin(angle - spread)),
        (x2 - head * math.cos(angle + spread), y2 - head * math.sin(angle + spread)),
    ]
    path = c.beginPath()
    path.moveTo(*points[0])
    path.lineTo(*points[1])
    path.lineTo(*points[2])
    path.close()
    c.drawPath(path, fill=1, stroke=0)


def draw_mechanism_pdf(output: Path) -> None:
    c = canvas.Canvas(str(output), pagesize=(PAGE_W, MECHANISM_H), invariant=1, pageCompression=1)
    c.setTitle("XS adaptive SmolVLA execution-horizon mechanism")
    _pdf_text(c, 18, MECHANISM_H - 22, "SmolVLA execution-horizon control", 12, bold=True)

    _rounded_box(
        c, 18, MECHANISM_H - 67, PAGE_W - 36, 31,
        fill=LIGHT_BLUE, stroke=STATIC_BLUE,
        lines=("Every model invocation predicts a fixed chunk: C = 50 native actions",), size=10,
    )

    _pdf_text(c, 18, MECHANISM_H - 91, "Static-H20", 10, bold=True, colour=STATIC_BLUE)
    static_y = MECHANISM_H - 143
    boxes = [
        (18, 112, ("Observe + invoke", "SmolVLA; C = 50"), LIGHT_BLUE, STATIC_BLUE),
        (156, 112, ("Execute prefix", "E = 20 actions"), LIGHT_BLUE, STATIC_BLUE),
        (294, 144, ("Re-observe", "discard actions 21--50"), LIGHT_GREY, MID),
    ]
    for x, w, lines, fill, stroke in boxes:
        _rounded_box(c, x, static_y, w, 43, fill=fill, stroke=stroke, lines=lines, size=9)
    _arrow(c, 130, static_y + 21.5, 156, static_y + 21.5, colour=STATIC_BLUE)
    _arrow(c, 268, static_y + 21.5, 294, static_y + 21.5, colour=STATIC_BLUE)
    _arrow(c, 438, static_y + 8, 489, static_y + 8, colour=STATIC_BLUE)
    _pdf_text(c, 492, static_y + 4.5, "repeat", 8.5, colour=STATIC_BLUE)

    adaptive_top = static_y - 30
    _pdf_text(c, 18, adaptive_top, "Adaptive-v1 / Adaptive-v2a (default H20)", 10,
              bold=True, colour=ADAPTIVE_ORANGE)
    y = adaptive_top - 56
    adaptive_boxes = [
        (18, 82, ("Monitor", "E = 20"), LIGHT_ORANGE, ADAPTIVE_ORANGE),
        (117, 94, ("Trigger", "during prefix"), LIGHT_ORANGE, ADAPTIVE_ORANGE),
        (228, 111, ("Execute current", "native action;", "drop unused tail"), LIGHT_ORANGE, ADAPTIVE_ORANGE),
        (356, 66, ("Next call", "E = 1"), LIGHT_TEAL, TEAL),
        (439, 62, ("Recover", "E = 20"), LIGHT_TEAL, TEAL),
    ]
    for x, w, lines, fill, stroke in adaptive_boxes:
        _rounded_box(c, x, y, w, 54, fill=fill, stroke=stroke, lines=lines, size=8.5)
    for x1, x2, colour in ((100, 117, ADAPTIVE_ORANGE), (211, 228, ADAPTIVE_ORANGE),
                           (339, 356, TEAL), (422, 439, TEAL)):
        _arrow(c, x1, y + 27, x2, y + 27, colour=colour)
    _pdf_text(c, 385, y - 14, "v2a: full 20-action cooldown; then monitoring resumes",
              8.5, colour=TEAL, align="center")
    c.setStrokeColor(HexColor(TEAL))
    c.setLineWidth(1.2)
    c.line(470, y - 3, 470, y - 25)
    c.line(470, y - 25, 59, y - 25)
    _arrow(c, 59, y - 25, 59, y - 1, colour=TEAL)

    _rounded_box(
        c, 18, 18, PAGE_W - 36, 34,
        fill="#FFF4CC", stroke="#E69F00",
        lines=("Invariant in all conditions: clip_actions = false; env.step receives each native action unchanged",),
        size=9.5,
    )
    c.showPage()
    c.save()


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        Path("C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf" if bold else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size=size)
    return ImageFont.load_default()


def _point_colour(point: ResultPoint) -> str:
    return ADAPTIVE_ORANGE if point.condition.startswith("Adaptive") else STATIC_BLUE


def _symbol_pdf(c: canvas.Canvas, x: float, y: float, point: ResultPoint) -> None:
    colour = HexColor(_point_colour(point))
    c.setStrokeColor(white)
    c.setFillColor(colour)
    c.setLineWidth(0.8)
    if point.condition.startswith("Adaptive"):
        c.rect(x - 4, y - 4, 8, 8, fill=1, stroke=1)
    elif point.condition == "Static-H10":
        path = c.beginPath()
        path.moveTo(x, y + 5)
        path.lineTo(x - 4.8, y - 4)
        path.lineTo(x + 4.8, y - 4)
        path.close()
        c.drawPath(path, fill=1, stroke=1)
    elif point.condition == "Static-H20":
        path = c.beginPath()
        path.moveTo(x, y + 5)
        path.lineTo(x - 5, y)
        path.lineTo(x, y - 5)
        path.lineTo(x + 5, y)
        path.close()
        c.drawPath(path, fill=1, stroke=1)
    else:
        c.circle(x, y, 4.3, fill=1, stroke=1)


def _label_offsets(point: ResultPoint) -> tuple[float, float, str]:
    key = (point.cohort, point.condition)
    return {
        ("development", "Static-H1"): (-6, 15, "right"),
        ("development", "Static-H10"): (8, 19, "left"),
        ("development", "Static-H20"): (8, 25, "left"),
        ("development", "Adaptive-v1-H20→H1"): (8, -34, "left"),
        ("confirmatory", "Static-H20"): (8, 25, "left"),
        ("confirmatory", "Adaptive-v2a-H20→H1"): (8, -29, "left"),
    }[key]


def _tradeoff_layout() -> tuple[list[tuple[float, float, float, float]], float, float]:
    left, right, gap = 45.0, 14.0, 25.0
    bottom, top = 43.0, 34.0
    width = (PAGE_W - left - right - gap) / 2
    height = TRADEOFF_H - bottom - top
    return [(left, bottom, width, height), (left + width + gap, bottom, width, height)], bottom, top


def _x_map(value: float, x: float, width: float) -> float:
    lo, hi = math.log10(6.0), math.log10(200.0)
    return x + (math.log10(value) - lo) / (hi - lo) * width


def _y_map(value: float, y: float, height: float) -> float:
    return y + value * height


def draw_tradeoff_pdf(points: Sequence[ResultPoint], output: Path) -> None:
    c = canvas.Canvas(str(output), pagesize=(PAGE_W, TRADEOFF_H), invariant=1, pageCompression=1)
    c.setTitle("XS success-compute trade-off from frozen VLA results")
    panels, _, _ = _tradeoff_layout()
    cohorts = ("development", "confirmatory")
    titles = ("a  Parity-corrected development", "b  Untouched held-out confirmatory")
    for panel_index, (cohort, title, (x, y, w, h)) in enumerate(zip(cohorts, titles, panels)):
        _pdf_text(c, x, TRADEOFF_H - 19, title, 10.5, bold=True)
        c.setStrokeColor(HexColor("#B8BEC6"))
        c.setLineWidth(0.45)
        for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            yy = _y_map(tick, y, h)
            c.line(x, yy, x + w, yy)
            _pdf_text(c, x - 6, yy - 3, f"{tick:.1f}", 9, colour=MID, align="right")
        c.setStrokeColor(HexColor(DARK))
        c.setLineWidth(0.8)
        c.line(x, y, x, y + h)
        c.line(x, y, x + w, y)
        for tick in (10, 20, 50, 100, 200):
            xx = _x_map(tick, x, w)
            c.line(xx, y, xx, y - 3)
            _pdf_text(c, xx, y - 14, str(tick), 9, colour=MID, align="center")
        _pdf_text(c, x + w / 2, 13, "Mean model invocations per episode (log scale)", 9.5, align="center")
        if panel_index == 0:
            c.saveState()
            c.translate(12, y + h / 2)
            c.rotate(90)
            _pdf_text(c, 0, 0, "Success@280", 9.5, align="center")
            c.restoreState()

        for point in (item for item in points if item.cohort == cohort):
            px = _x_map(point.model_calls_mean, x, w)
            py = _y_map(point.success_rate, y, h)
            low = _y_map(point.wilson_low, y, h)
            high = _y_map(point.wilson_high, y, h)
            colour = _point_colour(point)
            c.setStrokeColor(HexColor(colour))
            c.setLineWidth(1.25)
            c.line(px, low, px, high)
            c.line(px - 3.2, low, px + 3.2, low)
            c.line(px - 3.2, high, px + 3.2, high)
            _symbol_pdf(c, px, py, point)
            dx, dy, align = _label_offsets(point)
            label = f"{_short_name(point.condition)} ({point.successes}/{point.episodes})"
            _pdf_text(c, px + dx, py + dy, label, 8.8, bold=True, colour=colour, align=align)

    c.showPage()
    c.save()


def draw_tradeoff_png(points: Sequence[ResultPoint], output: Path, dpi: int = 300) -> None:
    scale = dpi / 72.0
    image = Image.new("RGB", (round(PAGE_W * scale), round(TRADEOFF_H * scale)), "white")
    draw = ImageDraw.Draw(image)

    def xy(value: float) -> int:
        return round(value * scale)

    def py(value: float) -> int:
        return image.height - xy(value)

    panels, _, _ = _tradeoff_layout()
    cohorts = ("development", "confirmatory")
    titles = ("a  Parity-corrected development", "b  Untouched held-out confirmatory")
    font9 = _font(round(9 * scale))
    font9b = _font(round(9 * scale), bold=True)
    font10 = _font(round(10.5 * scale), bold=True)
    for panel_index, (cohort, title, (x, y, w, h)) in enumerate(zip(cohorts, titles, panels)):
        draw.text((xy(x), py(TRADEOFF_H - 17)), title, font=font10, fill=DARK, anchor="ls")
        for tick in (0.0, 0.2, 0.4, 0.6, 0.8, 1.0):
            yy = _y_map(tick, y, h)
            draw.line((xy(x), py(yy), xy(x + w), py(yy)), fill="#B8BEC6", width=max(1, xy(0.45)))
            draw.text((xy(x - 6), py(yy)), f"{tick:.1f}", font=font9, fill=MID, anchor="rm")
        draw.line((xy(x), py(y), xy(x), py(y + h)), fill=DARK, width=xy(0.8))
        draw.line((xy(x), py(y), xy(x + w), py(y)), fill=DARK, width=xy(0.8))
        for tick in (10, 20, 50, 100, 200):
            xx = _x_map(tick, x, w)
            draw.line((xy(xx), py(y), xy(xx), py(y - 3)), fill=DARK, width=xy(0.6))
            draw.text((xy(xx), py(y - 14)), str(tick), font=font9, fill=MID, anchor="mm")
        draw.text((xy(x + w / 2), py(13)), "Mean model invocations per episode (log scale)",
                  font=font9, fill=DARK, anchor="mm")
        if panel_index == 0:
            label = Image.new("RGBA", (xy(120), xy(18)), (255, 255, 255, 0))
            label_draw = ImageDraw.Draw(label)
            label_draw.text((label.width // 2, label.height // 2), "Success@280", font=font9,
                            fill=DARK, anchor="mm")
            label = label.rotate(90, expand=True)
            image.paste(label, (xy(4), py(y + h / 2) - label.height // 2), label)

        for point in (item for item in points if item.cohort == cohort):
            px = _x_map(point.model_calls_mean, x, w)
            value_y = _y_map(point.success_rate, y, h)
            low = _y_map(point.wilson_low, y, h)
            high = _y_map(point.wilson_high, y, h)
            colour = _point_colour(point)
            draw.line((xy(px), py(low), xy(px), py(high)), fill=colour, width=xy(1.25))
            draw.line((xy(px - 3.2), py(low), xy(px + 3.2), py(low)), fill=colour, width=xy(1.25))
            draw.line((xy(px - 3.2), py(high), xy(px + 3.2), py(high)), fill=colour, width=xy(1.25))
            radius = xy(4.2)
            cx, cy = xy(px), py(value_y)
            if point.condition.startswith("Adaptive"):
                draw.rectangle((cx - radius, cy - radius, cx + radius, cy + radius), fill=colour, outline="white", width=xy(0.8))
            elif point.condition == "Static-H10":
                draw.polygon(((cx, cy - radius), (cx - radius, cy + radius), (cx + radius, cy + radius)), fill=colour)
            elif point.condition == "Static-H20":
                draw.polygon(((cx, cy - radius), (cx - radius, cy), (cx, cy + radius), (cx + radius, cy)), fill=colour)
            else:
                draw.ellipse((cx - radius, cy - radius, cx + radius, cy + radius), fill=colour, outline="white", width=xy(0.8))
            dx, dy, align = _label_offsets(point)
            anchor = "rs" if align == "right" else "ls"
            label = f"{_short_name(point.condition)} ({point.successes}/{point.episodes})"
            draw.text((xy(px + dx), py(value_y + dy)), label, font=font9b, fill=colour, anchor=anchor)

    image.save(output, dpi=(dpi, dpi), optimize=True)


def _write_source_csv(points: Iterable[ResultPoint], output: Path) -> None:
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(("cohort", "condition", "episodes", "successes", "success_at_280",
                         "wilson95_low", "wilson95_high", "mean_model_invocations"))
        for point in points:
            writer.writerow((point.cohort, point.condition, point.episodes, point.successes,
                             f"{point.success_rate:.12g}", f"{point.wilson_low:.12g}",
                             f"{point.wilson_high:.12g}", f"{point.model_calls_mean:.12g}"))


def _write_provenance(root: Path, results_csv: Path, statistics_json: Path,
                      output_dir: Path, outputs: Sequence[Path]) -> None:
    def display_path(path: Path) -> str:
        resolved = path.resolve()
        try:
            return resolved.relative_to(root).as_posix()
        except ValueError:
            return str(resolved)

    method_inputs = [
        root / "docs" / "ADAPTIVE_V2_PREREG.md",
        root / "docs" / "ADAPTIVE_V2_FORMAL_HELDOUT_PREREG.md",
        root / "src" / "libero_platform" / "policies" / "fixed_h_action_buffer.py",
        root / "src" / "libero_platform" / "policies" / "adaptive_v2_trigger.py",
        root / "configs" / "evaluation" / "libero_spatial_adaptive_v2_formal_heldout.yaml",
    ]
    provenance = {
        "schema_version": 1,
        "quantitative_inputs_only": [
            {"path": display_path(results_csv), "sha256": _sha256(results_csv)},
            {"path": display_path(statistics_json), "sha256": _sha256(statistics_json)},
        ],
        "method_diagram_inputs": [
            {"path": display_path(path), "sha256": _sha256(path)}
            for path in method_inputs if path.exists()
        ],
        "constraints": {
            "cohorts_separate": True,
            "cross_cohort_pooling": False,
            "early_parity_results_excluded": True,
            "prediction_chunk_C": 50,
            "clip_actions": False,
            "native_actions_unchanged": True,
            "tradeoff_y_axis": [0.0, 1.0],
            "tradeoff_x_axis": "logarithmic full data range",
            "png_dpi": 300,
            "palette": "Okabe-Ito",
        },
        "outputs": [
            {"path": str(path.relative_to(output_dir)), "sha256": _sha256(path)}
            for path in outputs
        ],
    }
    target = output_dir / "xs_figure_provenance.json"
    target.write_text(json.dumps(provenance, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--root", type=Path, default=default_root)
    parser.add_argument("--results-csv", type=Path)
    parser.add_argument("--statistics-json", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    results_csv = (args.results_csv or root / "analysis" / "final_vla_results.csv").resolve()
    statistics_json = (args.statistics_json or root / "analysis" / "final_vla_statistics.json").resolve()
    output_dir = (args.output_dir or root / "figures" / "xs_adaptive").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    points = _read_points(results_csv, statistics_json)
    mechanism = output_dir / "xs_adaptive_mechanism.pdf"
    tradeoff_pdf = output_dir / "xs_success_compute_tradeoff.pdf"
    tradeoff_png = output_dir / "xs_success_compute_tradeoff.png"
    source_csv = output_dir / "xs_success_compute_source.csv"
    draw_mechanism_pdf(mechanism)
    draw_tradeoff_pdf(points, tradeoff_pdf)
    draw_tradeoff_png(points, tradeoff_png, dpi=300)
    _write_source_csv(points, source_csv)
    _write_provenance(root, results_csv, statistics_json, output_dir,
                      (mechanism, tradeoff_pdf, tradeoff_png, source_csv))
    print(json.dumps({
        "status": "ok",
        "output_dir": str(output_dir),
        "outputs": [path.name for path in (mechanism, tradeoff_pdf, tradeoff_png, source_csv,
                                            output_dir / "xs_figure_provenance.json")],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
