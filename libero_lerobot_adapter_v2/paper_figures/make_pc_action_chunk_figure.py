from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


OUTPUT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 2160, 945
BLUE, ORANGE = "#3B6EA8", "#D9782D"
DARK, MID, GRID, WHITE = "#202124", "#55595E", "#D9DDE2", "#FFFFFF"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(f"C:/Windows/Fonts/{name}", size)


def centered(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str,
             text_font: ImageFont.FreeTypeFont, fill: str = DARK) -> None:
    box = draw.textbbox((0, 0), text, font=text_font)
    draw.text((xy[0] - (box[2] - box[0]) / 2, xy[1]), text,
              font=text_font, fill=fill)


def panel(draw: ImageDraw.ImageDraw, left: int, right: int, title: str,
          ylabel: str, maximum: float, tick: int, values: list[float],
          value_labels: list[str]) -> tuple[list[tuple[int, int, int, int]], int]:
    top, bottom = 285, 790
    draw.text((left + 260, 210), title, font=font(32, True), fill=DARK)
    axis_left, axis_right = left + 150, right - 55
    for value in range(0, int(maximum) + 1, tick):
        y = bottom - int((value / maximum) * (bottom - top))
        draw.line((axis_left, y, axis_right, y), fill=GRID, width=2)
        label = str(value)
        box = draw.textbbox((0, 0), label, font=font(24))
        draw.text((axis_left - 18 - (box[2] - box[0]), y - 14), label,
                  font=font(24), fill=MID)
    draw.line((axis_left, top, axis_left, bottom), fill=DARK, width=3)
    draw.line((axis_left, bottom, axis_right, bottom), fill=DARK, width=3)

    # Rotated y-axis label keeps the figure compact and journal-like.
    label_layer = Image.new("RGBA", (520, 70), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_layer)
    label_draw.text((5, 5), ylabel, font=font(27), fill=DARK)
    rotated = label_layer.rotate(90, expand=True)
    image.alpha_composite(rotated, (left + 25, top + 55))

    centers = [axis_left + 245, axis_right - 245]
    bars = []
    for x, value, color, label in zip(centers, values, (BLUE, ORANGE), value_labels):
        bar_top = bottom - int((value / maximum) * (bottom - top))
        rect = (x - 105, bar_top, x + 105, bottom)
        draw.rounded_rectangle(rect, radius=8, fill=color)
        bars.append(rect)
        centered(draw, (x, bar_top - 55), label, font(27, True))
    for x, name, action_steps in zip(centers, ("Native", "Smooth"), ("n_action = 1", "n_action = 10")):
        centered(draw, (x, bottom + 24), name, font(29, True))
        centered(draw, (x, bottom + 62), action_steps, font(23), MID)
    return bars, bottom


image = Image.new("RGBA", (WIDTH, HEIGHT), WHITE)
draw = ImageDraw.Draw(image)
centered(draw, (WIDTH // 2, 42), "PC-Side Evaluation of Action-Chunk Strategies", font(43, True))
centered(
    draw,
    (WIDTH // 2, 105),
    "LIBERO-Object | 10 tasks x 3 rollouts per strategy | Seed 42 | Hard reset",
    font(27),
    MID,
)
draw.line((90, 165, WIDTH - 90, 165), fill=GRID, width=3)

panel(draw, 70, 1050, "(a) Task Success", "Success rate (%)", 100, 20,
      [73.33, 80.00], ["73.33%  (22/30)", "80.00%  (24/30)"])
time_bars, _ = panel(draw, 1110, 2090, "(b) Execution Efficiency",
                     "Mean rollout time (s)", 250, 50,
                     [197.94, 37.05], ["197.94 s", "37.05 s"])

# Place the speed-up callout in unused space above the Smooth bar.
smooth = time_bars[1]
callout_x, callout_y = (smooth[0] + smooth[2]) // 2, 420
centered(draw, (callout_x, callout_y), "5.34x faster", font(29, True), ORANGE)
draw.line((callout_x, callout_y + 48, callout_x, smooth[1] - 14), fill=ORANGE, width=5)
draw.polygon([(callout_x - 10, smooth[1] - 25), (callout_x + 10, smooth[1] - 25),
              (callout_x, smooth[1] - 8)], fill=ORANGE)

png = OUTPUT / "pc_action_chunk_results.png"
pdf = OUTPUT / "pc_action_chunk_results.pdf"
image.convert("RGB").save(png, dpi=(300, 300), optimize=True)
image.convert("RGB").save(pdf, resolution=300.0)
print(png)
print(pdf)
