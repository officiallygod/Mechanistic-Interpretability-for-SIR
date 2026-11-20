"""Generate the MIS logo: the letters "MI" followed by a 7x7 activation map whose
lit cells form the "S".

The grid stands for the ResNet-18 layer-4 feature map (7x7 at 224 px input);
the lit cells are the "latent kept set". "MI" is set in Arial Bold, converted
to outlines so the logo does not depend on installed fonts. Arial Bold ships
with Windows but may not be redistributed, so it is read from the system font
folder (override with MIS_LOGO_FONT=<path to .ttf>).
"MI" is white (dark on light backgrounds), the S is orange.

Writes to docs/assets/branding/ (served by the docs site, reusable in slides):
  mis-logo-dark.svg / .png   for dark backgrounds
  mis-logo-light.svg / .png  for light backgrounds

Usage: python tools/make_logo.py [out_dir]
"""
import os
import sys
from pathlib import Path

from fontTools.pens.basePen import BasePen
from fontTools.pens.boundsPen import BoundsPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont
from PIL import Image, ImageChops, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
FONT_PATH = Path(os.environ.get("MIS_LOGO_FONT", "C:/Windows/Fonts/arialbd.ttf"))
TEXT = "MI"

# 7x7 map, row by row. "S" = lit cell, "." = inactive cell.
PATTERN = [
    ".......",
    "..SSSS.",
    ".S.....",
    "..SSS..",
    ".....S.",
    ".SSSS..",
    ".......",
]
S_ROWS = (1, 5)  # first and last row of the lit S; "MI" is scaled to the same height

ORANGE = "F26B1D"
THEMES = {
    # inactive cells cycle through three shades so the grid reads as an activation map
    "dark": {"text": "FFFFFF", "inactive": ["2A2A44", "34344F", "3E3E5C"]},
    "light": {"text": "222222", "inactive": ["E4E4EC", "D9D9E3", "CDCDDA"]},
}

N = len(PATTERN)
GAP_RATIO = 0.035      # gap between cells, relative to the grid side
RADIUS_RATIO = 0.22    # corner radius, relative to the cell side
SPACING_CELLS = 0.9    # space between "MI" and the grid, in cell widths


def cell_color(ch, r, c, theme):
    if ch == "S":
        return ORANGE
    shades = theme["inactive"]
    return shades[(r * 3 + c * 5) % len(shades)]


class Layout:
    """All positions in one unit system: the grid is `side` units tall."""

    def __init__(self, side):
        self.side = side
        self.gap = side * GAP_RATIO
        self.cell = (side - (N - 1) * self.gap) / N
        pitch = self.cell + self.gap
        self.cap_top = S_ROWS[0] * pitch
        self.baseline = S_ROWS[1] * pitch + self.cell
        if not FONT_PATH.exists():
            sys.exit(f"font not found: {FONT_PATH} (set MIS_LOGO_FONT to an Arial Bold .ttf)")
        self.font = TTFont(FONT_PATH)
        self.glyphs = self.font.getGlyphSet()
        cmap = self.font.getBestCmap()
        self.names = [cmap[ord(ch)] for ch in TEXT]
        cap_height = getattr(self.font["OS/2"], "sCapHeight", 0)
        if not cap_height:  # older OS/2 tables: measure the "H" instead
            h_bounds = BoundsPen(self.glyphs)
            self.glyphs[cmap[ord("H")]].draw(h_bounds)
            cap_height = h_bounds.bounds[3]
        self.scale = (self.baseline - self.cap_top) / cap_height
        # place glyphs on their advance widths, then shift so the ink starts at x = 0
        bounds = BoundsPen(self.glyphs)
        x = 0.0
        self.offsets = []
        for name in self.names:
            self.offsets.append(x)
            self.glyphs[name].draw(TransformPen(bounds, (self.scale, 0, 0, -self.scale, x, self.baseline)))
            x += self.glyphs[name].width * self.scale
        x_min, _, x_max, _ = bounds.bounds
        self.text_shift = -x_min
        self.text_width = x_max - x_min
        self.grid_x = self.text_width + SPACING_CELLS * self.cell
        self.width = self.grid_x + side

    def glyph_transforms(self):
        for name, off in zip(self.names, self.offsets):
            yield name, (self.scale, 0, 0, -self.scale, off + self.text_shift, self.baseline)

    def cells(self):
        pitch = self.cell + self.gap
        for r, row in enumerate(PATTERN):
            for c, ch in enumerate(row):
                yield ch, r, c, self.grid_x + c * pitch, r * pitch


def svg(theme, side=100):
    lay = Layout(side)
    paths = []
    for name, tf in lay.glyph_transforms():
        pen = SVGPathPen(lay.glyphs)
        lay.glyphs[name].draw(TransformPen(pen, tf))
        paths.append(pen.getCommands())
    rects = [f'<rect x="{x:.2f}" y="{y:.2f}" width="{lay.cell:.2f}" height="{lay.cell:.2f}" '
             f'rx="{lay.cell * RADIUS_RATIO:.2f}" fill="#{cell_color(ch, r, c, theme)}"/>'
             for ch, r, c, x, y in lay.cells()]
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {lay.width:.2f} {side}" role="img" '
            f'aria-label="MIS logo">\n<title>MIS</title>\n'
            f'<path fill="#{theme["text"]}" fill-rule="nonzero" d="{" ".join(paths)}"/>\n'
            + "\n".join(rects) + "\n</svg>\n")


class PolygonPen(BasePen):
    """Flattens glyph outlines into polygons (one per contour) for raster drawing."""

    def __init__(self, glyph_set, steps=16):
        super().__init__(glyph_set)
        self.steps, self.contours, self.current = steps, [], []

    def _moveTo(self, pt):
        self.current = [pt]

    def _lineTo(self, pt):
        self.current.append(pt)

    def _curveToOne(self, p1, p2, p3):
        p0 = self.current[-1]
        for i in range(1, self.steps + 1):
            t = i / self.steps
            u = 1 - t
            self.current.append((u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
                                 u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1]))

    def _qCurveToOne(self, p1, p2):
        p0 = self.current[-1]
        for i in range(1, self.steps + 1):
            t = i / self.steps
            u = 1 - t
            self.current.append((u * u * p0[0] + 2 * u * t * p1[0] + t * t * p2[0],
                                 u * u * p0[1] + 2 * u * t * p1[1] + t * t * p2[1]))

    def _closePath(self):
        self.contours.append(self.current)
        self.current = []

    _endPath = _closePath


def png(theme, height=512, supersample=4):
    big = height * supersample
    lay = Layout(big)
    size = (round(lay.width), big)
    im = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(im)
    # glyphs: even-odd fill via XOR of contour masks, so counters stay open
    mask = Image.new("1", size, 0)
    for name, tf in lay.glyph_transforms():
        pen = PolygonPen(lay.glyphs)
        lay.glyphs[name].draw(TransformPen(pen, tf))
        for contour in pen.contours:
            m = Image.new("1", size, 0)
            ImageDraw.Draw(m).polygon(contour, fill=1)
            mask = ImageChops.logical_xor(mask, m)
    im.paste("#" + theme["text"], (0, 0), mask.convert("L"))
    for ch, r, c, x, y in lay.cells():
        draw.rounded_rectangle([x, y, x + lay.cell, y + lay.cell], radius=lay.cell * RADIUS_RATIO,
                               fill="#" + cell_color(ch, r, c, theme))
    return im.resize((round(size[0] / supersample), height), Image.LANCZOS)


def main(out):
    out.mkdir(parents=True, exist_ok=True)
    for name, theme in THEMES.items():
        (out / f"mis-logo-{name}.svg").write_text(svg(theme), encoding="utf8")
        png(theme).save(out / f"mis-logo-{name}.png")
    print(f"wrote logo variants to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "docs" / "assets" / "branding")
