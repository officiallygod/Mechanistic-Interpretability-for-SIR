# MIS logo

The letters **MI** in Arial Bold, followed by a 7×7 activation map (like the ResNet-18 layer-4 feature map at 224 px input) whose lit cells form the **S**. "MI" is white (dark in the light variant), the S is orange (`#F26B1D`). The letters are stored as outlines, so the logo looks the same whether or not the font is installed.

| File | Use on |
|---|---|
| `mis-logo-dark.svg` / `.png` | dark backgrounds (slides, dark theme) |
| `mis-logo-light.svg` / `.png` | light backgrounds |

Use the SVG on the website and the PNG (512 px tall, transparent) in PowerPoint. The logo is about 2.1 times as wide as it is tall, so size it by height.

Do not edit these files by hand. Change `tools/make_logo.py` (the S pattern and colors are at the top) and run it. It needs `pillow` and `fonttools`, and reads Arial Bold from `C:/Windows/Fonts/arialbd.ttf`. On another system, set `MIS_LOGO_FONT` to the font file. The font is not committed because its license does not allow redistribution.

```
python tools/make_logo.py
```
