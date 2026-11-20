"""Publish every slide deck in templates/ to the docs site.

Copies each .pptx (except the master template) into <out_dir> and writes
<out_dir>/manifest.json, which docs/index.html renders as the "Slides" section.
Title and description come from the deck's document properties; the date is
the last commit that touched the file.

Usage: python build_slides_manifest.py <templates_dir> <out_dir>
"""
import json
import re
import shutil
import subprocess
import sys
import zipfile
from datetime import date
from pathlib import Path

MASTER_TEMPLATE = "Mechanistic_Interpretability_Templates.pptx"


def core_property(xml: str, tag: str) -> str:
    match = re.search(rf"<{tag}[^>]*>(.*?)</{tag}>", xml, re.S)
    if not match:
        return ""
    value = match.group(1).strip()
    for entity, char in (("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&apos;", "'"), ("&amp;", "&")):
        value = value.replace(entity, char)
    return value


def last_commit_date(path: Path) -> str:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%cs", "--", str(path)],
                             capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        out = ""
    return out or date.fromtimestamp(path.stat().st_mtime).isoformat()


def describe(path: Path) -> dict:
    with zipfile.ZipFile(path) as z:
        core = z.read("docProps/core.xml").decode("utf8") if "docProps/core.xml" in z.namelist() else ""
        slides = len(re.findall(r"<p:sldId\b", z.read("ppt/presentation.xml").decode("utf8")))
    title = core_property(core, "dc:title") or path.stem.replace("_", " ")
    return {
        "file": path.name,
        "title": title,
        "description": core_property(core, "dc:description"),
        "slides": slides,
        "updated": last_commit_date(path),
        "size_kb": round(path.stat().st_size / 1024),
    }


def main(src: Path, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    decks = []
    for path in sorted(src.glob("*.pptx")):
        if path.name == MASTER_TEMPLATE:
            continue
        decks.append(describe(path))
        shutil.copy2(path, out / path.name)
    decks.sort(key=lambda d: (d["updated"], d["file"]), reverse=True)
    (out / "manifest.json").write_text(json.dumps({"decks": decks}, indent=2), encoding="utf8")
    print(f"published {len(decks)} deck(s) to {out}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
