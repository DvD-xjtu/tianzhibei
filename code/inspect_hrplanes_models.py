"""Make source-linked contact sheets for manual model screening in HRPlanesv2."""

import argparse
import io
import json
import math
import zipfile
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("archive", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--columns", type=int, default=12)
    ap.add_argument("--rows", type=int, default=10)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tile, label_h = 120, 20
    entries = []
    pages = []
    with zipfile.ZipFile(args.archive) as z:
        labels = sorted(n for n in z.namelist() if n.endswith(".txt"))
        for label in labels:
            image_name = label[:-4] + ".jpg"
            im = Image.open(io.BytesIO(z.read(image_name))).convert("RGB")
            width, height = im.size
            for i, line in enumerate(z.read(label).decode().splitlines()):
                if not line.strip():
                    continue
                cls, cx, cy, bw, bh = map(float, line.split()[:5])
                if cls != 0:
                    raise ValueError((label, line))
                box_w, box_h = bw * width, bh * height
                x, y = cx * width, cy * height
                side = max(box_w, box_h) * 1.30
                crop = im.crop((x - side / 2, y - side / 2, x + side / 2, y + side / 2))
                crop.thumbnail((tile, tile), Image.Resampling.LANCZOS)
                idx = len(entries)
                entries.append({"index": idx, "image": image_name, "label": label,
                                "line": i + 1, "bbox_xywh_pixels": [x, y, box_w, box_h]})
                page_id = idx // (args.columns * args.rows)
                while len(pages) <= page_id:
                    pages.append(Image.new("RGB", (args.columns * tile, args.rows * (tile + label_h)), "white"))
                slot = idx % (args.columns * args.rows)
                px, py = slot % args.columns * tile, slot // args.columns * (tile + label_h)
                pages[page_id].paste(crop, (px + (tile - crop.width) // 2, py + (tile - crop.height) // 2))
                draw = ImageDraw.Draw(pages[page_id])
                draw.text((px + 2, py + tile + 2), f"{idx} {int(box_w)}x{int(box_h)}", fill="black")
    (args.output / "candidates.json").write_text(json.dumps(entries, indent=2))
    for i, page in enumerate(pages):
        page.save(args.output / f"contact_{i:02d}.jpg", quality=90)
    print(json.dumps({"images": len(labels), "aircraft_boxes": len(entries), "pages": len(pages)}))


if __name__ == "__main__":
    main()
