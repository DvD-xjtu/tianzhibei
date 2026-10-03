"""Build deduplicated aircraft contact sheets from CORS-ADD OBB patches."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("extracted", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--site", action="append", required=True)
    ap.add_argument("--min-side", type=float, default=20)
    ap.add_argument("--columns", type=int, default=12)
    ap.add_argument("--rows", type=int, default=10)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    tile, label_h = 120, 20
    entries, pages = [], []
    seen = set()
    for split in ("train2017", "val2017"):
        imgdir = args.extracted / "images" / split
        labeldir = args.extracted / "dota_labels" / split
        if not imgdir.exists():
            continue
        for imgpath in sorted(imgdir.glob("*.tif")):
            source = imgpath.stem.split("__")[0]
            if not any(s.lower() in source.lower() for s in args.site):
                continue
            parts = imgpath.stem.split("___")
            tile_y = float(parts[-1])
            tile_x = float(parts[-2].split("__")[-1])
            labelpath = labeldir / (imgpath.stem + ".txt")
            if not labelpath.exists():
                continue
            try:
                im = Image.open(imgpath).convert("RGB")
            except Exception as exc:
                print(f"Skipping unreadable image {imgpath}: {exc}")
                continue
            for line_no, line in enumerate(labelpath.read_text().splitlines(), 1):
                a = line.split()
                if len(a) < 9 or a[8] != "plane":
                    continue
                xy = [float(x) for x in a[:8]]
                xs, ys = xy[0::2], xy[1::2]
                x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
                w, h = x1 - x0, y1 - y0
                if min(w, h) < args.min_side:
                    continue
                x, y = (x0 + x1) / 2, (y0 + y1) / 2
                key = (source, round((tile_x + x) / 8), round((tile_y + y) / 8))
                if key in seen:
                    continue
                seen.add(key)
                side = max(w, h) * 1.4
                crop = im.crop((x - side / 2, y - side / 2, x + side / 2, y + side / 2))
                crop.thumbnail((tile, tile), Image.Resampling.LANCZOS)
                idx = len(entries)
                entries.append({"index": idx, "image": str(imgpath), "label": str(labelpath),
                                "line": line_no, "source": source, "global_xy": [tile_x + x, tile_y + y],
                                "bbox_xywh_pixels": [x, y, w, h]})
                page_id = idx // (args.columns * args.rows)
                while len(pages) <= page_id:
                    pages.append(Image.new("RGB", (args.columns * tile, args.rows * (tile + label_h)), "white"))
                slot = idx % (args.columns * args.rows)
                px, py = slot % args.columns * tile, slot // args.columns * (tile + label_h)
                pages[page_id].paste(crop, (px + (tile - crop.width) // 2, py + (tile - crop.height) // 2))
                ImageDraw.Draw(pages[page_id]).text((px + 2, py + tile + 2), f"{idx} {int(w)}x{int(h)}", fill="black")
    (args.output / "candidates.json").write_text(json.dumps(entries, indent=2))
    for i, page in enumerate(pages):
        page.save(args.output / f"contact_{i:02d}.jpg", quality=91)
    print(json.dumps({"candidate_boxes": len(entries), "pages": len(pages)}))


if __name__ == "__main__":
    main()
