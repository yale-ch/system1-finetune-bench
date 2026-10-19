"""Render the animated SVG diagrams to MP4 and GIF (for social posts), or to still frames for checking.

    python blog/render_video.py SVG_DIR                       # every *.svg -> NAME.mp4 + NAME.gif
    python blog/render_video.py SVG_DIR --stills 5 10 15 20   # NAME-t05.png ... (layout check)

Each SVG is opened in headless Chromium (Playwright); its CSS animations are paused and stepped to each frame's
time, screenshotted, and encoded with ffmpeg: H.264 MP4 at 1800 px wide (30 fps) and a GIF at 900 px (15 fps).
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from playwright.sync_api import sync_playwright

SCALE = 1.5


def duration(svg: Path) -> float:
    m = re.search(r"animation:k\d+ ([\d.]+)s", svg.read_text())
    return float(m.group(1)) if m else 10.0


def frames(page, svg: Path, times: list[float], out: Path, prefix: str) -> list[Path]:
    page.goto(svg.resolve().as_uri())
    page.wait_for_timeout(300)
    paths = []
    for i, t in enumerate(times):
        page.evaluate("t => document.getAnimations().forEach(a => { a.pause(); a.currentTime = t; })", t * 1000)
        p = out / f"{prefix}{i:05d}.png"
        page.screenshot(path=str(p), clip={"x": 0, "y": 0, "width": 1200, "height": 760})
        paths.append(p)
    return paths


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("svg_dir")
    ap.add_argument("--only", nargs="*", help="names (without .svg) to render")
    ap.add_argument("--stills", nargs="*", type=float, help="seconds; write PNG stills instead of video")
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()
    d = Path(args.svg_dir)
    svgs = [p for p in sorted(d.glob("*.svg")) if not args.only or p.stem in args.only]
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1200, "height": 760}, device_scale_factor=SCALE)
        for svg in svgs:
            if args.stills:
                with tempfile.TemporaryDirectory() as tmp:
                    for t, p in zip(args.stills, frames(page, svg, args.stills, Path(tmp), "s")):
                        dest = d / "stills" / f"{svg.stem}-t{t:04.1f}.png"
                        dest.parent.mkdir(exist_ok=True)
                        shutil.move(p, dest)
                print(f"{svg.stem}: stills at {args.stills}")
                continue
            dur = duration(svg)
            n = int(dur * args.fps)
            with tempfile.TemporaryDirectory() as tmp:
                frames(page, svg, [i / args.fps for i in range(n)], Path(tmp), "f")
                src = f"{tmp}/f%05d.png"
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", src,
                                "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", "-preset", "slow",
                                "-movflags", "+faststart", str(d / f"{svg.stem}.mp4")], check=True)
                subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(args.fps), "-i", src,
                                "-vf", "fps=15,scale=900:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=128[p];[b][p]paletteuse=dither=bayer:bayer_scale=4",
                                str(d / f"{svg.stem}.gif")], check=True)
            mp4, gif = d / f"{svg.stem}.mp4", d / f"{svg.stem}.gif"
            print(f"{svg.stem}: {dur:g}s, {n} frames -> {mp4.name} ({mp4.stat().st_size / 1e6:.1f} MB), {gif.name} ({gif.stat().st_size / 1e6:.1f} MB)")
        browser.close()


if __name__ == "__main__":
    main()
