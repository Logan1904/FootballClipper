"""Capture the Mux HLS manifest behind a Veo share link and remux it to a local mp4.

The playback ID is injected by the page's JavaScript, so static scraping finds nothing —
a headless browser has to load the page and we intercept the manifest request it makes.
Segment URLs are signed with a short-lived expiry, so capture and download in one go
rather than caching the manifest.

Inputs:
    url        : str   - Veo share URL (https://live.veo.co/matches/<uuid>@<epoch_ms>/)
    -o/--out   : str   - output mp4 path (default <YYYYMMDD>_<uuid8>.mp4)
    --height   : int   - target vertical resolution, 0 for highest (default 0)
    --headed   : bool  - show the browser, for debugging capture failures
    --timeout  : float - seconds to wait for the manifest

Outputs:
    <out> : stream-copied match video

Author: Logan Sritharan (l.sritharan@astroscale.com)
"""

import argparse
import re
import subprocess
import sys
import time

from playwright.sync_api import sync_playwright

MANIFEST_RE = re.compile(r"https://stream\.mux\.com/[\w-]+\.m3u8")
# The page's own API response carries the playback ID before the player touches it,
# so we can recover the manifest even if playback never starts.
ID_RE = re.compile(r"stream\.mux\.com/([\w-]+)\.m3u8|playback[_-]?id\"\s*:\s*\"([\w-]+)\"")
URL_RE = re.compile(r"matches/([\w-]+)@(\d+)")


def capture_manifest(url, headed=False, timeout=45.0):
    """Load the Veo page and recover the Mux master playlist URL.

    Watches network requests for the manifest directly, and falls back to scanning
    Veo's own JSON responses for the playback ID — some browser builds lack H.264,
    in which case the player never requests the manifest at all.

    Inputs:
        url     : str
        headed  : bool
        timeout : float - seconds
    Outputs:
        manifest : str - master .m3u8 URL
    """
    found = []
    seen = []  # every URL, kept for the failure message

    def on_response(r):
        seen.append(r.url)
        if MANIFEST_RE.fullmatch(r.url.split("?")[0]):
            found.append(r.url.split("?")[0])
            return
        # Only worth reading bodies of the page's own API calls.
        ctype = r.headers.get("content-type") or ""
        if "veo.co" in r.url and ("json" in ctype or "text" in ctype):
            try:
                m = ID_RE.search(r.text())
            except Exception:
                return  # body already consumed, or binary
            if m:
                found.append(f"https://stream.mux.com/{m.group(1) or m.group(2)}.m3u8")

    with sync_playwright() as p:
        # channel="msedge" uses the installed branded build, which has H.264/AAC;
        # Playwright's bundled Chromium does not, and hls.js won't start without them.
        browser = p.chromium.launch(channel="msedge", headless=not headed)
        page = browser.new_context().new_page()
        page.on("response", on_response)

        page.goto(url, wait_until="domcontentloaded")
        try:
            page.click("video", timeout=5000)  # some share pages defer player init
        except Exception:
            pass

        deadline = time.time() + timeout
        while not found and time.time() < deadline:
            page.wait_for_timeout(500)
        browser.close()

    if not found:
        print("\n".join(u for u in seen if "veo.co" in u or "mux" in u), file=sys.stderr)
        sys.exit("No manifest found — see candidate URLs above.")
    return found[0]


def ffprobe(manifest, entries, select=None):
    """Run ffprobe against the manifest and return its whitespace-split output.

    Inputs:
        manifest : str
        entries  : str - value for -show_entries
        select   : str | None - value for -select_streams
    Outputs:
        fields : list[str] - empty if the probe failed
    """
    cmd = ["ffprobe", "-v", "error"]
    if select:
        cmd += ["-select_streams", select]
    cmd += ["-show_entries", entries, "-of", "default=nw=1:nk=1", manifest]
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=60).stdout.split()
    except Exception:
        return []


def pick_program(manifest, target_height):
    """Program index for the requested rendition height.

    target_height of 0 selects the tallest rendition; otherwise the nearest match.
    Playlist order is arbitrary — a given index is 720p in one match and 480p in the
    next — so the rendition must be chosen by resolution, not position.

    Inputs:
        manifest      : str
        target_height : int - 0 for the highest available
    Outputs:
        index : int | None - None if probing failed
    """
    heights = [int(h) for h in ffprobe(manifest, "stream=height", select="v")]
    if not heights:
        return None
    if target_height == 0:
        return max(range(len(heights)), key=lambda i: heights[i])
    return min(range(len(heights)), key=lambda i: abs(heights[i] - target_height))


def download(manifest, out, program):
    """Stream-copy the chosen rendition to mp4, printing a single-line progress readout.

    ffmpeg's stderr is unusable here — it logs every segment URL — so it is silenced and
    progress is parsed from the machine-readable -progress stream instead.

    Inputs:
        manifest : str
        out      : str
        program  : int - program index from pick_program
    Outputs:
        None (writes out)
    """
    dur = ffprobe(manifest, "format=duration")
    total = float(dur[0]) if dur else None

    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-nostats",
        "-progress", "pipe:1",
        "-i", manifest,
        "-map", f"0:p:{program}",
        "-c", "copy",
        "-bsf:a", "aac_adtstoasc",  # ADTS AAC from TS segments -> the form mp4 expects
        out,
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True, bufsize=1)

    for line in proc.stdout:
        key, _, val = line.strip().partition("=")
        # out_time_us is unambiguous; out_time_ms is a misnomer and also microseconds.
        if key == "out_time_us" and val.isdigit():
            secs = int(val) / 1e6
            pct = f" ({100 * secs / total:.1f}%)" if total else ""
            print(f"\r  {secs / 60:6.1f} min written{pct}", end="", flush=True)

    if proc.wait() != 0:
        sys.exit(f"\nffmpeg failed (exit {proc.returncode})")
    print(f"\nwrote {out}")


def default_name(url):
    """Filename as <YYYYMMDD>_<uuid8>.mp4, so a folder of matches sorts chronologically.

    Inputs:
        url : str
    Outputs:
        name : str
    """
    m = URL_RE.search(url)
    if not m:
        return "match.mp4"
    date = time.strftime("%Y%m%d", time.localtime(int(m.group(2)) / 1000))
    return f"{date}_{m.group(1)[:8]}.mp4"


def main():
    ap = argparse.ArgumentParser(description="Download a Veo match via its Mux manifest.")
    ap.add_argument("url")
    ap.add_argument("-o", "--out")
    ap.add_argument("--height", type=int, default=0,
                    help="target vertical resolution; 0 (default) picks the highest "
                         "available. 720 decodes far faster when scrubbing at 8x")
    ap.add_argument("--headed", action="store_true")
    ap.add_argument("--timeout", type=float, default=45.0)
    args = ap.parse_args()

    out = args.out or default_name(args.url)

    manifest = capture_manifest(args.url, args.headed, args.timeout)
    print(f"manifest: {manifest}")

    program = pick_program(manifest, args.height)
    if program is None:
        sys.exit("Could not probe renditions — the manifest may have expired.")

    heights = ffprobe(manifest, "stream=height", select="v")
    print(f"rendition: {heights[program]}p")

    download(manifest, out, program)


if __name__ == "__main__":
    main()
