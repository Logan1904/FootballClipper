"""Scrub a match video at speed, flag highlights with a hotkey, stitch them into a reel.

Playback advances by grabbing and discarding intermediate frames rather than seeking,
because non-keyframe seeks force the decoder to backtrack to the previous keyframe and
re-decode — that is what makes naive fast-forward stutter. Seeks are reserved for the
manual skip keys, where a one-off cost is fine.

Inputs:
    video       : str  - path to the match file (any container ffmpeg can read)
    --speed     : float - initial playback multiplier (default 4.0)
    --pre/--post: float - seconds of padding kept before/after each flag
    --from-flags: bool  - skip scrubbing, re-stitch from an existing <stem>_flags.txt

Outputs:
    <stem>_flags.txt      : one flag timestamp in seconds per line
    <stem>_highlights.mp4 : flags cut with padding, merged, concatenated

Author: Logan Sritharan (l.sritharan@astroscale.com)
"""

import argparse
import os
import subprocess
import sys
import tempfile
import time

import cv2

DEFAULT_SPEED = 4.0
DEFAULT_PRE = 5.0
DEFAULT_POST = 7.0

# waitKeyEx returns platform-specific extended codes for arrows; cover Windows and Linux,
# and expose WASD/RE letter alternates so the tool still works if none of them match.
LEFT_KEYS = {2424832, 65361, 81}
RIGHT_KEYS = {2555904, 65363, 83}
UP_KEYS = {2490368, 65362, 82}
DOWN_KEYS = {2621440, 65364, 84}
SHIFT_LEFT_KEYS = {2424832 | 0x10000, 65361 | 0x10000}
SHIFT_RIGHT_KEYS = {2555904 | 0x10000, 65363 | 0x10000}


def fmt_time(t):
    """Seconds to H:MM:SS for the HUD."""
    h, rem = divmod(int(t), 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}"


def scrub(path, speed=DEFAULT_SPEED):
    """Play the video with flagging controls and return the flagged timestamps.

    Inputs:
        path  : str   - video file path
        speed : float - initial playback multiplier
    Outputs:
        flags : list[float] - flag timestamps in seconds, ascending
    """
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        sys.exit(f"Could not open {path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    flags = []
    paused = False
    cur_frame = 0
    last_frame = None  # cached so a paused window still redraws

    win = "Highlight Clipper - F to flag, Q to finish"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)

    # Render at native fps regardless of speed; each rendered frame just represents
    # `speed` source frames, which is what makes it look like fast-forward not a slideshow.
    display_interval = 1.0 / fps
    last_display = time.time()

    while True:
        if not paused:
            step = max(1, int(round(speed)))
            for _ in range(step - 1):
                cap.grab()  # decodes without the YUV->BGR conversion, ~3-5x cheaper than read()
                cur_frame += 1
                if cur_frame >= total_frames:
                    break

            ret, frame = cap.read()
            if not ret:
                print("\nReached end of video.")
                break
            cur_frame += 1
            last_frame = frame

            wait = display_interval - (time.time() - last_display)
            if wait > 0:
                time.sleep(wait)
            last_display = time.time()
        else:
            if last_frame is None:
                continue
            frame = last_frame.copy()

        ts = cur_frame / fps
        hud = f"{fmt_time(ts)} / {fmt_time(duration)}   speed={speed:.1f}x   flags={len(flags)}"
        if paused:
            hud += "   [PAUSED]"
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
        cv2.putText(frame, hud, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow(win, frame)
        key = cv2.waitKeyEx(1)
        key_low = key & 0xFF if key != -1 else -1

        if key_low == ord('q'):
            break
        elif key_low == ord('f'):
            flags.append(ts)
            print(f"flagged {fmt_time(ts)}  (total {len(flags)})")
        elif key_low == ord(' '):
            paused = not paused
        elif key in SHIFT_RIGHT_KEYS or key_low == ord('r'):
            cur_frame = min(total_frames - 1, cur_frame + int(30 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        elif key in SHIFT_LEFT_KEYS or key_low == ord('e'):
            cur_frame = max(0, cur_frame - int(30 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        elif key in RIGHT_KEYS or key_low == ord('d'):
            cur_frame = min(total_frames - 1, cur_frame + int(5 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        elif key in LEFT_KEYS or key_low == ord('a'):
            cur_frame = max(0, cur_frame - int(5 * fps))
            cap.set(cv2.CAP_PROP_POS_FRAMES, cur_frame)
        elif key in UP_KEYS or key_low == ord('w'):
            speed = min(32.0, speed + 1.0)
        elif key in DOWN_KEYS or key_low == ord('s'):
            speed = max(1.0, speed - 1.0)
        elif key_low != -1 and ord('1') <= key_low <= ord('9'):
            n = key_low - ord('0')
            speed = float(n) if n < 9 else 16.0

    cap.release()
    cv2.destroyAllWindows()
    return sorted(flags)


def merge_windows(flags, pre, post):
    """Convert flags to padded (start, end) windows, merging any that overlap.

    Inputs:
        flags : list[float] - ascending flag timestamps, seconds
        pre   : float       - padding before each flag
        post  : float       - padding after each flag
    Outputs:
        windows : list[tuple[float, float]] - disjoint, ascending
    """
    windows = []
    for t in flags:
        start, end = max(0.0, t - pre), t + post
        if windows and start <= windows[-1][1]:
            windows[-1] = (windows[-1][0], max(windows[-1][1], end))
        else:
            windows.append((start, end))
    return windows


def run_ffmpeg(args):
    """Run ffmpeg, surfacing stderr on failure — silent exits here are near-impossible to debug."""
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        print(proc.stderr[-4000:], file=sys.stderr)
        sys.exit(f"ffmpeg failed (exit {proc.returncode})")


def stitch(video, windows, out_path):
    """Cut each window and concatenate them into a single reel.

    Windows are re-encoded rather than stream-copied so cuts land on the requested
    timestamp instead of the nearest preceding keyframe.

    Inputs:
        video    : str - source video path
        windows  : list[tuple[float, float]]
        out_path : str - destination mp4
    Outputs:
        None (writes out_path)
    """
    with tempfile.TemporaryDirectory() as tmp:
        parts = []
        for i, (start, end) in enumerate(windows):
            part = os.path.join(tmp, f"part_{i:03d}.mp4")
            run_ffmpeg([
                "ffmpeg", "-y", "-ss", f"{start:.3f}", "-i", video,
                "-t", f"{end - start:.3f}",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-c:a", "aac", part,
            ])
            parts.append(part)
            print(f"cut {i + 1}/{len(windows)}  {fmt_time(start)}-{fmt_time(end)}")

        listfile = os.path.join(tmp, "parts.txt")
        with open(listfile, "w") as f:
            for p in parts:
                f.write(f"file '{p}'\n")

        run_ffmpeg([
            "ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", listfile,
            "-c", "copy", out_path,
        ])
    print(f"\nwrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description="Scrub, flag and stitch football highlights.")
    ap.add_argument("video")
    ap.add_argument("--speed", type=float, default=DEFAULT_SPEED)
    ap.add_argument("--pre", type=float, default=DEFAULT_PRE)
    ap.add_argument("--post", type=float, default=DEFAULT_POST)
    ap.add_argument("--from-flags", action="store_true",
                    help="re-stitch from the saved flags file without scrubbing again")
    args = ap.parse_args()

    stem = os.path.splitext(args.video)[0]
    flags_path = f"{stem}_flags.txt"
    out_path = f"{stem}_highlights.mp4"

    if args.from_flags:
        if not os.path.exists(flags_path):
            sys.exit(f"No flags file at {flags_path}")
        with open(flags_path) as f:
            flags = sorted(float(line) for line in f if line.strip())
    else:
        print("F flag | SPACE pause | A/D or arrows +-5s | E/R or shift-arrows +-30s | "
              "W/S speed | 1-9 preset speed | Q finish")
        flags = scrub(args.video, args.speed)
        with open(flags_path, "w") as f:
            f.write("\n".join(f"{t:.3f}" for t in flags))
        print(f"saved {len(flags)} flags to {flags_path}")

    if not flags:
        sys.exit("No flags — nothing to stitch.")

    stitch(args.video, merge_windows(flags, args.pre, args.post), out_path)


if __name__ == "__main__":
    main()
