"""
Log the license plates of cars that come into a driveway, fully offline.

Camera mode watches the feed with cheap motion detection, runs the plate reader only while
something is moving, and when the motion stops logs each plate it read consistently:
time, plate, confidence and a snapshot. The same plate isn't logged again within --cooldown seconds.
If nothing has been scanned for --scan-every seconds (default 5 minutes), it scans one frame anyway.

Image mode reads plates from still photos, one photo per sighting, for testing without a camera.

Logs go to backend/logs/plates.db (SQLite) and snapshots to backend/logs/snaps/, wherever it's run from.
Snapshot paths in the database are relative to the logs folder (e.g. snaps/20261026-173803-WPG6392.jpg).

Usage (from /home/pi/alpr/backend):
    venv/bin/python driveway_logger.py --images ../img/              # test on photos
    venv/bin/python driveway_logger.py --camera 0                    # USB webcam
    venv/bin/python driveway_logger.py --camera picam                # Pi Camera Module (needs picamera2)
    venv/bin/python driveway_logger.py --camera clip.mp4             # a recorded video
    venv/bin/python driveway_logger.py --camera 0 --zone 0.2,0.4,0.9,1.0   # watch part of the frame
"""

import argparse
import logging
import sqlite3
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from fast_alpr import ALPR  # noqa: E402

from read_plate import crop, plate_conf, upscale  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
DEFAULT_OUT = Path(__file__).resolve().parent / "logs"

# Readings that differ by at most this many characters are treated as the same plate (OCR noise).
MAX_PLATE_DIFF = 2


def plate_distance(a: str, b: str) -> int:
    """Edit distance: the number of characters inserted, deleted or changed to turn a into b."""
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def same_plate(a: str, b: str) -> bool:
    return plate_distance(a, b) <= MAX_PLATE_DIFF


class MotionDetector:
    """Background subtraction on a small grayscale copy of each frame."""

    def __init__(self, zone=None, min_area=0.01, warmup=15):
        self.zone = zone  # (x1, y1, x2, y2) as fractions of the frame
        self.min_area = min_area  # fraction of the zone that must change
        self.warmup = warmup
        self.frames = 0
        self.bg = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=32, detectShadows=False)

    def __call__(self, frame: np.ndarray) -> bool:
        small = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
        if self.zone:
            h, w = small.shape[:2]
            x1, y1, x2, y2 = self.zone
            small = small[int(y1 * h):int(y2 * h), int(x1 * w):int(x2 * w)]
        gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (5, 5), 0)
        mask = self.bg.apply(gray)
        self.frames += 1
        if self.frames <= self.warmup:
            return False
        return cv2.countNonZero(mask) / mask.size >= self.min_area


def read_plates(alpr: ALPR, frame: np.ndarray):
    """Return [(text, confidence, bbox)] for every plate in the frame."""
    plates = []
    for r in alpr.predict(frame):
        if not (r.ocr and r.ocr.text):
            continue
        b = r.detection.bounding_box
        text, conf = r.ocr.text, plate_conf(r)
        # Small plates: zoom in and re-read, keeping whichever read is more confident.
        if b.x2 - b.x1 < 150:
            zoom = upscale(crop(frame, b.x1, b.y1, b.x2, b.y2, 1.0, 1.0), 900)
            for z in alpr.predict(zoom):
                if z.ocr and z.ocr.text and plate_conf(z) > conf:
                    text, conf = z.ocr.text, plate_conf(z)
        plates.append((text, conf, (b.x1, b.y1, b.x2, b.y2)))
    return plates


class Event:
    """Plate reads collected while one car is in view. OCR variants of the same plate are grouped."""

    def __init__(self):
        self.groups = []

    def add(self, frame: np.ndarray, plates):
        for text, conf, bbox in plates:
            group = next((g for g in self.groups if any(same_plate(text, t) for t in g["votes"])), None)
            if group is None:
                group = {"votes": {}, "reads": 0, "best": (0.0, None, None)}
                self.groups.append(group)
            group["votes"][text] = group["votes"].get(text, 0.0) + conf
            group["reads"] += 1
            if conf > group["best"][0]:
                group["best"] = (conf, frame, bbox)

    def _merge_groups(self):
        """Combine groups that turned out to be the same plate (a later read can bridge two groups)."""
        merged = []
        for g in self.groups:
            target = next((m for m in merged if any(
                same_plate(a, b) for a in g["votes"] for b in m["votes"])), None)
            if target is None:
                merged.append(g)
                continue
            for text, total in g["votes"].items():
                target["votes"][text] = target["votes"].get(text, 0.0) + total
            target["reads"] += g["reads"]
            if g["best"][0] > target["best"][0]:
                target["best"] = g["best"]
        self.groups = merged

    def results(self):
        """Yield (plate, best_confidence, reads, frame, bbox) per plate; the plate text is the
        variant with the highest total confidence across reads."""
        self._merge_groups()
        for g in self.groups:
            conf, frame, bbox = g["best"]
            yield max(g["votes"], key=g["votes"].get), conf, g["reads"], frame, bbox


SCHEMA = """
-- One row per visit: time is when the car was first logged, last_seen the latest scan that read it.
CREATE TABLE IF NOT EXISTS sightings (id INTEGER PRIMARY KEY, time TEXT, plate TEXT,
    confidence REAL, reads INTEGER, snapshot TEXT, source TEXT, last_seen TEXT);

-- Logger state, e.g. last_periodic_scan: when the last no-motion scan of the driveway started.
CREATE TABLE IF NOT EXISTS status (key TEXT PRIMARY KEY, value TEXT);

-- Vehicles you know, filled in by hand, to cross-reference against sightings by plate.
-- Plates are stored like the logger writes them: no dashes or spaces (e.g. WPG6392).
CREATE TABLE IF NOT EXISTS known_vehicles (
    id INTEGER PRIMARY KEY,
    plate TEXT NOT NULL UNIQUE COLLATE NOCASE,
    state TEXT,                          -- two-letter code, e.g. VA
    year INTEGER,
    make TEXT,
    model TEXT,
    color TEXT,
    owner TEXT,
    relationship TEXT,                   -- e.g. resident, family, friend, service
    notes TEXT,
    added TEXT DEFAULT (datetime('now', 'localtime'))
);
"""


class Logger:
    def __init__(self, out_dir: Path, cooldown: float):
        self.out_dir = out_dir
        self.snaps = out_dir / "snaps"
        self.snaps.mkdir(parents=True, exist_ok=True)
        self.db = sqlite3.connect(out_dir / "plates.db")
        init_db(self.db)
        self.cooldown = cooldown
        self.last_seen = {}  # logged plate -> time it was last seen (logged or skipped)

    def log(self, plate, conf, reads, frame, bbox, source, now, window=None):
        window = window or self.cooldown
        recent = next((p for p, t in self.last_seen.items()
                       if now - t < window and same_plate(plate, p)), None)
        if recent is not None:
            # Restart the timer so a car idling in the driveway isn't logged again when it runs out,
            # and record that it's still here.
            self.last_seen[recent] = now
            self.db.execute(
                "UPDATE sightings SET last_seen = ? WHERE id = (SELECT max(id) FROM sightings WHERE plate = ?)",
                (datetime.now().isoformat(timespec="seconds"), recent),
            )
            self.db.commit()
            print(f"  (skipped {plate}: same car as {recent}, seen in the last {window:.0f}s)")
            return
        self.last_seen[plate] = now
        stamp = datetime.now()

        snap = frame.copy()
        x1, y1, x2, y2 = bbox
        cv2.rectangle(snap, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(snap, plate, (x1, max(20, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        snap = cv2.resize(snap, None, fx=min(1.0, 1280 / snap.shape[1]), fy=min(1.0, 1280 / snap.shape[1]))
        path = self.snaps / f"{stamp:%Y%m%d-%H%M%S}-{plate}.jpg"
        cv2.imwrite(str(path), snap)
        # Also keep a close-up of the plate so the state can be read by eye.
        cv2.imwrite(str(path.with_name(path.stem + "-plate.jpg")),
                    upscale(crop(frame, x1, y1, x2, y2, 0.3, 0.6), 400))

        stamp_text = stamp.isoformat(timespec="seconds")
        self.db.execute(
            "INSERT INTO sightings (time, plate, confidence, reads, snapshot, source, last_seen)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (stamp_text, plate, round(conf, 3), reads, str(path.relative_to(self.out_dir)), source, stamp_text),
        )
        self.db.commit()
        print(f"{stamp:%Y-%m-%d %H:%M:%S}  {plate:<10} conf {conf:.2f}  reads {reads}  -> {path}")

    def mark_periodic_scan(self):
        """Record that a periodic scan is starting; cars it reads get a later last_seen than this."""
        self.db.execute(
            "INSERT OR REPLACE INTO status (key, value) VALUES ('last_periodic_scan', ?)",
            (datetime.now().isoformat(timespec="seconds"),),
        )
        self.db.commit()


def init_db(db: sqlite3.Connection):
    """Create the tables, and add last_seen to a sightings table made before it existed."""
    db.executescript(SCHEMA)
    if "last_seen" not in [c[1] for c in db.execute("PRAGMA table_info(sightings)")]:
        db.execute("ALTER TABLE sightings ADD COLUMN last_seen TEXT")
        db.execute("UPDATE sightings SET last_seen = time")
    db.commit()


def finish_event(event: Event, logger: Logger, min_conf: float, min_reads: int, source: str, now: float,
                 window=None):
    found = False
    for plate, conf, reads, frame, bbox in event.results():
        if conf >= min_conf and reads >= min_reads:
            logger.log(plate, conf, reads, frame, bbox, source, now, window)
            found = True
        else:
            print(f"  (ignored {plate}: conf {conf:.2f}, reads {reads})")
    if not found:
        print(f"  no plate logged for {source}")


def run_images(paths, alpr, logger, args):
    files = []
    for p in map(Path, paths):
        files += sorted(f for f in p.iterdir() if f.suffix.lower() in IMAGE_EXTS) if p.is_dir() else [p]
    for f in files:
        frame = cv2.imread(str(f))
        if frame is None:
            print(f"Could not read image: {f}", file=sys.stderr)
            continue
        t = time.perf_counter()
        event = Event()
        plates = read_plates(alpr, frame)
        if args.debug:
            print(f"  reads: {[(p, round(c, 2)) for p, c, _ in plates]}")
        event.add(frame, plates)
        print(f"{f}: read in {time.perf_counter() - t:.1f}s")
        finish_event(event, logger, args.min_conf, 1, str(f), time.time())


def open_camera(spec: str):
    """Return a read() function giving (BGR frame, seconds) or (None, None) when the source ends.

    Seconds are wall-clock time for live sources and the frame's position for video files, so a
    file processed slower than real time still splits sightings the way the recording happened."""
    if spec == "picam":
        from picamera2 import Picamera2  # apt install python3-picamera2; venv needs --system-site-packages

        cam = Picamera2()
        cam.configure(cam.create_video_configuration(main={"size": (1920, 1080), "format": "RGB888"}))
        cam.start()
        return lambda: (cam.capture_array(), time.time())  # picamera2's RGB888 is BGR byte order

    cap = cv2.VideoCapture(int(spec) if spec.isdigit() else spec)
    if not cap.isOpened():
        sys.exit(f"Could not open camera/video: {spec}")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # always get a fresh frame after a slow plate read
    is_file = Path(spec).is_file()

    def read():
        ok, frame = cap.read()
        if not ok:
            return None, None
        return frame, cap.get(cv2.CAP_PROP_POS_MSEC) / 1000 if is_file else time.time()

    return read


def run_camera(spec, alpr, logger, args):
    read = open_camera(spec)
    zone = tuple(float(v) for v in args.zone.split(",")) if args.zone else None
    motion = MotionDetector(zone=zone, min_area=args.min_motion)
    event, last_motion, started = None, 0.0, 0.0
    last_scan = float("-inf")  # so the first frame gets a scan, catching cars already parked
    print(f"Watching {spec} (Ctrl+C to stop)")

    now = 0.0
    while True:
        frame, now = read()
        if frame is None:
            break
        if motion(frame):
            last_motion = now
            if event is None:
                event, started = Event(), now
                print(f"{datetime.now():%H:%M:%S}  motion")

        if event is None and args.scan_every and now - last_scan >= args.scan_every:
            # No scan for a while: read this one frame even without motion. A car parked since the
            # last scan is logged. A car seen at the previous scan is still parked, so it's skipped
            # even if the cooldown is no longer than the time between scans.
            print(f"{datetime.now():%H:%M:%S}  periodic scan")
            last_scan = now
            logger.mark_periodic_scan()
            single = Event()
            single.add(frame, read_plates(alpr, frame))
            finish_event(single, logger, args.min_conf, 1, f"camera {spec}", now,
                         window=max(args.cooldown, args.scan_every + 60))

        if event is not None:
            last_scan = now
            plates = read_plates(alpr, frame)
            if args.debug:
                print(f"  t={now:.2f}s reads: {[(p, round(c, 2)) for p, c, _ in plates]}")
            event.add(frame, plates)
            if now - last_motion > args.idle or now - started > args.max_event:
                finish_event(event, logger, args.min_conf, args.min_reads, f"camera {spec}", now)
                event = None

    if event is not None:
        finish_event(event, logger, args.min_conf, args.min_reads, f"camera {spec}", now)


def main() -> int:
    ap = argparse.ArgumentParser(description="Log license plates of cars entering a driveway (offline).")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--images", nargs="+", metavar="PATH", help="photos or folders of photos to test on")
    src.add_argument("--camera", metavar="SOURCE", help="webcam index (0), 'picam', or a video file/stream URL")
    ap.add_argument("--out", default=str(DEFAULT_OUT), help=f"folder for plates.db and snapshots (default: {DEFAULT_OUT})")
    ap.add_argument("--min-conf", type=float, default=0.8, help="minimum plate read confidence to log")
    ap.add_argument("--min-reads", type=int, default=2, help="camera: reads of a plate needed to log it")
    ap.add_argument("--cooldown", type=float, default=300, help="seconds before the same plate is logged again")
    ap.add_argument("--zone", help="camera: watch only x1,y1,x2,y2 (fractions of the frame) for motion")
    ap.add_argument("--min-motion", type=float, default=0.01, help="camera: fraction of the zone that must change")
    ap.add_argument("--idle", type=float, default=3, help="camera: seconds without motion that end a sighting")
    ap.add_argument("--max-event", type=float, default=30, help="camera: longest a sighting can last, seconds")
    ap.add_argument("--scan-every", type=float, default=300,
                    help="camera: scan even without motion if nothing was scanned for this many seconds (0 = off)")
    ap.add_argument("--debug", action="store_true", help="print every plate reading")
    args = ap.parse_args()

    alpr = ALPR()
    logger = Logger(Path(args.out), args.cooldown)
    try:
        if args.images:
            run_images(args.images, alpr, logger, args)
        else:
            run_camera(args.camera, alpr, logger, args)
    except KeyboardInterrupt:
        print("\nStopped.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
