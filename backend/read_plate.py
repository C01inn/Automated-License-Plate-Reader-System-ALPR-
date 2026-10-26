"""
Read a US license plate (state + number) from a photo of a car, fully offline.

Pipeline (all local):
  1. fast-alpr: YOLOv9 plate detector + CCT plate OCR (ONNX models, cached after first download)
  2. Small plates are cropped, upscaled, and re-read for better accuracy
  3. EasyOCR reads the small text on the plate; it is fuzzy-matched against
     state names, abbreviations and common slogans to determine the state

Usage:
    python read_plate.py path/to/car.jpg [--debug]
"""

import argparse
import difflib
import logging
import re
import statistics
import sys
import warnings

import cv2
import numpy as np

warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

from fast_alpr import ALPR  # noqa: E402

STATES = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "DC": "District of Columbia",
    "FL": "Florida", "GA": "Georgia", "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois",
    "IN": "Indiana", "IA": "Iowa", "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana",
    "ME": "Maine", "MD": "Maryland", "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota",
    "MS": "Mississippi", "MO": "Missouri", "MT": "Montana", "NE": "Nebraska", "NV": "Nevada",
    "NH": "New Hampshire", "NJ": "New Jersey", "NM": "New Mexico", "NY": "New York",
    "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio", "OK": "Oklahoma", "OR": "Oregon",
    "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina", "SD": "South Dakota",
    "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont", "VA": "Virginia",
    "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
}

# Slogans / words commonly printed on plates, mapped to their state code.
SLOGANS = {
    "virginia is for lovers": "VA", "sunshine state": "FL", "empire state": "NY",
    "garden state": "NJ", "land of lincoln": "IL", "great lakes": "MI", "pure michigan": "MI",
    "lone star": "TX", "live free or die": "NH", "keystone state": "PA", "visitpa": "PA",
    "first in flight": "NC", "first in freedom": "NC", "grand canyon state": "AZ",
    "famous potatoes": "ID", "show me state": "MO", "land of enchantment": "NM",
    "the first state": "DE", "ocean state": "RI", "big sky": "MT", "peach state": "GA",
    "volunteer state": "TN", "aloha state": "HI", "evergreen state": "WA", "beehive state": "UT",
    "constitution state": "CT", "green mountain": "VT", "wild wonderful": "WV",
    "sportsman's paradise": "LA", "heart of dixie": "AL", "sweet home alabama": "AL",
    "the natural state": "AR", "colorful colorado": "CO", "crossroads of america": "IN",
    "bluegrass state": "KY", "vacationland": "ME", "10,000 lakes": "MN", "explore minnesota": "MN",
    "the good life": "NE", "the silver state": "NV", "peace garden": "ND", "birthplace of aviation": "OH",
    "native america": "OK", "dairyland": "WI", "americas dairyland": "WI", "great faces": "SD",
    "the last frontier": "AK", "north to the future": "AK", "taxation without representation": "DC",
    "end taxation": "DC", "spirit of america": "MA", "treasure state": "MT", "palmetto": "SC",
    "while i breathe i hope": "SC", "farmers": "IA", "ohio": "OH", "seek wisdom": "IA",
}


def plate_conf(result) -> float:
    c = result.ocr.confidence if result.ocr else 0.0
    return statistics.mean(c) if isinstance(c, list) and c else float(c or 0.0)


def crop(img: np.ndarray, x1: int, y1: int, x2: int, y2: int, pad_x: float, pad_y: float):
    """Crop a box from img with fractional padding on each side."""
    w, h = x2 - x1, y2 - y1
    X1, Y1 = max(0, int(x1 - w * pad_x)), max(0, int(y1 - h * pad_y))
    X2, Y2 = min(img.shape[1], int(x2 + w * pad_x)), min(img.shape[0], int(y2 + h * pad_y))
    return img[Y1:Y2, X1:X2]


def upscale(img: np.ndarray, min_width: int) -> np.ndarray:
    if img.shape[1] >= min_width:
        return img
    f = min_width / img.shape[1]
    return cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)


def read_plate_number(alpr: ALPR, img: np.ndarray):
    """Detect the best plate and return (bbox, text, confidence)."""
    results = [r for r in alpr.predict(img) if r.ocr and r.ocr.text]
    if not results:
        return None
    best = max(results, key=lambda r: r.detection.confidence * plate_conf(r))
    b = best.detection.bounding_box
    candidates = [(best.ocr.text, plate_conf(best))]

    # Small plates: zoom in around the detection and re-run detection + OCR on the enlarged crop.
    zoom = upscale(crop(img, b.x1, b.y1, b.x2, b.y2, 1.0, 1.0), 900)
    for r in alpr.predict(zoom):
        if r.ocr and r.ocr.text:
            candidates.append((r.ocr.text, plate_conf(r)))

    text, conf = max(candidates, key=lambda c: c[1])
    return (b.x1, b.y1, b.x2, b.y2), text, conf


def normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9, ']", "", s.lower()).strip()


def match_state(texts: list[str]):
    """Fuzzy-match OCR'd text fragments to a US state. Returns (code, score)."""
    best_code, best_score = None, 0.0
    names = {normalize(n): c for c, n in STATES.items()}
    joined = normalize(" ".join(texts))

    # Slogans can span several OCR fragments, so check them against the joined text.
    for slogan, code in SLOGANS.items():
        if slogan in joined:
            score = 0.9
            if score > best_score:
                best_code, best_score = code, score

    for t in texts:
        t = normalize(t)
        if len(t) < 4:
            continue
        for name, code in names.items():
            # Exact substring match (e.g. "VIRGINIA" inside "VIRGINIA 26") wins outright,
            # with longer names preferred so "west virginia" beats "virginia".
            if name in t:
                score = 1.0 + len(name) / 100
            else:
                score = difflib.SequenceMatcher(None, t, name).ratio()
            if score > best_score:
                best_code, best_score = code, score

    return (best_code, best_score) if best_score >= 0.75 else (None, best_score)


def read_state(reader, img: np.ndarray, bbox, plate_text: str):
    x1, y1, x2, y2 = bbox
    # Pad generously: state names/slogans often sit at the very top/bottom edge of the plate,
    # which the detector box sometimes clips.
    region = upscale(crop(img, x1, y1, x2, y2, 0.15, 0.45), 900)
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # State names and slogans live in the top/bottom bands; OCR those separately and
    # enlarged, since the small text is easily missed next to the big plate characters.
    h = gray.shape[0]
    views = [gray, upscale(gray[: int(h * 0.45)], 1400), upscale(gray[int(h * 0.6):], 1400)]
    texts = []
    for view in views:
        texts += [t for _, t, conf in reader.readtext(view) if conf > 0.05]

    code, score = match_state(texts)
    if code is None:
        # Last resort: one more pass with a sharpened, color version of the whole plate.
        sharp = cv2.filter2D(region, -1, np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
        texts += [t for _, t, conf in reader.readtext(sharp) if conf > 0.05]

    # Drop the fragment that is just the plate number itself.
    plate_norm = normalize(plate_text)
    texts = [t for t in texts if normalize(t).replace(" ", "") != plate_norm]
    code, _ = match_state(texts)
    return code, texts


def main() -> int:
    ap = argparse.ArgumentParser(description="Read a US license plate's state and number (offline).")
    ap.add_argument("image", help="path to the car image")
    ap.add_argument("--debug", action="store_true", help="print raw OCR text used for the state")
    ap.add_argument("--cpu", action="store_true", help="force EasyOCR onto the CPU")
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        print(f"Could not read image: {args.image}", file=sys.stderr)
        return 1

    alpr = ALPR()
    found = read_plate_number(alpr, img)
    if not found:
        print("No license plate found.")
        return 2
    bbox, number, conf = found

    import easyocr  # imported lazily; it pulls in torch which is slow to load
    import torch

    reader = easyocr.Reader(["en"], gpu=torch.cuda.is_available() and not args.cpu, verbose=False)
    code, texts = read_state(reader, img, bbox, number)

    state = f"{STATES[code]} ({code})" if code else "Unknown"
    print(f"State:  {state}")
    print(f"Number: {number}")
    if args.debug:
        print(f"[debug] plate confidence {conf:.2f}, bbox {bbox}, state text {texts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
