"""
Look up license plates in the known_vehicles table of the driveway logger's database.

Usage:
    from utils.known_vehicles import lookup_vehicles
    lookup_vehicles(["WPG6392", "STH-8661", "ABC1234"])
    # -> {"WPG6392": {"id": 1, "plate": "WPG6392", "year": 2013, "make": "Honda", ...}}
"""

import re
import sqlite3
from pathlib import Path

DEFAULT_DB = Path(__file__).resolve().parent.parent / "logs" / "plates.db"


def normalize_plate(plate: str) -> str:
    """Uppercase and drop dashes, spaces and other punctuation, matching how the logger stores plates."""
    return re.sub(r"[^A-Z0-9]", "", plate.upper())


def lookup_vehicles(plates: list[str], db_path: str | Path = DEFAULT_DB) -> dict[str, dict]:
    """Return {plate: known_vehicles row as a dict} for each plate that is a known vehicle.

    Keys are the plates exactly as passed in; matching ignores case, dashes and spaces.
    Plates that aren't in known_vehicles are left out of the result.
    """
    wanted = {p: normalize_plate(p) for p in plates}
    if not wanted:
        return {}

    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        normalized = sorted(set(wanted.values()))
        # Also strip dashes/spaces on the table side, in case a plate was entered as "STH-8661".
        rows = db.execute(
            "SELECT * FROM known_vehicles WHERE upper(replace(replace(plate, '-', ''), ' ', ''))"
            f" IN ({','.join('?' * len(normalized))})",
            normalized,
        ).fetchall()
    finally:
        db.close()

    by_plate = {normalize_plate(r["plate"]): dict(r) for r in rows}
    return {p: by_plate[n] for p, n in wanted.items() if n in by_plate}
