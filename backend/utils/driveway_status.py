"""
Which cars are in the driveway right now, and for how long, from the logger's latest data.

No new scan is done. A car counts as present if the logger has read its plate since the last
periodic scan started (every 5 minutes by default), so the answer can be up to one scan interval old:
a car that just left is still listed until the next periodic scan misses it.

Usage:
    from utils.driveway_status import cars_in_driveway
    for car in cars_in_driveway():
        print(car["plate"], car["duration"])
"""

import sqlite3
from datetime import datetime
from pathlib import Path

from .known_vehicles import lookup_vehicles

DEFAULT_DB = Path(__file__).resolve().parent.parent / "logs" / "plates.db"


def format_duration(seconds: int) -> str:
    """e.g. 45 -> '45s', 125 -> '2m 05s', 7500 -> '2h 05m', 180000 -> '2d 2h'."""
    days, rest = divmod(seconds, 86400)
    hours, rest = divmod(rest, 3600)
    minutes, secs = divmod(rest, 60)
    if days:
        return f"{days}d {hours}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def cars_in_driveway(db_path: str | Path = DEFAULT_DB, now: datetime | None = None) -> list[dict]:
    """Return one dict per car currently in the driveway, longest-parked first.

    Each dict holds the car's sightings row (id, time = when it arrived, plate, confidence, reads,
    snapshot, source, last_seen) plus:
      duration_seconds  how long it has been in the driveway, in whole seconds
      duration          the same, readable, e.g. "2h 05m"
      vehicle           its known_vehicles row as a dict, or None if it isn't a known vehicle
    Returns [] if the logger hasn't done a periodic scan yet.
    """
    now = now or datetime.now()
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        last_scan = db.execute("SELECT value FROM status WHERE key = 'last_periodic_scan'").fetchone()
        if last_scan is None:
            return []
        # Latest visit of each plate, if it was read during or after the last periodic scan.
        rows = db.execute(
            "SELECT * FROM sightings WHERE id IN (SELECT max(id) FROM sightings GROUP BY plate)"
            " AND last_seen >= ? ORDER BY time",
            (last_scan["value"],),
        ).fetchall()
    finally:
        db.close()

    cars = [dict(r) for r in rows]
    known = lookup_vehicles([c["plate"] for c in cars], db_path)
    for car in cars:
        seconds = max(0, int((now - datetime.fromisoformat(car["time"])).total_seconds()))
        car["duration_seconds"] = seconds
        car["duration"] = format_duration(seconds)
        car["vehicle"] = known.get(car["plate"])
    return cars
