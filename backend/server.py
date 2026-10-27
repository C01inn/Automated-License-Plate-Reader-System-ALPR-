"""
Minimal JSON API over the driveway logger's database. No authentication: keep it on your home network.

Endpoints:
    GET    /api/status                  logger health: last periodic scan, counts, latest sighting
    GET    /api/driveway                cars in the driveway now, with how long they've been there
    GET    /api/sightings               visit history, newest first
                                        ?plate=WPG6392 &since=2026-10-26 &until=2026-10-27 &limit=50 &offset=0
    GET    /api/vehicles                known vehicles
    POST   /api/vehicles                add one: JSON body with plate (required) and any of
                                        state, year, make, model, color, owner, relationship, notes
    GET    /api/vehicles/<id>           one known vehicle
    PATCH  /api/vehicles/<id>           change some of its fields
    DELETE /api/vehicles/<id>           remove it
    GET    /snapshots/<file>            snapshot image (sighting rows include snapshot_url and plate_snapshot_url)
    GET    /                            the web frontend, from ../frontend

Run (from backend/):
    venv/bin/python server.py           # serves on http://<pi address>:5000
"""

import sqlite3
import sys
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_from_directory

from utils.driveway_status import cars_in_driveway
from utils.known_vehicles import lookup_vehicles, normalize_plate

LOGS = Path(__file__).resolve().parent / "logs"
FRONTEND = Path(__file__).resolve().parent.parent / "frontend"
VEHICLE_FIELDS = ["plate", "state", "year", "make", "model", "color", "owner", "relationship", "notes"]

app = Flask(__name__)
app.config["DB"] = LOGS / "plates.db"


def db() -> sqlite3.Connection:
    conn = sqlite3.connect(app.config["DB"])
    conn.row_factory = sqlite3.Row
    return conn


def with_urls(row: dict) -> dict:
    """Add URLs for a sighting's snapshot and its plate close-up."""
    if row.get("snapshot"):
        name = Path(row["snapshot"]).name
        row["snapshot_url"] = f"/snapshots/{name}"
        row["plate_snapshot_url"] = f"/snapshots/{name.removesuffix('.jpg')}-plate.jpg"
    return row


def vehicle_or_404(conn: sqlite3.Connection, vehicle_id: int) -> dict:
    row = conn.execute("SELECT * FROM known_vehicles WHERE id = ?", (vehicle_id,)).fetchone()
    if row is None:
        abort(404, "no known vehicle with that id")
    return dict(row)


def vehicle_fields(body: dict) -> dict:
    """The known_vehicles columns present in a request body, with the plate normalized."""
    fields = {k: body[k] for k in VEHICLE_FIELDS if k in body}
    if "plate" in fields:
        fields["plate"] = normalize_plate(str(fields["plate"] or ""))
        if not fields["plate"]:
            abort(400, "plate can't be empty")
    return fields


@app.errorhandler(400)
@app.errorhandler(404)
@app.errorhandler(409)
def json_error(e):
    return jsonify(error=e.description), e.code


@app.get("/api/status")
def status():
    with db() as conn:
        last_scan = conn.execute("SELECT value FROM status WHERE key = 'last_periodic_scan'").fetchone()
        latest = conn.execute("SELECT * FROM sightings ORDER BY id DESC LIMIT 1").fetchone()
        return jsonify(
            last_periodic_scan=last_scan[0] if last_scan else None,
            sightings=conn.execute("SELECT count(*) FROM sightings").fetchone()[0],
            known_vehicles=conn.execute("SELECT count(*) FROM known_vehicles").fetchone()[0],
            latest_sighting=with_urls(dict(latest)) if latest else None,
        )


@app.get("/api/driveway")
def driveway():
    return jsonify([with_urls(car) for car in cars_in_driveway(app.config["DB"])])


@app.get("/api/sightings")
def sightings():
    where, params = [], []
    if plate := request.args.get("plate"):
        where.append("plate = ?")
        params.append(normalize_plate(plate))
    if since := request.args.get("since"):
        where.append("time >= ?")
        params.append(since)
    if until := request.args.get("until"):
        where.append("time < ?")
        params.append(until)
    limit = min(request.args.get("limit", 50, type=int), 500)
    offset = request.args.get("offset", 0, type=int)

    sql = "SELECT * FROM sightings"
    if where:
        sql += " WHERE " + " AND ".join(where)
    with db() as conn:
        rows = [dict(r) for r in conn.execute(f"{sql} ORDER BY id DESC LIMIT ? OFFSET ?", (*params, limit, offset))]
    known = lookup_vehicles([r["plate"] for r in rows], app.config["DB"])
    for r in rows:
        with_urls(r)
        r["vehicle"] = known.get(r["plate"])
    return jsonify(rows)


@app.get("/api/vehicles")
def list_vehicles():
    with db() as conn:
        return jsonify([dict(r) for r in conn.execute("SELECT * FROM known_vehicles ORDER BY plate")])


@app.post("/api/vehicles")
def add_vehicle():
    fields = vehicle_fields(request.get_json(silent=True) or {})
    if "plate" not in fields:
        abort(400, "plate is required")
    with db() as conn:
        try:
            cur = conn.execute(
                f"INSERT INTO known_vehicles ({', '.join(fields)}) VALUES ({', '.join('?' * len(fields))})",
                list(fields.values()),
            )
        except sqlite3.IntegrityError:
            abort(409, f"{fields['plate']} is already a known vehicle")
        return jsonify(vehicle_or_404(conn, cur.lastrowid)), 201


@app.get("/api/vehicles/<int:vehicle_id>")
def get_vehicle(vehicle_id):
    with db() as conn:
        return jsonify(vehicle_or_404(conn, vehicle_id))


@app.patch("/api/vehicles/<int:vehicle_id>")
def update_vehicle(vehicle_id):
    fields = vehicle_fields(request.get_json(silent=True) or {})
    if not fields:
        abort(400, f"send at least one of: {', '.join(VEHICLE_FIELDS)}")
    with db() as conn:
        vehicle_or_404(conn, vehicle_id)
        try:
            conn.execute(
                f"UPDATE known_vehicles SET {', '.join(f'{k} = ?' for k in fields)} WHERE id = ?",
                [*fields.values(), vehicle_id],
            )
        except sqlite3.IntegrityError:
            abort(409, f"{fields['plate']} is already a known vehicle")
        return jsonify(vehicle_or_404(conn, vehicle_id))


@app.delete("/api/vehicles/<int:vehicle_id>")
def delete_vehicle(vehicle_id):
    with db() as conn:
        vehicle_or_404(conn, vehicle_id)
        conn.execute("DELETE FROM known_vehicles WHERE id = ?", (vehicle_id,))
    return "", 204


@app.get("/snapshots/<path:name>")
def snapshot(name):
    return send_from_directory(LOGS / "snaps", name)


@app.get("/")
def frontend_index():
    return send_from_directory(FRONTEND, "index.html")


@app.get("/<path:name>")
def frontend_file(name):
    return send_from_directory(FRONTEND, name)


if __name__ == "__main__":
    if not app.config["DB"].exists():
        sys.exit(f"No database at {app.config['DB']}: run driveway_logger.py once to create it.")
    # WAL lets the API read while the logger is writing, without either waiting on the other.
    with db() as conn:
        conn.execute("PRAGMA journal_mode=WAL")
    app.run(host="0.0.0.0", port=5000)
