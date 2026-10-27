// Driveway frontend: plain JS talking to the Flask API in backend/server.py.

const REFRESH_MS = 30000;
const PAGE_SIZE = 50;
const VEHICLE_FIELDS = ["plate", "state", "year", "make", "model", "color", "owner", "relationship", "notes"];

const $ = (sel) => document.querySelector(sel);

async function api(path, options = {}) {
  const res = await fetch(path, {
    ...options,
    headers: options.body ? { "Content-Type": "application/json" } : {},
  });
  if (res.status === 204) return null;
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.error || `Request failed (${res.status})`);
  return data;
}

// Everything shown comes from OCR or user input, so escape it before putting it in HTML.
function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
}

function fmtTime(iso) {
  if (!iso) return "";
  return new Date(iso).toLocaleString([], { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" });
}

function ago(iso) {
  const mins = Math.round((Date.now() - new Date(iso)) / 60000);
  if (mins < 1) return "just now";
  if (mins < 60) return `${mins} min ago`;
  if (mins < 1440) return `${Math.round(mins / 60)} h ago`;
  return `${Math.round(mins / 1440)} d ago`;
}

function vehicleName(v) {
  return v ? [v.year, v.color, v.make, v.model].filter(Boolean).join(" ") || "Known vehicle" : "";
}

function knownTag(v) {
  return v
    ? `<span class="tag known">${esc(vehicleName(v))}${v.owner ? ` · ${esc(v.owner)}` : ""}</span>`
    : `<span class="tag unknown">Unknown</span>`;
}

function thumb(url, full) {
  return url ? `<img class="thumb" src="${esc(url)}" data-full="${esc(full || url)}" alt="" loading="lazy">` : "";
}

// ---- Status bar --------------------------------------------------------------------------------

async function loadStatus() {
  const el = $("#status");
  try {
    const s = await api("/api/status");
    const stale = !s.last_periodic_scan || Date.now() - new Date(s.last_periodic_scan) > 10 * 60000;
    const scan = s.last_periodic_scan ? `Last scan ${ago(s.last_periodic_scan)}` : "Logger hasn't scanned yet";
    el.textContent = `${scan} · ${s.sightings} visits · ${s.known_vehicles} known vehicles`;
    el.classList.toggle("stale", stale);
  } catch (e) {
    el.textContent = `Can't reach the server: ${e.message}`;
    el.classList.add("stale");
  }
}

// ---- Now ---------------------------------------------------------------------------------------

async function loadDriveway() {
  const el = $("#driveway");
  try {
    const cars = await api("/api/driveway");
    el.innerHTML = cars.length
      ? cars.map((c) => `
        <article class="card">
          <img src="${esc(c.snapshot_url)}" data-full="${esc(c.snapshot_url)}" alt="">
          <div class="body">
            <span class="plate">${esc(c.plate)}</span> ${knownTag(c.vehicle)}
            <div class="duration">${esc(c.duration)}</div>
            <div class="muted">Arrived ${esc(fmtTime(c.time))} · seen ${esc(ago(c.last_seen))}</div>
          </div>
        </article>`).join("")
      : `<p class="empty">No cars in the driveway.</p>`;
  } catch (e) {
    el.innerHTML = `<p class="error">${esc(e.message)}</p>`;
  }
}

// ---- History -----------------------------------------------------------------------------------

let historyOffset = 0;

function historyQuery() {
  const form = new FormData($("#history-filter"));
  const params = new URLSearchParams({ limit: PAGE_SIZE, offset: historyOffset });
  if (form.get("plate")) params.set("plate", form.get("plate"));
  if (form.get("since")) params.set("since", form.get("since"));
  if (form.get("until")) {
    // "To" is inclusive, so ask for everything before the start of the next day.
    const next = new Date(form.get("until") + "T00:00:00");
    next.setDate(next.getDate() + 1);
    params.set("until", next.toLocaleDateString("sv")); // sv gives YYYY-MM-DD
  }
  return params;
}

async function loadSightings(reset = true) {
  if (reset) historyOffset = 0;
  const body = $("#sightings");
  try {
    const rows = await api(`/api/sightings?${historyQuery()}`);
    const html = rows.map((r) => `
      <tr>
        <td><span class="plate">${esc(r.plate)}</span></td>
        <td>${knownTag(r.vehicle)}</td>
        <td>${esc(fmtTime(r.time))}</td>
        <td class="muted">${esc(fmtTime(r.last_seen))}</td>
        <td>${thumb(r.plate_snapshot_url, r.snapshot_url)}</td>
      </tr>`).join("");
    if (reset) body.innerHTML = html || `<tr><td colspan="5" class="empty">No visits found.</td></tr>`;
    else body.insertAdjacentHTML("beforeend", html);
    historyOffset += rows.length;
    $("#more").hidden = rows.length < PAGE_SIZE;
  } catch (e) {
    body.innerHTML = `<tr><td colspan="5" class="error">${esc(e.message)}</td></tr>`;
  }
}

$("#history-filter").addEventListener("submit", (e) => { e.preventDefault(); loadSightings(); });
$("#history-filter").addEventListener("reset", () => setTimeout(loadSightings));
$("#more").addEventListener("click", () => loadSightings(false));

// ---- Known vehicles ----------------------------------------------------------------------------

let vehicles = [];

async function loadVehicles() {
  const body = $("#vehicle-list");
  try {
    vehicles = await api("/api/vehicles");
    body.innerHTML = vehicles.length
      ? vehicles.map((v) => `
        <tr>
          <td><span class="plate">${esc(v.plate)}</span>${v.state ? ` <span class="muted">${esc(v.state)}</span>` : ""}</td>
          <td>${esc(vehicleName(v))}</td>
          <td>${esc(v.owner)}</td>
          <td>${esc(v.relationship)}</td>
          <td class="muted">${esc(v.notes)}</td>
          <td>
            <button class="link" data-edit="${v.id}">Edit</button>
            <button class="link danger" data-delete="${v.id}">Delete</button>
          </td>
        </tr>`).join("")
      : `<tr><td colspan="6" class="empty">No known vehicles yet. Add one above.</td></tr>`;
  } catch (e) {
    body.innerHTML = `<tr><td colspan="6" class="error">${esc(e.message)}</td></tr>`;
  }
}

function resetVehicleForm() {
  const form = $("#vehicle-form");
  form.reset();
  form.elements.id.value = "";
  $("#vehicle-save").textContent = "Add vehicle";
  $("#vehicle-cancel").hidden = true;
  $("#vehicle-error").textContent = "";
}

$("#vehicle-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const form = e.target;
  const body = {};
  for (const field of VEHICLE_FIELDS) {
    const value = form.elements[field].value.trim();
    body[field] = value === "" ? null : field === "year" ? Number(value) : value;
  }
  const id = form.elements.id.value;
  try {
    await api(id ? `/api/vehicles/${id}` : "/api/vehicles", {
      method: id ? "PATCH" : "POST",
      body: JSON.stringify(body),
    });
    resetVehicleForm();
    loadVehicles();
    loadStatus();
  } catch (err) {
    $("#vehicle-error").textContent = err.message;
  }
});

$("#vehicle-cancel").addEventListener("click", resetVehicleForm);

$("#vehicle-list").addEventListener("click", async (e) => {
  const { edit, delete: del } = e.target.dataset;
  if (edit) {
    const v = vehicles.find((x) => x.id === Number(edit));
    const form = $("#vehicle-form");
    form.elements.id.value = v.id;
    for (const field of VEHICLE_FIELDS) form.elements[field].value = v[field] ?? "";
    $("#vehicle-save").textContent = "Save changes";
    $("#vehicle-cancel").hidden = false;
    form.scrollIntoView({ behavior: "smooth" });
  } else if (del) {
    const v = vehicles.find((x) => x.id === Number(del));
    if (!confirm(`Delete ${v.plate} from known vehicles?`)) return;
    try {
      await api(`/api/vehicles/${del}`, { method: "DELETE" });
      loadVehicles();
      loadStatus();
    } catch (err) {
      alert(err.message);
    }
  }
});

// ---- Snapshot viewer ---------------------------------------------------------------------------

document.addEventListener("click", (e) => {
  const full = e.target.dataset?.full;
  if (!full) return;
  const viewer = $("#viewer");
  viewer.querySelector("img").src = full;
  viewer.showModal();
});

// ---- Views and refresh -------------------------------------------------------------------------

const loaders = { now: loadDriveway, history: loadSightings, vehicles: loadVehicles };

function currentView() {
  const name = location.hash.slice(1);
  return name in loaders ? name : "now";
}

function show() {
  const view = currentView();
  document.querySelectorAll(".view").forEach((s) => { s.hidden = s.id !== view; });
  document.querySelectorAll("nav a").forEach((a) => a.classList.toggle("active", a.hash === `#${view}`));
  loaders[view]();
}

window.addEventListener("hashchange", show);
show();
loadStatus();

// Keep the status bar and the "Now" view current; leave history and the vehicle form alone.
setInterval(() => {
  if (document.hidden) return;
  loadStatus();
  if (currentView() === "now") loadDriveway();
}, REFRESH_MS);
