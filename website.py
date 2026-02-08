#!/usr/bin/env python3
import os
import json
import pickle
import argparse
import csv
import re
import math
import shutil
from collections import Counter


# ---------------------------- helpers ----------------------------
def _to_json_safe(obj):
    try:
        import numpy as np
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
    except Exception:
        pass
    return str(obj)


def safe_filename(name: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")
    return s if s else "untitled"


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def copy_file_if_needed(src: str, dst: str):
    ensure_dir(os.path.dirname(dst))
    if os.path.exists(dst):
        return
    shutil.copy2(src, dst)


def _extract_ev_tag(s: str) -> str:
    if not s:
        return ""
    s = str(s)
    for ev in ("ev-00", "ev-25", "ev-50"):
        if ev in s:
            return ev
    return ""


def _normalize_id_and_ev(s: str):
    s = str(s).strip()
    s = os.path.basename(s)

    ev = _extract_ev_tag(s)

    low = s.lower()
    if low.endswith(".gz"):
        s = s[:-3]
        low = s.lower()
    if low.endswith(".json"):
        s = s[:-5]
        low = s.lower()

    if low.endswith("_points"):
        s = s[:-7]
        low = s.lower()

    for tag in ("_ev-00", "_ev-25", "_ev-50"):
        if low.endswith(tag):
            s = s[: -len(tag)]
            low = s.lower()
            break

    s = re.sub(r"^(?:rank\s*)?r[123]\s*[:_\-]+", "", s, flags=re.IGNORECASE)

    s = s.replace(" ", "_")
    while "__" in s:
        s = s.replace("__", "_")
    s = s.strip("_")

    return s, ev


def confidence_from_row(row, E_good=1.0, E_bad=8.0):
    try:
        frac = float(row.get("inlier_weight_frac", 0.0))
    except Exception:
        frac = 0.0
    try:
        mae = float(row.get("mae_in_deg_weighted", 1e9))
    except Exception:
        mae = 1e9

    if E_bad <= E_good:
        mae_conf = 1.0 if mae <= E_good else 0.0
    else:
        t = (mae - E_good) / (E_bad - E_good)
        if t < 0.0:
            mae_conf = 1.0
        elif t > 1.0:
            mae_conf = 0.0
        else:
            mae_conf = 1.0 - t

    conf = frac * mae_conf
    return max(0.0, min(1.0, conf))


def load_confidence_map_by_id_and_rank(csv_path: str, desired_ev: str, E_good=1.0, E_bad=8.0, debug=False):
    conf_map = {}
    if not csv_path:
        return conf_map
    if not os.path.exists(csv_path):
        print(f"WARNING: CSV not found: {csv_path}")
        return conf_map

    kept = 0
    skipped_ev = 0
    skipped_rank = 0

    with open(csv_path, newline="", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                rk = int(float(row.get("rank", "0")))
            except Exception:
                skipped_rank += 1
                continue

            base_id, ev = _normalize_id_and_ev(row.get("file", ""))

            if desired_ev and ev != desired_ev:
                skipped_ev += 1
                continue

            conf_map[(base_id, rk)] = confidence_from_row(row, E_good=E_good, E_bad=E_bad)
            kept += 1

    print(f"[CSV] Loaded confidence rows kept={kept} | skipped_ev={skipped_ev} | skipped_rank={skipped_rank}")
    if debug:
        sample_keys = list(conf_map.keys())[:8]
        print(f"[CSV] Sample conf_map keys (base_id, rank): {sample_keys}")
    return conf_map


# ---------------- physical direction extraction ----------------
DIR_KEY_CANDIDATES = [
    ("dir_x", "dir_y", "dir_z"),
    ("dir_vx", "dir_vy", "dir_vz"),
    ("dirvx", "dirvy", "dirvz"),
    ("light_dir_x", "light_dir_y", "light_dir_z"),
    ("direction_x", "direction_y", "direction_z"),
    ("vx", "vy", "vz"),
    ("Lx", "Ly", "Lz"),
    ("L_x", "L_y", "L_z"),
]

SINGLE_VEC_CANDIDATES = [
    "dir", "dir_v", "light_dir", "direction", "dir_vec", "dirv", "L", "light_direction"
]


def _as_float(x):
    try:
        return float(x)
    except Exception:
        return None


def extract_direction_components(d: dict):
    for kx, ky, kz in DIR_KEY_CANDIDATES:
        if kx in d and ky in d and kz in d:
            x = _as_float(d.get(kx))
            y = _as_float(d.get(ky))
            z = _as_float(d.get(kz))
            if x is not None and y is not None and z is not None:
                return x, y, z, f"{kx},{ky},{kz}"

    for vk in SINGLE_VEC_CANDIDATES:
        if vk in d:
            v = d.get(vk)
            if isinstance(v, (list, tuple)) and len(v) >= 3:
                x = _as_float(v[0])
                y = _as_float(v[1])
                z = _as_float(v[2])
                if x is not None and y is not None and z is not None:
                    return x, y, z, vk

    for root in ("dir", "direction", "light_dir", "L"):
        obj = d.get(root)
        if isinstance(obj, dict):
            for cand in [("x", "y", "z"), ("vx", "vy", "vz"), ("dir_x", "dir_y", "dir_z")]:
                if all(k in obj for k in cand):
                    x = _as_float(obj.get(cand[0]))
                    y = _as_float(obj.get(cand[1]))
                    z = _as_float(obj.get(cand[2]))
                    if x is not None and y is not None and z is not None:
                        return x, y, z, f"{root}.{cand[0]},{root}.{cand[1]},{root}.{cand[2]}"

    return None, None, None, "no_direction_keys_found"


def normalize_vec(x, y, z, eps=1e-12):
    n = math.sqrt(x * x + y * y + z * z)
    if not math.isfinite(n) or n < eps:
        return None, None, None, n
    return x / n, y / n, z / n, n


# ---------------------------- recursive indexing ----------------------------
def build_basename_index(root_dir: str):
    """
    Map filename basename -> full path (first occurrence wins).
    This is what makes nested artist folders work.
    """
    idx = {}
    for r, _dirs, files in os.walk(root_dir):
        for fn in files:
            if fn not in idx:
                idx[fn] = os.path.join(r, fn)
    return idx


# ---------------------------- asset staging (GitHub Pages) ----------------------------
def stage_painting_images(points, img_src_dir: str, out_dir: str, img_prefix: str, debug=False):
    if not img_src_dir:
        print("[ASSETS] img-src not provided; images will not be copied.")
        return
    img_src_dir = os.path.normpath(img_src_dir)
    if not os.path.exists(img_src_dir):
        print(f"[ASSETS] WARNING: img-src not found: {img_src_dir}")
        return

    dst_root = os.path.join(out_dir, img_prefix)
    ensure_dir(dst_root)

    idx = build_basename_index(img_src_dir)

    needed = set()
    for d in points:
        p = str(d.get("img_path", "") or "")
        if not p:
            continue
        base = os.path.basename(p.replace("\\", "/"))
        if base:
            needed.add(base)

    copied = 0
    missing = 0
    for base in sorted(needed):
        src = idx.get(base)
        dst = os.path.join(dst_root, base)
        if src and os.path.exists(src):
            copy_file_if_needed(src, dst)
            copied += 1
        else:
            missing += 1
            if debug:
                print(f"[ASSETS] Missing painting image basename='{base}' under img-src")

    print(f"[ASSETS] Painting images staged: copied={copied}, missing={missing}, dst='{dst_root}'")


def stage_plots(plots_src_dir: str, out_dir: str, plots_prefix: str, debug=False):
    if not plots_src_dir:
        print("[ASSETS] plots-src not provided; arrow JSON will not be copied.")
        return
    plots_src_dir = os.path.normpath(plots_src_dir)
    if not os.path.exists(plots_src_dir):
        print(f"[ASSETS] WARNING: plots-src not found: {plots_src_dir}")
        return

    dst_root = os.path.join(out_dir, plots_prefix)
    ensure_dir(dst_root)

    copied = 0
    for r, _dirs, files in os.walk(plots_src_dir):
        for fn in files:
            if fn.endswith(".plot.json"):
                src = os.path.join(r, fn)
                dst = os.path.join(dst_root, fn)
                copy_file_if_needed(src, dst)
                copied += 1

    print(f"[ASSETS] Plot JSON staged: copied={copied}, dst='{dst_root}'")
    if debug and copied == 0:
        print("[ASSETS] NOTE: No .plot.json files found in plots-src.")


def stage_balls(balls_src_dir: str, out_dir: str, balls_prefix: str, debug=False):
    if not balls_src_dir:
        print("[ASSETS] balls-src not provided; chrome balls will not be copied.")
        return
    balls_src_dir = os.path.normpath(balls_src_dir)
    if not os.path.exists(balls_src_dir):
        print(f"[ASSETS] WARNING: balls-src not found: {balls_src_dir}")
        return

    dst_root = os.path.join(out_dir, balls_prefix)
    ensure_dir(dst_root)

    copied = 0
    for r, _dirs, files in os.walk(balls_src_dir):
        for fn in files:
            low = fn.lower()
            if low.endswith(".png") and ("_ev-00" in low or "_ev-25" in low or "_ev-50" in low):
                src = os.path.join(r, fn)
                dst = os.path.join(dst_root, fn)
                copy_file_if_needed(src, dst)
                copied += 1

    print(f"[ASSETS] Chrome balls staged: copied={copied}, dst='{dst_root}'")
    if debug and copied == 0:
        print("[ASSETS] NOTE: No *_ev-XX.png files found in balls-src.")


# ---------------------------- HTML ----------------------------
def generate_html(all_data_points, img_prefix="data", plots_prefix="plots_embedded", balls_prefix="balls"):
    html = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>Interactive Light Direction</title>
<script src="https://cdn.plot.ly/plotly-2.32.0.min.js"></script>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif;
         margin:0; padding:0; background:#f2f2f7; color:#1d1d1f; }
  .container { max-width:1480px; margin:32px auto; padding:28px;
               background:rgba(255,255,255,0.86);
               border:0.5px solid rgba(0,0,0,0.06);
               border-radius:14px;
               box-shadow:0 8px 32px rgba(0,0,0,0.06);
               backdrop-filter: blur(18px); }
  .title { text-align:center; font-size:30px; font-weight:650; margin:0 0 10px; letter-spacing:-0.02em; }
  .row { display:flex; gap:18px; align-items:flex-start; }
  .left { flex:1; min-width:720px; background:rgba(255,255,255,0.92);
          border:0.5px solid rgba(0,0,0,0.06); border-radius:14px; padding:18px;
          box-shadow:0 4px 16px rgba(0,0,0,0.06); }
  .right { width:440px; background:rgba(255,255,255,0.92);
           border:0.5px solid rgba(0,0,0,0.06); border-radius:14px; padding:18px;
           box-shadow:0 4px 16px rgba(0,0,0,0.06); position:sticky; top:18px; }
  #plot { width:100%; height:680px; border-radius:12px; }

  .controls { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:12px; justify-content:space-between; }
  .control-box { flex:1; min-width:240px; background:rgba(0,0,0,0.03);
                 border:1px solid rgba(0,0,0,0.05); border-radius:12px; padding:12px; }
  .control-title { font-size:13px; font-weight:650; color:#374151; margin-bottom:8px; }
  select, input[type="text"] { width:100%; padding:10px 12px; border-radius:10px;
                               border:1px solid rgba(0,0,0,0.10); background:white; outline:none; font-size:14px; }
  .hint { font-size:12px; color:#6b7280; margin-top:6px; }

  .pillbar { display:flex; gap:8px; flex-wrap:wrap; margin-top:10px; }
  .pill { padding:6px 10px; border-radius:999px; background:rgba(0,0,0,0.04);
          border:1px solid rgba(0,0,0,0.06); font-size:12px; color:#374151; }

  .btnrow { display:flex; gap:10px; margin-top:10px; flex-wrap:wrap; }
  .btn { flex:1; min-width:160px; padding:10px 12px; border-radius:999px; border:0; cursor:pointer;
         background:rgba(0,0,0,0.04); color:#1d1d1f; font-weight:600;
         transition:transform 0.15s ease, background 0.15s ease; }
  .btn:hover { background:rgba(0,122,255,0.12); color:#007aff; transform:translateY(-1px); }
  .btn.active { background:#007aff; color:#ffffff; box-shadow:0 2px 8px rgba(0,122,255,0.25); }

  .rankbar { display:flex; gap:10px; margin-top:10px; flex-wrap:wrap; }
  .rankbtn {
    padding:9px 12px;
    border-radius:999px;
    border:1px solid rgba(0,0,0,0.10);
    background:white;
    cursor:pointer;
    font-weight:750;
    font-size:12px;
    color:#111827;
    box-shadow: 0 2px 0 rgba(0,0,0,0.18), 0 8px 18px rgba(0,0,0,0.08);
    transition: transform 0.12s ease, box-shadow 0.12s ease, opacity 0.12s ease;
    user-select:none;
  }
  .rankbtn:hover { transform: translateY(-1px); }
  .rankbtn.off {
    opacity:0.35;
    box-shadow: 0 1px 0 rgba(0,0,0,0.12), 0 4px 10px rgba(0,0,0,0.06);
  }

  .panel-title { font-size:18px; font-weight:700; margin:0 0 10px; }
  .empty { padding:34px 14px; text-align:center; color:#9CA3AF;
           border:2px dashed rgba(0,0,0,0.10); border-radius:14px; background:rgba(248,249,250,0.6); }

  .kv { margin-top:12px; border:1px solid rgba(0,0,0,0.06); border-radius:12px; overflow:hidden; }
  .kvrow { display:flex; justify-content:space-between; padding:10px 12px;
           border-bottom:1px solid rgba(0,0,0,0.06); background:rgba(255,255,255,0.9); }
  .kvrow:last-child { border-bottom:0; }
  .k { color:#6b7280; font-size:13px; font-weight:600; }
  .v { color:#111827; font-size:13px; font-weight:650; text-align:right;
       max-width:260px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }

  table.ranktbl { width:100%; border-collapse:collapse; font-size:12px; margin-top:8px; }
  table.ranktbl th, table.ranktbl td { padding:6px 6px; border-bottom:1px solid rgba(0,0,0,0.07); vertical-align:top; }
  table.ranktbl th { text-align:left; color:#374151; font-weight:700; background:rgba(0,0,0,0.02); }

  .badge { display:inline-block; padding:3px 8px; border-radius:999px;
           background:rgba(0,0,0,0.05); border:1px solid rgba(0,0,0,0.07);
           font-size:12px; font-weight:650; color:#374151; }

  .neighbors { margin-top:14px; }
  .neighbors-title { font-weight:800; color:#111827; margin-top:12px; }
  .neighbor-grid { display:grid; grid-template-columns: 1fr 1fr 1fr; gap:10px; margin-top:10px; }
  .neighbor-card { background:white; border:1px solid rgba(0,0,0,0.08); border-radius:12px;
                   padding:8px; cursor:pointer; transition:transform 0.15s ease, box-shadow 0.15s ease; }
  .neighbor-card:hover { transform:translateY(-2px); box-shadow:0 10px 22px rgba(0,0,0,0.10); border-color:rgba(0,122,255,0.35); }
  .neighbor-card img { width:100%; height:100px; object-fit:cover; border-radius:10px; }
  .neighbor-meta { margin-top:6px; font-size:11px; color:#374151; line-height:1.25; }
  .neighbor-meta .id { font-weight:750; color:#111827; }
  .neighbor-meta .ang { color:#6b7280; }

  .imgwrap { margin-top:12px; }
  .imgwrap img { width:100%; border-radius:12px; box-shadow:0 10px 26px rgba(0,0,0,0.18); }

  .plotwrap { margin-top:12px; border:1px solid rgba(0,0,0,0.06); border-radius:12px; overflow:hidden; background:white; }
  #arrowPlotWrapper {
    height: 260px;
    max-height: 260px;
    width: 100%;
    overflow: hidden;
    border-radius: 12px;
    background: #f8fafc;
    box-shadow: inset 0 0 0 1px rgba(0,0,0,0.06);
  }
  #arrowPlot { width:100%; height:100%; }
  .smallhint { font-size:12px; color:#6b7280; margin-top:8px; }

  .balls-title { font-weight:800; color:#111827; margin-top:14px; font-size:20px; }
  .balls-sub { font-size:12px; color:#6b7280; margin-top:6px; }

  .ball-stack { display:flex; flex-direction:column; gap:12px; margin-top:12px; }
  .ball-card { background:white; border:1px solid rgba(0,0,0,0.08); border-radius:14px;
               padding:10px; overflow:hidden; }
  .ball-card img {
    width: 100%;
    height: auto;
    max-height: 520px;
    object-fit: contain;
    background: #fff;
    border-radius: 12px;
    display: block;
  }
  .ball-label { margin-top:8px; font-weight:800; color:#111827; }
  .ball-fn { margin-top:4px; font-size:12px; color:#6b7280; word-break:break-all; }
</style>
</head>
<body>
  <div class="container">
    <h1 class="title">Interactive 3D Light Direction</h1>

    <div class="controls">
      <div class="control-box">
        <div class="control-title">Artist filter</div>
        <select id="artistSelect" onchange="applyFilters()">
          <option value="__ALL__">All artists</option>
        </select>
        <div class="hint">Colors are deterministic per artist (muted palette).</div>
      </div>

      <div class="control-box">
        <div class="control-title">Genre filter</div>
        <select id="genreSelect" onchange="applyFilters()">
          <option value="__ALL__">All genres</option>
        </select>
        <div class="hint">Genre comes from your folder structure.</div>
      </div>

      <div class="control-box">
        <div class="control-title">Search (painting id or rank id)</div>
        <input id="searchBox" type="text" placeholder="Type to find... e.g. 'R2' or name" oninput="applyFilters()" />
        <div class="hint">Matches painting_info and painting_rank_id.</div>
      </div>
    </div>

    <div class="pillbar" id="statsPills"></div>

    <div class="row">
      <div class="left">
        <div class="btnrow">
          <button class="btn" onclick="resetView()">Reset View</button>
          <button class="btn" onclick="clearSelection()">Clear Selection (ESC)</button>
        </div>

        <div class="rankbar">
          <button id="rk1" class="rankbtn" onclick="toggleRank(1)">Rank 1</button>
          <button id="rk2" class="rankbtn" onclick="toggleRank(2)">Rank 2</button>
          <button id="rk3" class="rankbtn" onclick="toggleRank(3)">Rank 3</button>
        </div>

        <div style="margin-top:12px;">
          <div id="plot"></div>
        </div>
      </div>

      <div class="right">
        <div class="panel-title">Sample Details</div>

        <div class="btnrow" style="margin-top: 0;">
          <button id="view2dBtn" class="btn active" onclick="showRightPanel('2d')">2D View</button>
          <button id="view3dBtn" class="btn" onclick="showRightPanel('3d')">3D View</button>
        </div>

        <div id="details2d">
          <div id="details">
            <div class="empty">Click a point to view details.</div>
          </div>
        </div>

        <div id="details3d" style="display:none;">
          <div id="details3dInner">
            <div class="empty">Click a point to view details.</div>
          </div>
        </div>
      </div>
    </div>
  </div>

<script>
  const IMG_PREFIX = __IMG_PREFIX__;
  const PLOTS_PREFIX = __PLOTS_PREFIX__;
  const BALLS_PREFIX = __BALLS_PREFIX__;
  const dataAll = __DATA_JSON__;

  function safeFilename(name) {
    const s = String(name ?? "").replace(/[^A-Za-z0-9._-]+/g, "_").replace(/^_+|_+$/g, "");
    return s ? s : "untitled";
  }

  function escapeHtml(s){
    return String(s ?? "")
      .replaceAll("&","&amp;")
      .replaceAll("<","&lt;")
      .replaceAll(">","&gt;")
      .replaceAll('"',"&quot;")
      .replaceAll("'","&#039;");
  }

  dataAll.forEach(d => {
    d.painting_info = String(d.painting_info ?? d.pose_info ?? "");
    d.painting_rank_id = String(d.painting_rank_id ?? d.painting_info ?? "");
    d.rank = (d.rank == null) ? null : Number(d.rank);

    d.artist = String(d.dataset ?? "Unknown");
    d.genre = String(d.genre_info ?? "Unknown");
    d.img_path = String(d.img_path ?? "");

    let clean = d.img_path.replaceAll('\\\\', '/');
    const parts = clean.split("/");
    clean = parts[parts.length - 1];
    clean = clean.replace(/^\/+/, '');
    d.img_url = encodeURI(IMG_PREFIX.replaceAll('\\\\','/') + '/' + clean);

    d.x = Number(d.x);
    d.y = Number(d.y);
    d.z = Number(d.z);
    d.norm = Math.sqrt(d.x*d.x + d.y*d.y + d.z*d.z);

    d.confidence = (d.confidence == null) ? null : Number(d.confidence);
    if (Number.isNaN(d.confidence)) d.confidence = null;

    d.dominance = (d.dominance == null) ? null : Number(d.dominance);
    if (Number.isNaN(d.dominance)) d.dominance = null;

    d.dir_norm_raw = (d.dir_norm_raw == null) ? null : Number(d.dir_norm_raw);
    if (Number.isNaN(d.dir_norm_raw)) d.dir_norm_raw = null;

    d.dir_src_keys = String(d.dir_src_keys ?? "");
  });

  const byPainting = new Map();
  dataAll.forEach(d => {
    const key = d.painting_info;
    if (!byPainting.has(key)) byPainting.set(key, []);
    byPainting.get(key).push(d);
  });
  byPainting.forEach(arr => arr.sort((a,b) => ((a.rank ?? 99) - (b.rank ?? 99))));

  const repByPainting = new Map();
  byPainting.forEach((arr, key) => {
    let rep = arr.find(x => x.rank === 1) ?? arr[0];
    repByPainting.set(key, rep);
  });

  const artists = Array.from(new Set(dataAll.map(d => d.artist))).sort((a,b)=>a.localeCompare(b));
  const genres  = Array.from(new Set(dataAll.map(d => d.genre))).sort((a,b)=>a.localeCompare(b));

  const artistSelect = document.getElementById('artistSelect');
  artists.forEach(a => { const opt=document.createElement('option'); opt.value=a; opt.textContent=a; artistSelect.appendChild(opt); });

  const genreSelect = document.getElementById('genreSelect');
  genres.forEach(g => { const opt=document.createElement('option'); opt.value=g; opt.textContent=g; genreSelect.appendChild(opt); });

  function hashString(str) {
    let h = 2166136261;
    for (let i=0;i<str.length;i++){ h ^= str.charCodeAt(i); h = Math.imul(h, 16777619); }
    return h >>> 0;
  }
  const MUTED = ["#5B8E7D","#F2D0A4","#B5838D","#6D597A","#A8C686","#E6B8A2","#7EA8BE","#CDB4DB","#9A8C98","#DDBEA9"];
  const artistColorMap = new Map();
  function getArtistColor(artist) {
    if (artistColorMap.has(artist)) return artistColorMap.get(artist);
    const idx = hashString(String(artist)) % MUTED.length;
    const color = MUTED[idx];
    artistColorMap.set(artist, color);
    return color;
  }

  function makeLegendTraces() {
    return artists.map(a => ({
      type: 'scatter3d',
      mode: 'markers',
      name: a,
      x: [NaN], y: [NaN], z: [NaN],
      marker: { size: 8, color: getArtistColor(a), opacity: 1.0 },
      hoverinfo: 'skip',
      showlegend: true
    }));
  }

  function fmtPct(x) {
    if (x == null || Number.isNaN(x)) return '—';
    return (x*100).toFixed(0) + '%';
  }

  let enabledRanks = new Set([1,2,3]);
  function syncRankButtons(){
    [1,2,3].forEach(rk => {
      const el = document.getElementById(`rk${rk}`);
      if (!el) return;
      if (enabledRanks.has(rk)) el.classList.remove("off");
      else el.classList.add("off");
    });
  }
  window.toggleRank = function(rk){
    if (enabledRanks.has(rk)) enabledRanks.delete(rk);
    else enabledRanks.add(rk);
    if (enabledRanks.size === 0) enabledRanks.add(rk);
    syncRankButtons();
    applyFilters();
  }

  let dataFiltered = dataAll.slice();
  const plotDiv = document.getElementById('plot');

  const defaultRanges = { x: [-1.05, 1.05], y: [-1.05, 1.05], z: [-1.05, 1.05] };
  const cameraYZ = { eye: {x: 2.6, y: 0.0, z: 0.0}, center: {x:0,y:0,z:0}, up: {x:0,y:0,z:1} };
  let currentSphereCamera = cameraYZ;

  const baseLayout = {
    scene: {
      xaxis: { title: 'dir_x', range: defaultRanges.x, autorange:false, fixedrange:true },
      yaxis: { title: 'dir_y', range: defaultRanges.y, autorange:false, fixedrange:true },
      zaxis: { title: 'dir_z', range: defaultRanges.z, autorange:false, fixedrange:true },
      camera: cameraYZ,
      aspectmode: 'cube',
      bgcolor: '#ffffff'
    },
    margin: {t:10, b:10, l:10, r:10},
    paper_bgcolor: '#ffffff',
    plot_bgcolor: '#ffffff',
    showlegend: true,
    legend: {
      title: { text: "Artist" },
      bgcolor: "rgba(255,255,255,0.85)",
      bordercolor: "rgba(0,0,0,0.08)",
      borderwidth: 1
    }
  };

  const config = { responsive: true, displayModeBar: true, displaylogo: false };
  function pointSize() { return 3.6; }

  function makeTrace(points, rank, name, opacity) {
    if (!enabledRanks.has(rank)) {
      return { x:[], y:[], z:[], mode:'markers', type:'scatter3d', name,
               marker:{size:pointSize(), opacity:0.0}, hoverinfo:'skip', showlegend:false };
    }
    const pts = points.filter(d => d.rank === rank);
    return {
      showlegend: false,
      x: pts.map(d => d.x),
      y: pts.map(d => d.y),
      z: pts.map(d => d.z),
      mode: 'markers',
      type: 'scatter3d',
      name,
      marker: {
        size: pointSize(),
        symbol: 'circle',
        color: pts.map(d => getArtistColor(d.artist)),
        opacity: opacity,
        line: { color: 'rgba(255,255,255,0.6)', width: 0.5 }
      },
      text: pts.map(d => {
        const confText = fmtPct(d.confidence);
        const normText = Number(d.norm ?? 0).toFixed(3);
        const rk = (d.rank == null) ? '—' : String(d.rank);
        const src = d.dir_src_keys ? `Source: ${d.dir_src_keys}<br>` : '';
        return `${d.painting_info} (R${rk})<br>` +
               `Artist: ${d.artist}<br>` +
               `Genre: ${d.genre}<br>` +
               `Confidence: ${confText}<br>` +
               `||dir||: ${normText}<br>` +
               src +
               `dir: (${Number(d.x).toFixed(3)}, ${Number(d.y).toFixed(3)}, ${Number(d.z).toFixed(3)})`;
      }),
      hovertemplate: '%{text}<extra></extra>',
      customdata: pts
    };
  }

  function makeTraces(points) {
    const rankTraces = [
      makeTrace(points, 3, 'Rank 3 (25%)', 0.25),
      makeTrace(points, 2, 'Rank 2 (50%)', 0.50),
      makeTrace(points, 1, 'Rank 1 (100%)', 1.00),
    ];
    return rankTraces.concat(makeLegendTraces());
  }

  function updateStatsPills() {
    const pillbar = document.getElementById('statsPills');
    pillbar.innerHTML = '';
    const totalPts = dataFiltered.length;
    const totalAll = dataAll.length;
    const nPaintings = new Set(dataFiltered.map(d => d.painting_info)).size;
    const nArtists = new Set(dataFiltered.map(d => d.artist)).size;
    const nGenres  = new Set(dataFiltered.map(d => d.genre)).size;
    const confCount = dataFiltered.filter(d => d.confidence != null && !Number.isNaN(d.confidence)).length;

    const mk = (txt) => { const p=document.createElement('div'); p.className='pill'; p.textContent=txt; pillbar.appendChild(p); };
    mk(`Showing: ${totalPts} / ${totalAll} points`);
    mk(`Paintings: ${nPaintings}`);
    mk(`Artists: ${nArtists} | Genres: ${nGenres}`);
    mk(`Confidence present: ${confCount}/${totalPts}`);
  }

  function resetView() {
    Plotly.relayout(plotDiv, {
      'scene.xaxis.range': defaultRanges.x,
      'scene.yaxis.range': defaultRanges.y,
      'scene.zaxis.range': defaultRanges.z,
      'scene.camera': currentSphereCamera
    });
  }

  function clearSelection() {
    document.getElementById('details').innerHTML =
      '<div class="empty">Click a point to view details.</div>';
    document.getElementById('details3dInner').innerHTML =
      '<div class="empty">Click a point to view details.</div>';
    lastSelectedPainting = null;
    resetView();
  }
  document.addEventListener('keydown', (e) => { if (e.key === 'Escape') clearSelection(); });

  function clamp(x, lo, hi){ return Math.max(lo, Math.min(hi, x)); }
  function angleDegBetween(a, b){
    const dot = clamp(a[0]*b[0] + a[1]*b[1] + a[2]*b[2], -1.0, 1.0);
    return Math.acos(dot) * 180.0 / Math.PI;
  }

  function getTop3Neighbors(selectedPaintingInfo){
    const selRep = repByPainting.get(selectedPaintingInfo);
    if (!selRep) return [];
    const a = [selRep.x, selRep.y, selRep.z];

    const visiblePaintings = new Set(dataFiltered.map(d => d.painting_info));
    const cand = [];
    for (const pid of visiblePaintings){
      if (pid === selectedPaintingInfo) continue;
      const rep = repByPainting.get(pid);
      if (!rep) continue;
      const b = [rep.x, rep.y, rep.z];
      const ang = angleDegBetween(a, b);
      cand.push({ painting_info: pid, rep: rep, ang_deg: ang });
    }
    cand.sort((u,v) => u.ang_deg - v.ang_deg);
    return cand.slice(0, 3);
  }

  let lastSelectedPainting = null;

  async function loadArrowPlot3D(baseName) {
    const plotDiv = document.getElementById('arrowPlot');
    if (!plotDiv) return;

    const fn = safeFilename(baseName) + ".plot.json";
    const url = encodeURI(PLOTS_PREFIX.replaceAll('\\\\','/') + "/" + fn);

    plotDiv.innerHTML = "";
    try {
      const resp = await fetch(url);
      if (!resp.ok) {
        plotDiv.innerHTML = `<div style="padding:12px; color:#b91c1c; font-weight:700;">
          Arrow plot not found<br>
          <div style="margin-top:6px; font-weight:600; color:#6b7280;">
            Expected: ${url}
          </div>
        </div>`;
        return;
      }
      const payload = await resp.json();

      const layout = Object.assign({}, payload.layout || {});
      layout.height = 260;
      layout.margin = layout.margin || {l:10, r:10, t:40, b:10};

      try { Plotly.purge(plotDiv); } catch (e) {}
      Plotly.react(plotDiv, payload.data, layout, {displaylogo:false, displayModeBar:true, responsive:true});
    } catch (e) {
      plotDiv.innerHTML = `<div style="padding:12px; color:#b91c1c; font-weight:700;">
        Failed to load arrow plot<br>
        <div style="margin-top:6px; font-weight:600; color:#6b7280;">${String(e)}</div>
      </div>`;
    }
  }

  function showRightPanel(mode){
    const rightPanelMode = (mode === '3d') ? '3d' : '2d';
    const d2 = document.getElementById('details2d');
    const d3 = document.getElementById('details3d');
    const b2 = document.getElementById('view2dBtn');
    const b3 = document.getElementById('view3dBtn');

    if (d2 && d3) {
      d2.style.display = (rightPanelMode === '2d') ? 'block' : 'none';
      d3.style.display = (rightPanelMode === '3d') ? 'block' : 'none';
    }
    if (b2 && b3) {
      b2.classList.toggle('active', rightPanelMode === '2d');
      b3.classList.toggle('active', rightPanelMode === '3d');
    }
  }
  window.showRightPanel = showRightPanel;

  function render3DPanel(d){
    const wrap = document.getElementById('details3dInner');
    if (!wrap) return;

    const pid = String(d.painting_info ?? "");
    const artist = String(d.artist ?? "Unknown");
    const genre = String(d.genre ?? "Unknown");
    const base = safeFilename(pid);

    const urls = [
      { tag: "ev-00", fn: `${base}_ev-00.png` },
      { tag: "ev-25", fn: `${base}_ev-25.png` },
      { tag: "ev-50", fn: `${base}_ev-50.png` },
    ].map(o => ({
      tag: o.tag,
      fn: o.fn,
      url: encodeURI(BALLS_PREFIX.replaceAll('\\\\','/') + "/" + o.fn)
    }));

    wrap.innerHTML = `
      <div class="kv">
        <div class="kvrow"><div class="k">Painting ID</div><div class="v" title="${escapeHtml(pid)}">${escapeHtml(pid)}</div></div>
        <div class="kvrow"><div class="k">Artist</div><div class="v" title="${escapeHtml(artist)}">${escapeHtml(artist)}</div></div>
        <div class="kvrow"><div class="k">Genre</div><div class="v" title="${escapeHtml(genre)}">${escapeHtml(genre)}</div></div>
      </div>

      <div class="balls-title">Chrome balls (ev-00 / ev-25 / ev-50)</div>
      <div class="balls-sub">Loaded from <b>${escapeHtml(BALLS_PREFIX)}/</b> using <code>${escapeHtml(base)}_ev-XX.png</code></div>

      <div class="ball-stack">
        ${urls.map(u => `
          <div class="ball-card">
            <img src="${u.url}" alt="${escapeHtml(u.tag)}" />
            <div class="ball-label">${escapeHtml(u.tag)}</div>
            <div class="ball-fn">${escapeHtml(u.fn)}</div>
          </div>
        `).join('')}
      </div>
    `;
  }

  function showDetailsBundle(d) {
    const details = document.getElementById('details');
    const siblings = byPainting.get(d.painting_info) ?? [d];
    lastSelectedPainting = d.painting_info;

    const dom = (d.dominance == null || Number.isNaN(d.dominance))
      ? '—'
      : (Number(d.dominance)*100).toFixed(1) + '%';

    const rows = siblings.map(s => {
      const rk = (s.rank == null) ? '—' : 'R' + String(s.rank);
      const confText = fmtPct(s.confidence);
      const dirText  = `(${Number(s.x).toFixed(3)}, ${Number(s.y).toFixed(3)}, ${Number(s.z).toFixed(3)})`;
      return `
        <tr>
          <td><span class="badge">${rk}</span></td>
          <td>${confText}</td>
          <td style="font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace;">
            ${dirText}
          </td>
        </tr>
      `;
    }).join('');

    const neigh = getTop3Neighbors(d.painting_info);
    const neighHTML = (neigh.length === 0) ? `<div class="k" style="margin-top:8px;">—</div>` : `
      <div class="neighbor-grid">
        ${neigh.map(n => {
          const img = n.rep.img_url;
          const artist = n.rep.artist ?? "Unknown";
          const ang = n.ang_deg.toFixed(2);
          const pid = n.painting_info;
          return `
            <div class="neighbor-card" onclick="jumpToPainting('${pid.replaceAll("'", "\\'")}')">
              <img src="${img}" alt="neighbor">
              <div class="neighbor-meta">
                <div class="id" title="${pid}">${pid}</div>
                <div>${artist}</div>
                <div class="ang">Δ angle: ${ang}°</div>
              </div>
            </div>
          `;
        }).join('')}
      </div>
    `;

    details.innerHTML = `
      <div class="kv">
        <div class="kvrow"><div class="k">Painting ID</div><div class="v" title="${escapeHtml(d.painting_info)}">${escapeHtml(d.painting_info)}</div></div>
        <div class="kvrow"><div class="k">Artist</div><div class="v" title="${escapeHtml(d.artist)}">${escapeHtml(d.artist)}</div></div>
        <div class="kvrow"><div class="k">Genre</div><div class="v" title="${escapeHtml(d.genre)}">${escapeHtml(d.genre)}</div></div>
        <div class="kvrow"><div class="k">Dominance</div><div class="v">${escapeHtml(dom)}</div></div>
      </div>

      <div class="imgwrap">
        <img src="${d.img_url}" alt="painting" />
      </div>

      <div class="plotwrap">
        <div id="arrowPlotWrapper">
          <div id="arrowPlot"></div>
        </div>
      </div>
      <div class="smallhint">
        3D loads from <b>${PLOTS_PREFIX}/</b> using <code>${escapeHtml(safeFilename(d.painting_info))}.plot.json</code>
      </div>

      <div style="margin-top:12px; font-weight:800; color:#111827;">All ranks for this painting</div>
      <table class="ranktbl">
        <thead>
          <tr>
            <th>Rank</th>
            <th>Confidence</th>
            <th>dir</th>
          </tr>
        </thead>
        <tbody>
          ${rows}
        </tbody>
      </table>

      <div class="neighbors">
        <div class="neighbors-title">Similar paintings (closest lighting direction)</div>
        ${neighHTML}
      </div>
    `;

    loadArrowPlot3D(lastSelectedPainting);
    render3DPanel(d);
  }

  window.jumpToPainting = function(pid){
    const rep = repByPainting.get(pid);
    if (!rep) return;
    showDetailsBundle(rep);
  }

  function applyFilters() {
    const a = artistSelect.value;
    const g = genreSelect.value;
    const q = document.getElementById('searchBox').value.trim().toLowerCase();

    dataFiltered = dataAll.filter(d => {
      const okA = (a === '__ALL__') ? true : d.artist === a;
      const okG = (g === '__ALL__') ? true : d.genre === g;
      const hay1 = (d.painting_info ?? "").toLowerCase();
      const hay2 = (d.painting_rank_id ?? "").toLowerCase();
      const okQ = (!q) ? true : (hay1.includes(q) || hay2.includes(q));
      return okA && okG && okQ;
    });

    updateStatsPills();
    syncRankButtons();
    Plotly.react(plotDiv, makeTraces(dataFiltered), baseLayout, config);
    resetView();
  }

  window.applyFilters = applyFilters;
  window.resetView = resetView;
  window.clearSelection = clearSelection;

  updateStatsPills();
  syncRankButtons();

  Plotly.newPlot(plotDiv, makeTraces(dataFiltered), baseLayout, config).then(() => {
    Plotly.relayout(plotDiv, { 'scene.camera': cameraYZ });
    showRightPanel('2d');

    plotDiv.on('plotly_click', (evt) => {
      const d = evt.points?.[0]?.customdata;
      if (!d) return;
      showDetailsBundle(d);
      showRightPanel('2d');
    });
  });
</script>
</body>
</html>"""

    img_prefix_json = json.dumps(img_prefix)
    plots_prefix_json = json.dumps(plots_prefix)
    balls_prefix_json = json.dumps(balls_prefix)
    data_json = json.dumps(all_data_points, ensure_ascii=False, default=_to_json_safe)

    html = html.replace("__IMG_PREFIX__", img_prefix_json)
    html = html.replace("__PLOTS_PREFIX__", plots_prefix_json)
    html = html.replace("__BALLS_PREFIX__", balls_prefix_json)
    html = html.replace("__DATA_JSON__", data_json)
    return html


# ---------------------------- main ----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--file-name", type=str, required=True)

    parser.add_argument("--img-src", type=str, required=True, help="Folder containing painting images (can be nested)")
    parser.add_argument("--plots-src", type=str, default="", help="Folder containing *.plot.json (can be nested)")
    parser.add_argument("--balls-src", type=str, default="", help="Folder containing *_ev-XX.png (can be nested)")

    parser.add_argument("--out-dir", type=str, required=True, help="Folder to write site assets + index.html")

    parser.add_argument("--img-prefix", type=str, default="data")
    parser.add_argument("--plots-prefix", type=str, default="plots_embedded")
    parser.add_argument("--balls-prefix", type=str, default="balls")

    parser.add_argument("--csv-path", type=str, default="")
    parser.add_argument("--conf-mae-good", type=float, default=1.0)
    parser.add_argument("--conf-mae-bad", type=float, default=8.0)

    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    pkl_path = os.path.normpath(os.path.join(args.data_path, args.file_name))
    if not os.path.exists(pkl_path):
        print(f"Pickle not found: {pkl_path}")
        return

    out_dir = os.path.normpath(args.out_dir)
    ensure_dir(out_dir)

    print(f"Loading data from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        all_data = pickle.load(f)

    if not isinstance(all_data, list) or (all_data and not isinstance(all_data[0], dict)):
        print("Pickle isn't list[dict].")
        return

    desired_ev = _extract_ev_tag(args.file_name)

    # confidence
    if args.csv_path:
        conf_map = load_confidence_map_by_id_and_rank(
            args.csv_path, desired_ev=desired_ev,
            E_good=args.conf_mae_good, E_bad=args.conf_mae_bad,
            debug=args.debug
        )

        hit = 0
        miss = 0
        for d in all_data:
            rk_raw = d.get("rank", 1)
            try:
                rk = int(float(rk_raw))
            except Exception:
                rk = 1

            candidates = [
                d.get("painting_info", ""),
                d.get("painting_rank_id", ""),
                d.get("pose_info", ""),
                d.get("img_path", ""),
                d.get("file", ""),
            ]

            conf = None
            for cand in candidates:
                if not cand:
                    continue
                base_id, _ = _normalize_id_and_ev(cand)
                if not base_id:
                    continue
                v = conf_map.get((base_id, rk))
                if v is not None:
                    conf = float(v)
                    break

            d["confidence"] = conf
            if conf is None:
                miss += 1
            else:
                hit += 1
        print(f"confidence matched={hit} | missing={miss} | total={len(all_data)}")
    else:
        print("csv-path not provided; confidence disabled (will show '—').")

    # physical xyz
    used = []
    skipped = 0
    key_counter = Counter()

    for d in all_data:
        x, y, z, src = extract_direction_components(d)
        if x is None:
            skipped += 1
            continue

        if args.no_normalize:
            ux, uy, uz = x, y, z
            nraw = math.sqrt(x * x + y * y + z * z)
        else:
            ux, uy, uz, nraw = normalize_vec(x, y, z)
            if ux is None:
                skipped += 1
                continue

        d["x"] = float(ux)
        d["y"] = float(uy)
        d["z"] = float(uz)
        d["dir_norm_raw"] = None if (nraw is None or not math.isfinite(nraw)) else float(nraw)
        d["dir_src_keys"] = src

        key_counter[src] += 1
        used.append(d)

    print(f"Physical xyz applied to: {len(used)} points | skipped(no physical dir): {skipped}")
    if len(used) == 0:
        print("ERROR: 0 points had physical direction keys.")
        return

    print("Most common physical direction sources:")
    for k, v in key_counter.most_common(10):
        print(f"  {k}: {v}")

    # stage assets
    stage_painting_images(used, args.img_src, out_dir, args.img_prefix, debug=args.debug)
    if args.plots_src:
        stage_plots(args.plots_src, out_dir, args.plots_prefix, debug=args.debug)
    if args.balls_src:
        stage_balls(args.balls_src, out_dir, args.balls_prefix, debug=args.debug)

    # write HTML
    html_content = generate_html(
        used,
        img_prefix=args.img_prefix,
        plots_prefix=args.plots_prefix,
        balls_prefix=args.balls_prefix
    )

    out_html = os.path.join(out_dir, "index.html")
    with open(out_html, "w", encoding="utf-8") as f:
        f.write(html_content)

    print("\n" + "=" * 44)
    print(f"FINISH! Output file: {out_html}")
    print("Test locally from out-dir:")
    print(f'  cd "{out_dir}"')
    print("  python -m http.server 8000")
    print("  http://localhost:8000/index.html")


if __name__ == "__main__":
    main()
