#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, io, sys, re, math, tempfile, subprocess
from hashlib import md5

import pandas as pd
import streamlit as st
import folium
from streamlit_folium import st_folium

# ======================
# Config & simple styles
# ======================
SCRIPT_PATH = os.environ.get("SCHEDULER_SCRIPT", "scheduler.py")  # your unchanged script
DEFAULT_OUTPUT_NAME = "schedule_output.xlsx"
PAGE_TITLE = "Merchandiser Scheduler"
PAGE_ICON = "📅"
PRIMARY_COLOR = "#1b6ef3"

CSS = """
<style>
  .stApp { background: radial-gradient(1200px 800px at 20% -10%, #f8fbff 0%, white 55%) no-repeat; }
  header[data-testid="stHeader"] { background: transparent; }
  .title h1 { letter-spacing: .3px; }
  .pill { display:inline-block; padding:2px 8px; border-radius:999px; background:#eef2ff; color:#3730a3; font-size:.8rem; margin-left:6px; }
  .small-muted { color:#6b7280; font-size:.9rem; }
  .linkbar a { margin-right:12px; text-decoration:none; }
</style>
"""

# =================
# Cached sheet read
# =================
@st.cache_data(show_spinner=False)
def read_sheet(xlsx_bytes: bytes, sheet: str):
    with io.BytesIO(xlsx_bytes) as bio:
        return pd.read_excel(bio, sheet_name=sheet)

def _ensure_script_exists():
    if not os.path.exists(SCRIPT_PATH):
        st.error(f"Cannot find scheduler script at '{SCRIPT_PATH}'. Place your script here or set SCHEDULER_SCRIPT.")
        st.stop()

# ===========================
# CLI builder for your script
# ===========================
def build_cli_args(params: dict) -> list[str]:
    args = ["--input", params["input_path"], "--output", params["output_path"]]

    def add_flag(name: str, value):
        if isinstance(value, bool):
            if value:
                args.append(name)
        elif value not in (None, ""):
            args.extend([name, str(value)])

    add_flag("--weeks", params["weeks"])
    add_flag("--workdays", params["workdays"])
    add_flag("--daily_capacity", params["daily_capacity"])
    add_flag("--speed", params["speed"])
    if params["default_travel_set"]:
        add_flag("--default_travel", params["default_travel"])
    add_flag("--max_km_same_day", params["max_km_same_day"])
    if params["strict_same_merch"]:
        args.append("--strict_same_merch")
    if params["clusters"]:
        add_flag("--clusters", params["clusters"])
    add_flag("--cluster_mode", params["cluster_mode"])
    if params["cluster_radius_km"]:
        add_flag("--cluster_radius_km", params["cluster_radius_km"])
    add_flag("--distance_model", params["distance_model"])
    if params["freq_is_total"]:
        args.append("--freq_is_total")
    add_flag("--merge_utilization_threshold", params["merge_utilization_threshold"])
    if params["merge_cross_cluster"]:
        args.append("--merge_cross_cluster")
    add_flag("--cache_precision", params["cache_precision"])
    if params["verbose"]:
        args.append("--verbose")
    return args

# ===============
# Map helpers
# ===============
def week_key(day_str: str):
    m = re.search(r"Week(\d+)", str(day_str) if pd.notna(day_str) else "")
    return (int(m.group(1)) if m else 1, str(day_str))

def color_for_merch(name: str):
    # deterministic soft color per merch
    h = md5(str(name).encode("utf-8")).hexdigest()
    r = 100 + (int(h[0:2], 16) % 156)
    g = 100 + (int(h[2:4], 16) % 156)
    b = 100 + (int(h[4:6], 16) % 156)
    return "#{:02x}{:02x}{:02x}".format(r, g, b)

def to_numeric_latlon(df: pd.DataFrame):
    # Coerce non-numeric / comma-decimal
    for col in ["Lat", "Long"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.replace(",", ".", regex=False)
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df

def fit_bounds(df: pd.DataFrame):
    if df.empty or df["Lat"].isna().all() or df["Long"].isna().all():
        # world-ish
        return [[-60, -130], [60, 130]], [0.0, 0.0], 2
    lat_min, lat_max = float(df["Lat"].min()), float(df["Lat"].max())
    lon_min, lon_max = float(df["Long"].min()), float(df["Long"].max())
    center = [(lat_min + lat_max) / 2.0, (lon_min + lon_max) / 2.0]
    bounds = [[lat_min, lon_min], [lat_max, lon_max]]
    # crude zoom heuristic
    span = max(lat_max - lat_min, lon_max - lon_min, 0.0001)
    zoom = max(2, min(14, int(12 - math.log(max(span, 1e-4), 2) - 1)))
    return bounds, center, zoom

def build_leaflet(schedule_df: pd.DataFrame,
                  merch_filter, day_filter, cluster_filter, city_filter):
    df = schedule_df.copy()
    if "Store" not in df.columns and "Store Name" in df.columns:
        df = df.rename(columns={"Store Name": "Store"})
    df = to_numeric_latlon(df).dropna(subset=["Lat", "Long"])
    df["Merchandiser"] = df["Merchandiser"].astype(str)

    if merch_filter:
        df = df[df["Merchandiser"].isin(merch_filter)]
    if day_filter:
        df = df[df["Day"].astype(str).isin(day_filter)]
    if cluster_filter and "Cluster" in df.columns:
        df = df[df["Cluster"].isin(cluster_filter)]
    if city_filter and "City" in df.columns:
        df = df[df["City"].astype(str).isin(city_filter)]

    bounds, center, zoom = fit_bounds(df)
    fmap = folium.Map(location=center, zoom_start=zoom, tiles="CartoDB Positron")

    # Draw by (Merchandiser, Day)
    for (m_id, day), g in df.sort_values(["Merchandiser", "Day", "Seq"]) \
                            .groupby(["Merchandiser", "Day"], sort=False):
        color = color_for_merch(m_id)
        coords_latlon = g[["Lat", "Long"]].to_numpy().tolist()
        if len(coords_latlon) >= 2:
            folium.PolyLine(
                coords_latlon, color=color, weight=4, opacity=0.9,
                tooltip=f"{m_id} — {day}"
            ).add_to(fmap)
        for _, r in g.iterrows():
            folium.CircleMarker(
                location=[float(r["Lat"]), float(r["Long"])],
                radius=4, color=color, fill=True, fill_opacity=1.0,
                tooltip=f'{r.get("Store","")}\n{r.get("City","")}\n{m_id} • Seq {r.get("Seq","")}'
            ).add_to(fmap)

    # fit to bounds (if not degenerate)
    if bounds != [[-60, -130], [60, 130]]:
        fmap.fit_bounds(bounds, padding=(10, 10))
    return fmap, df

# =========================
# App routing & navigation
# =========================
st.set_page_config(page_title=PAGE_TITLE, page_icon=PAGE_ICON, layout="wide")
st.markdown(CSS, unsafe_allow_html=True)

def render_linkbar():
    current = dict(st.query_params)
    def _qs(d): return "&".join(f"{k}={v}" for k, v in d.items() if v is not None)
    st.markdown(
        f'<div class="linkbar">🔗 '
        f'<a href="?{_qs({**current, "view": "home"})}">Home</a> | '
        f'<a href="?{_qs({**current, "view": "map"})}">Map</a>'
        f'</div>', unsafe_allow_html=True
    )

view = st.query_params.get("view", "home")

# -------------
# MAP VIEW (Leaflet only)
# -------------
if view == "map":
    render_linkbar()
    st.markdown(
        '<div class="title"><h1>Map <span class="pill">routes & filters</span></h1>'
        '<p class="small-muted">Visualize routes and filter by cluster → city → merchandiser → day.</p></div>',
        unsafe_allow_html=True
    )

    # get latest schedule from session or allow upload
    xlsx_bytes = st.session_state.get("last_xlsx_bytes")
    if xlsx_bytes is None:
        up2 = st.file_uploader("Upload a generated Schedule .xlsx (sheet: 'Schedule')", type=["xlsx"], key="map_uploader")
        if up2 is not None:
            xlsx_bytes = up2.read()

    if xlsx_bytes is None:
        st.stop()

    try:
        sched_df = read_sheet(xlsx_bytes, "Schedule")
    except Exception as e:
        st.error(f"Could not read 'Schedule' sheet: {e}")
        st.stop()

    required_cols = {"Lat", "Long", "Merchandiser", "Day", "Seq"}
    if not required_cols.issubset(sched_df.columns):
        st.warning(f"'Schedule' must include: {sorted(required_cols)}.")
        st.dataframe(sched_df.head())
        st.stop()

    # ----------------- STRICTLY CASCADING FILTERS -----------------
    def multiselect_cascade(label, key, options, *, default_all=True, dependents=(), sort_key=None):
        opts = list(options) if options is not None else []
        if sort_key:
            opts = sorted(opts, key=sort_key)
        else:
            opts = sorted(opts)

        prev = st.session_state.get(key)
        if prev is None or any(v not in opts for v in prev):
            st.session_state[key] = opts if default_all else []

        val = st.multiselect(label, opts, default=st.session_state[key], key=key)

        if set(val) != set(prev or []):
            for dep in dependents:
                st.session_state.pop(dep, None)
            st.rerun()
        return val

    df_all = sched_df.copy()
    df_all["Merchandiser"] = df_all["Merchandiser"].astype(str)
    if "City" in df_all.columns:
        df_all["City"] = df_all["City"].astype(str)
    if "Cluster" in df_all.columns:
        df_all["Cluster"] = pd.to_numeric(df_all["Cluster"], errors="coerce")

    c1, c2, c3, c4 = st.columns([1, 1, 1, 1])

    with c1:
        cluster_opts = []
        if "Cluster" in df_all.columns:
            cluster_opts = [int(c) for c in df_all["Cluster"].dropna().unique().tolist()]
        sel_clusters = multiselect_cascade(
            "Clusters", "sel_clusters", cluster_opts,
            default_all=True,
            dependents=("sel_cities", "sel_merch", "sel_days"),
        )
    df_c1 = df_all[df_all["Cluster"].isin(sel_clusters)] if sel_clusters else df_all

    with c2:
        city_opts = df_c1["City"].dropna().unique().tolist() if "City" in df_c1.columns else []
        sel_cities = multiselect_cascade(
            "Cities", "sel_cities", city_opts,
            default_all=True,
            dependents=("sel_merch", "sel_days"),
        )
    df_c2 = df_c1[df_c1["City"].isin(sel_cities)] if sel_cities else df_c1

    with c3:
        merch_opts = df_c2["Merchandiser"].dropna().unique().tolist()
        sel_merch = multiselect_cascade(
            "Merchandisers", "sel_merch", merch_opts,
            default_all=True,
            dependents=("sel_days",),
        )
    df_c3 = df_c2[df_c2["Merchandiser"].isin(sel_merch)] if sel_merch else df_c2

    with c4:
        day_opts = df_c3["Day"].astype(str).dropna().unique().tolist()
        sel_days = multiselect_cascade(
            "Days", "sel_days", day_opts,
            default_all=True,
            dependents=(),
            sort_key=week_key,
        )
    # ------------------------------------------------------

    # Build & render Leaflet map
    fmap, filtered_df = build_leaflet(
        sched_df,
        merch_filter=sel_merch,
        day_filter=sel_days,
        cluster_filter=sel_clusters,
        city_filter=sel_cities
    )
    st_folium(fmap, width=None, height=650)

    st.caption(
        f"{len(filtered_df)} stops shown • "
        f"{len(filtered_df[['Merchandiser','Day']].drop_duplicates())} day-routes."
    )

# -------------
# HOME / GENERATE VIEW
# -------------
else:
    render_linkbar()
    st.markdown(
        '<div class="title"><h1>Merchandiser Scheduler <span class="pill">no code changes</span></h1>'
        '<p class="small-muted">Upload your Excel, configure options, generate the plan — then jump to the Map view.</p></div>',
        unsafe_allow_html=True
    )

    col1, col2 = st.columns([1.4, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload input Excel", type=["xlsx", "xls"],
            help="Must include the expected columns used by your script."
        )
        st.caption("Required: City, Store Name, Estimated Duration In a store, Frequency per week. Optional: Lat, Long, Cluster.")

    # Sidebar params
    with st.sidebar:
        st.header("Parameters")
        weeks = st.number_input("Weeks", min_value=1, max_value=52, value=1)
        workdays = st.number_input("Workdays per week", min_value=1, max_value=7, value=5)
        daily_capacity = st.number_input("Daily capacity (minutes)", min_value=60, max_value=1440, value=480)
        speed = st.number_input("Avg speed (mph)", min_value=1.0, max_value=200.0, value=30.0)

        default_travel_set = st.toggle("Use fixed travel minutes per hop?", value=False)
        default_travel = None
        if default_travel_set:
            default_travel = st.number_input("Fixed travel minutes per hop", min_value=0.0, max_value=300.0, value=0.0)

        max_km_same_day = st.number_input("Max miles between same-day visits", min_value=0.0, max_value=1000.0, value=50.0)
        strict_same_merch = st.checkbox("Strict same merch per store", value=False)

        st.divider(); st.subheader("Clustering")
        cluster_mode = st.selectbox("Cluster mode", options=["kmeans","radius"], index=0)
        clusters = st.text_input("KMeans k (leave blank for auto)", value="")
        cluster_radius_km = st.text_input("Radius (miles) for DBSCAN (if radius mode)", value="")

        st.divider(); st.subheader("Distance + Frequency")
        distance_model = st.selectbox("Distance model", options=["haversine","euclidean"], index=0)
        freq_is_total = st.checkbox("'Frequency per week' is TOTAL to schedule (ignore Weeks)")

        st.divider(); st.subheader("Merge underutilized")
        merge_utilization_threshold = st.slider("Utilization threshold (0 disables)", 0.0, 1.0, 0.0, 0.05)
        merge_cross_cluster = st.checkbox("Allow cross-cluster merge", value=False)

        st.divider()
        cache_precision = st.slider("Haversine cache precision (decimals)", 0, 7, 5)
        verbose = st.checkbox("Verbose console logs", value=True)

    st.markdown("### Run\nConfigure options, then click Run.")
    run = st.button("Run Scheduler", type="primary", use_container_width=True, disabled=(uploaded is None))

    if run and uploaded is not None:
        _ensure_script_exists()
        with st.status("Running scheduler…", expanded=True) as status:
            tmpdir = tempfile.mkdtemp(prefix="sched_app_")
            in_path = os.path.join(tmpdir, uploaded.name)
            out_path = os.path.join(tmpdir, DEFAULT_OUTPUT_NAME)
            with open(in_path, "wb") as f:
                f.write(uploaded.getbuffer())

            params = dict(
                input_path=in_path, output_path=out_path,
                weeks=weeks, workdays=workdays, daily_capacity=daily_capacity,
                speed=speed, default_travel_set=default_travel_set, default_travel=default_travel,
                max_km_same_day=max_km_same_day, strict_same_merch=strict_same_merch,
                clusters=clusters.strip(), cluster_mode=cluster_mode, cluster_radius_km=cluster_radius_km.strip(),
                distance_model=distance_model, freq_is_total=freq_is_total,
                merge_utilization_threshold=merge_utilization_threshold, merge_cross_cluster=merge_cross_cluster,
                cache_precision=cache_precision, verbose=verbose,
            )
            cli_args = [sys.executable, SCRIPT_PATH] + build_cli_args(params)
            st.write("**Command**:", " ".join(cli_args))

            proc = subprocess.Popen(cli_args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
            lines, placeholder = [], st.empty()
            for line in iter(proc.stdout.readline, ''):
                lines.append(line)
                placeholder.code("".join(lines), language="bash")
            proc.stdout.close()
            ret = proc.wait()

            if ret != 0:
                status.update(label=f"Script exited with code {ret}", state="error", expanded=True)
                st.error("The scheduler reported an error. Check logs above.")
            else:
                status.update(label="Done!", state="complete", expanded=False)
                with open(out_path, "rb") as f:
                    xlsx_bytes = f.read()
                st.session_state["last_xlsx_bytes"] = xlsx_bytes

                st.success("Schedule generated.")
                st.download_button("⬇️ Download Excel (Schedule, Summary, DailyKM, Weekly_KM)",
                                   data=xlsx_bytes, file_name=DEFAULT_OUTPUT_NAME,
                                   mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")