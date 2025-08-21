#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Merchandiser scheduler — performance-optimized edition (v3)
- Deterministic placement (stable hash)
- O(1) day checks with pre-computed day state
- Optional fast heuristics and adjacency-only distance scope
- Vectorized hop-km computation
- Unscheduled reporting when constraints make a visit impossible
- Effort-based merchandiser counts + Peak-day concurrency headcount
- Merge helper bugfix (row-wise same store+city)
- **CHANGE**: `--strict_same_merch` is now a HARD constraint that also respects
  `daily_capacity`. Once a store is assigned to a merch, all its visits must go
  to that merch and will **not** exceed per-day capacity; otherwise they are
  reported in **Unscheduled** (no auto-spawned merch and no over-capacity force).
"""

import argparse
import logging
import math
import sys
from collections import defaultdict
from functools import lru_cache
from hashlib import md5
from typing import List, Tuple, Optional, Dict, Any

# ---- Friendly dependency checks ----
try:
    import numpy as np
    import pandas as pd
except ImportError as e:
    print("This script requires numpy and pandas. Install with: pip install numpy pandas", file=sys.stderr)
    raise

try:
    from geopy.distance import geodesic
except ImportError:
    print("This script requires geopy. Install with: pip install geopy", file=sys.stderr)
    raise

try:
    from sklearn.cluster import KMeans, DBSCAN
except ImportError:
    print("This script requires scikit-learn. Install with: pip install scikit-learn", file=sys.stderr)
    raise

# =========================
# Distance helpers + caching
# =========================

@lru_cache(maxsize=None)
def cached_distance(lat1, lon1, lat2, lon2) -> float:
    return geodesic((float(lat1), float(lon1)), (float(lat2), float(lon2))).km


def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    phi1, lam1, phi2, lam2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dphi = phi2 - phi1
    dlam = lam2 - lam1
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlam / 2) ** 2
    return 2 * R * math.asin(math.sqrt(a))


CACHED_PREC = 5


def _set_cache_precision(n: int):
    global CACHED_PREC
    CACHED_PREC = max(0, int(n))


@lru_cache(maxsize=500_000)
def _cached_haversine_key(lat1_r, lon1_r, lat2_r, lon2_r) -> float:
    return _haversine_km(lat1_r, lon1_r, lat2_r, lon2_r)


def cached_haversine(lat1, lon1, lat2, lon2) -> float:
    l1 = round(float(lat1), CACHED_PREC)
    o1 = round(float(lon1), CACHED_PREC)
    l2 = round(float(lat2), CACHED_PREC)
    o2 = round(float(lon2), CACHED_PREC)
    return _cached_haversine_key(l1, o1, l2, o2)


def _stable_id(s: str) -> int:
    return int(md5(s.encode("utf-8")).hexdigest()[:8], 16)


def dist_km(lat1, lon1, lat2, lon2, model: str) -> float:
    if model == "haversine":
        return cached_haversine(lat1, lon1, lat2, lon2)
    return cached_distance(lat1, lon1, lat2, lon2)


# =========================
# Data prep + clustering
# =========================

def load_and_prepare_data(
    file_path: str,
    clusters_override: Optional[int],
    *,
    cluster_mode: str,
    cluster_radius_km: Optional[float],
    distance_model: str,  # kept for signature compatibility
    seed: int,
) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    for col in ["Lat", "Long", "Estimated Duration In a store", "Estimated Travel Time", "Frequency", "Cluster"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["City", "Store Name", "Estimated Duration In a store", "Frequency"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["Estimated Duration In a store", "Frequency"]).copy()

    if "Cluster" in df.columns and df["Cluster"].notna().any():
        df["Cluster"] = df["Cluster"].fillna(0).astype(int)
        logging.info("ℹ️ Using existing Cluster column from input.")
        return df

    gps_df = df.dropna(subset=["Lat", "Long"]).copy()
    missing_df = df[df["Lat"].isna() | df["Long"].isna()].copy()

    if gps_df.empty:
        df["Cluster"] = 0
        logging.info("⚠️ No GPS rows; assigning all to Cluster 0.")
        return df

    if cluster_mode == "radius":
        if not cluster_radius_km or cluster_radius_km <= 0:
            raise ValueError("--cluster_radius_km must be > 0 for cluster_mode=radius")
        coords_deg = gps_df[["Lat", "Long"]].to_numpy(dtype=float)
        coords_rad = np.radians(coords_deg)
        eps_rad = float(cluster_radius_km) / 6371.0088
        db = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine", algorithm="ball_tree")
        labels = db.fit_predict(coords_rad)
        gps_df["Cluster"] = labels
        logging.info(f"📦 Radius clustering (DBSCAN) eps={cluster_radius_km} km → clusters={labels.max()+1}")
    else:
        if clusters_override is not None and int(clusters_override) > 0:
            k = int(clusters_override)
        else:
            k = max(1, int(round(len(gps_df) ** 0.5)))
        unique_coords = gps_df.drop_duplicates(subset=["Lat", "Long"])
        k = max(1, min(k, len(unique_coords)))
        logging.info(f"📦 K-Means clustering with k={k}")
        gps_df["Cluster"] = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(gps_df[["Lat", "Long"]])

    if not missing_df.empty:
        gps_df["_CityNorm"] = gps_df["City"].astype(str).str.strip().str.upper()
        missing_df["_CityNorm"] = missing_df["City"].astype(str).str.strip().str.upper()
        city_map = gps_df.groupby("_CityNorm")["Cluster"].agg(lambda x: x.value_counts().idxmax())
        missing_df["Cluster"] = missing_df["_CityNorm"].map(city_map)
        most_common = gps_df["Cluster"].mode().iloc[0]
        missing_df["Cluster"] = missing_df["Cluster"].fillna(most_common)
        gps_df.drop(columns=["_CityNorm"], inplace=True)
        missing_df.drop(columns=["_CityNorm"], inplace=True)

    return pd.concat([gps_df, missing_df], ignore_index=True)


# =========================
# Scheduler
# =========================

def schedule_with_constraints(
    full_df: pd.DataFrame,
    *,
    weeks: int,
    workdays: int,
    daily_capacity: float,
    avg_speed_kmph: float,
    default_travel: Optional[float],
    max_km_same_day: float,
    strict_same_merch: bool,
    distance_model: str,
    frequency_period: str,
    month_weeks: float,
    fixed_merch_names: Optional[List[str]] = None,
    fixed_strict_capacity: bool = False,
    distance_scope: str = "all_day",
    fast: bool = False,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = full_df.copy()

    base_freq = pd.to_numeric(df["Frequency"], errors="coerce").fillna(0.0)
    if "Frequency Period" in df.columns:
        row_period = df["Frequency Period"].astype(str).str.strip().str.lower().where(lambda s: s.isin(["week", "month"]))
    else:
        row_period = pd.Series([None] * len(df))

    mw = float(month_weeks) if month_weeks and month_weeks > 0 else 4.345

    def _factor(i: int) -> float:
        p = row_period.iat[i] if row_period.iat[i] in ("week", "month") else frequency_period
        return (weeks / mw) if p == "month" else float(weeks)

    factors = np.array([_factor(i) for i in range(len(df))], dtype=float)
    df["Total Visits"] = np.maximum(0, np.rint(base_freq.to_numpy() * factors)).astype(int)

    eff_period = []
    for i in range(len(df)):
        p = row_period.iat[i] if row_period.iat[i] in ("week", "month") else frequency_period
        eff_period.append(p)
    df["_FreqPeriod"] = eff_period

    weekdays_all = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    workdays = max(1, min(int(workdays), 7))
    day_names = weekdays_all[:workdays]
    days = [f"Week{w+1}-{d}" for w in range(weeks) for d in day_names]
    L = len(days)
    if L == 0:
        return (
            pd.DataFrame(columns=["Store", "City", "Cluster", "Merchandiser", "Day", "Seq", "Duration", "ActualTravel", "Lat", "Long"]),
            pd.DataFrame(columns=["Store", "City", "Cluster", "Reason", "Duration", "Lat", "Long"]),
        )

    day_to_index = {d: i for i, d in enumerate(days)}

    def week_of(day_label: str) -> int:
        return int(day_label.split("-")[0].replace("Week", ""))

    def evenly_spaced_targets(n_visits: int, total_slots: int, store_key: str) -> List[int]:
        if n_visits <= 0:
            return []
        step = total_slots / float(n_visits)
        frac = (_stable_id(store_key) % 997) / 997.0
        start = frac * min(step, total_slots)
        idxs = []
        for i in range(n_visits):
            x = (i + 0.5) * step + start
            idx = int(min(total_slots - 1, max(0, math.floor(x))))
            idxs.append(idx)
        for i in range(1, len(idxs)):
            if idxs[i] <= idxs[i - 1]:
                idxs[i] = min(total_slots - 1, idxs[i - 1] + 1)
        return idxs

    def indices_by_closeness(target_idx: int, total: int) -> List[int]:
        order, used = [], set()
        for d in range(total):
            for sgn in (0, -1, 1):
                if d == 0 and sgn != 0:
                    continue
                j = target_idx + (d if sgn == 1 else (-d if sgn == -1 else 0))
                if 0 <= j < total and j not in used:
                    order.append(j)
                    used.add(j)
        return order

    def even_weeks(n_visits: int, n_weeks: int) -> List[int]:
        """Return 1-based week indices spaced across the period.
        Ensures unique weeks when n_visits <= n_weeks (one per week),
        and spreads as evenly as possible otherwise.
        """
        if n_visits <= 0 or n_weeks <= 0:
            return []
        anchors = np.linspace(0, n_weeks - 1, num=n_visits, endpoint=True)
        weeks_list = [int(round(a)) + 1 for a in anchors]
        weeks_list = [min(max(1, w), n_weeks) for w in weeks_list]
        if n_visits <= n_weeks:
            used, out = set(), []
            for w in weeks_list:
                if w not in used:
                    out.append(w); used.add(w); continue
                best = None
                for d in range(1, n_weeks):
                    for s in (-1, 1):
                        cand = w + s * d
                        if 1 <= cand <= n_weeks and cand not in used:
                            best = cand; break
                    if best is not None:
                        break
                out.append(best if best is not None else w)
                used.add(out[-1])
            weeks_list = out
        return weeks_list

    fixed_mode = bool(fixed_merch_names)
    if fixed_mode:
        merch_schedules: Dict[str, Dict[str, Dict[str, Any]]] = {
            name: {d: {"total": 0.0, "visits": [], "stores": set(), "last_lat": None, "last_lon": None} for d in days}
            for name in fixed_merch_names
        }
    else:
        merch_schedules = {}
    store_merch_map: Dict[str, str] = {}

    store_assigned_idx: Dict[str, List[int]] = defaultdict(list)
    store_week_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    unscheduled: List[Dict[str, Any]] = []

    for cluster_id, group in df.groupby("Cluster"):
        tasks_by_store: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)

        for _, row in group.iterrows():
            total_v = int(row["Total Visits"]) if pd.notna(row["Total Visits"]) else 0
            if total_v <= 0:
                continue
            store_name = row.get("Store Name", "")
            city_name = row.get("City", "")
            store_key = f"{store_name}|{city_name}"
            is_month = (str(row.get("_FreqPeriod", "week")).lower() == "month")

            if is_month:
                target_weeks = even_weeks(total_v, weeks)
                day_rot = _stable_id(store_key) % workdays
                day_order = day_names[day_rot:] + day_names[:day_rot]
                for wk in target_weeks:
                    week_days = [f"Week{wk}-{d}" for d in day_order]
                    tasks_by_store[(store_name, city_name)].append({
                        "Store": store_name, "City": city_name, "Cluster": int(cluster_id),
                        "Duration": float(row["Estimated Duration In a store"]),
                        "Lat": row.get("Lat", np.nan), "Long": row.get("Long", np.nan),
                        "PreferredDaysOrdered": ordered_days, "Monthly": True, "TotalV": total_v,
                    })
            else:
                for _ in range(total_v):
                    tasks_by_store[(store_name, city_name)].append({
                        "Store": store_name, "City": city_name, "Cluster": int(cluster_id),
                        "Duration": float(row["Estimated Duration In a store"]),
                        "Lat": row.get("Lat", np.nan), "Long": row.get("Long", np.nan),
                        "PreferredDaysOrdered": None, "Monthly": False, "TotalV": total_v,
                    })

        merch_id_counter = 1

        for (store, city), visits in tasks_by_store.items():
            skey = f"{store}|{city}"
            base_rot = _stable_id(skey) % L

            preferred_merch = store_merch_map.get(skey) if strict_same_merch else None
            if strict_same_merch and preferred_merch and preferred_merch in merch_schedules:
                merch_pool = [preferred_merch]  # HARD: only this merch is allowed now
            elif fixed_mode:
                merch_pool = fixed_merch_names
            else:
                if preferred_merch and preferred_merch in merch_schedules:
                    merch_pool = [preferred_merch]
                else:
                    merch_pool = list(merch_schedules.keys()) + [f"Merch_{merch_id_counter}"]

            total_v_for_store = len(visits)
            max_per_week_cap = math.ceil(total_v_for_store / max(1, weeks))
            min_gap_slots = max(1, L // max(1, total_v_for_store))

            for visit in visits:
                assigned = False

                for merch in merch_pool:
                    if merch not in merch_schedules:
                        merch_schedules[merch] = {d: {"total": 0.0, "visits": [], "stores": set(), "last_lat": None, "last_lon": None} for d in days}
                        merch_id_counter += 1
                    cal = merch_schedules[merch]

                    candidate_day_lists = []
                    if visit.get("PreferredDaysOrdered"):
                        candidate_day_lists.append(visit["PreferredDaysOrdered"])
                    candidate_day_lists.append([days[(base_rot + k) % L] for k in range(L)])

                    def can_place(day_label: str, *, enforce_gap=True, enforce_wcap=True, enforce_dist=True, enforce_cap=True) -> Tuple[bool, float]:
                        slot = cal[day_label]
                        if (store, city) in slot["stores"]:
                            return (False, 0.0)
                        if visit["Monthly"]:
                            if enforce_wcap:
                                wk = week_of(day_label)
                                if store_week_counts[skey][wk] >= max_per_week_cap:
                                    return (False, 0.0)
                            if enforce_gap and store_assigned_idx[skey]:
                                idx = day_to_index[day_label]
                                if min(abs(idx - j) for j in store_assigned_idx[skey]) < min_gap_slots:
                                    return (False, 0.0)
                        if enforce_dist and max_km_same_day and slot["visits"] and pd.notna(visit["Lat"]) and pd.notna(visit["Long"]):
                            if distance_scope == "adjacent":
                                if slot["last_lat"] is not None and slot["last_lon"] is not None:
                                    if dist_km(slot["last_lat"], slot["last_lon"], visit["Lat"], visit["Long"], model=distance_model) > max_km_same_day:
                                        return (False, 0.0)
                            else:
                                for v in slot["visits"]:
                                    if all(pd.notna([v["Lat"], v["Long"], visit["Lat"], visit["Long"]])):
                                        if dist_km(v["Lat"], v["Long"], visit["Lat"], visit["Long"], model=distance_model) > max_km_same_day:
                                            return (False, 0.0)
                        if default_travel is not None:
                            travel_min = float(default_travel)
                        else:
                            if slot["visits"] and all(pd.notna([slot["last_lat"], slot["last_lon"], visit["Lat"], visit["Long"]])):
                                km = dist_km(slot["last_lat"], slot["last_lon"], visit["Lat"], visit["Long"], model=distance_model)
                                travel_min = (km / avg_speed_kmph) * 60.0
                            else:
                                travel_min = 0.0
                        total_time = visit["Duration"] + travel_min
                        if enforce_cap and (slot["total"] + total_time > daily_capacity):
                            return (False, travel_min)
                        return (True, travel_min)

                    def place(day_label: str, travel_min: float):
                        v = {**visit, "ActualTravel": float(travel_min), "OverCapacity": 0}
                        slot = cal[day_label]
                        slot["visits"].append(v)
                        slot["total"] += visit["Duration"] + float(travel_min)
                        slot["stores"].add((store, city))
                        if pd.notna(visit["Lat"]) and pd.notna(visit["Long"]):
                            slot["last_lat"], slot["last_lon"] = float(visit["Lat"]), float(visit["Long"])
                        store_merch_map[skey] = merch
                        if visit["Monthly"]:
                            idx = day_to_index[day_label]
                            store_assigned_idx[skey].append(idx)
                            store_week_counts[skey][week_of(day_label)] += 1

                    def sorted_by_load(day_list: List[str]) -> List[str]:
                        return sorted(day_list, key=lambda d: cal[d]["total"]) if fast else day_list

                    placed = False
                    # 1) strict everything (includes capacity)
                    for lst in candidate_day_lists:
                        for dlab in sorted_by_load(lst):
                            if fast and (cal[dlab]["total"] + visit["Duration"] > daily_capacity) and (default_travel in (0, None)):
                                continue
                            ok, tmin = can_place(dlab, enforce_gap=True, enforce_wcap=True, enforce_dist=True, enforce_cap=True)
                            if ok:
                                place(dlab, tmin)
                                assigned = True
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break

                    # 2) relax distance, keep capacity
                    for lst in candidate_day_lists:
                        for dlab in sorted_by_load(lst):
                            ok, tmin = can_place(dlab, enforce_gap=True, enforce_wcap=True, enforce_dist=False, enforce_cap=True)
                            if ok:
                                place(dlab, tmin)
                                assigned = True
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break

                    # 3) relax weekly cap (keep gap), keep distance+capacity
                    for lst in candidate_day_lists:
                        for dlab in sorted_by_load(lst):
                            ok, tmin = can_place(dlab, enforce_gap=True, enforce_wcap=False, enforce_dist=True, enforce_cap=True)
                            if ok:
                                place(dlab, tmin)
                                assigned = True
                                placed = True
                                break
                        if placed:
                            break
                    if placed:
                        break

                # STRICT SAME MERCH hard rule: if this store already had a merch and we couldn't place
                if strict_same_merch and store_merch_map.get(skey) and not assigned:
                    unscheduled.append({
                        "Store": visit["Store"], "City": visit["City"], "Cluster": int(cluster_id),
                        "Reason": "strict_same_merch hard: no slot on assigned merch within daily capacity",
                        "Duration": float(visit["Duration"]), "Lat": visit.get("Lat", np.nan), "Long": visit.get("Long", np.nan),
                    })
                    continue  # next visit

                # FINAL FALLBACKS (fixed_mode only): choose best place but still CAPACITY-RESPECTING
                if not assigned and fixed_mode:
                    best = None
                    for merch in (fixed_merch_names or []):
                        cal = merch_schedules[merch]
                        for dlab in days:
                            slot = cal[dlab]
                            if (store, city) in slot["stores"]:
                                continue
                            ok, tmin = can_place(dlab, enforce_gap=False, enforce_wcap=False, enforce_dist=False, enforce_cap=True)
                            if not ok:
                                continue
                            added = slot["total"] + visit["Duration"] + (tmin if not math.isnan(tmin) else 0.0)
                            if (best is None) or (added < best[0]):
                                best = (added, merch, dlab, tmin)
                    if best is not None:
                        _, merch, dlab, tmin = best
                        place(dlab, tmin)
                        assigned = True

                # Not assigned and not fixed_mode → create a new merch and place on its lightest day (capacity will be respected by can_place)
                if not assigned and not fixed_mode:
                    new_merch = f"Merch_{len(merch_schedules) + 1}"
                    merch_schedules[new_merch] = {d: {"total": 0.0, "visits": [], "stores": set(), "last_lat": None, "last_lon": None} for d in days}
                    cal = merch_schedules[new_merch]
                    lightest_day = min(days, key=lambda d: cal[d]["total"])
                    ok, tmin = True, float(default_travel or 0.0)  # first hop
                    # verify capacity
                    if cal[lightest_day]["total"] + visit["Duration"] + tmin <= daily_capacity:
                        v = {**visit, "ActualTravel": tmin, "OverCapacity": 0}
                        cal[lightest_day]["visits"].append(v)
                        cal[lightest_day]["total"] += visit["Duration"] + tmin
                        cal[lightest_day]["stores"].add((store, city))
                        if pd.notna(visit["Lat"]) and pd.notna(visit["Long"]):
                            cal[lightest_day]["last_lat"], cal[lightest_day]["last_lon"] = float(visit["Lat"]), float(visit["Long"])
                        store_merch_map[skey] = new_merch
                        assigned = True
                    else:
                        # even a fresh merch couldn't take it under capacity ⇒ unscheduled
                        unscheduled.append({
                            "Store": visit["Store"], "City": visit["City"], "Cluster": int(cluster_id),
                            "Reason": "no capacity even on fresh merch (duration too large)",
                            "Duration": float(visit["Duration"]), "Lat": visit.get("Lat", np.nan), "Long": visit.get("Long", np.nan),
                        })

    # ---- flatten to dataframe ----
    all_rows: List[Dict[str, Any]] = []
    for merch, plans in merch_schedules.items():
        for day, info in plans.items():
            for seq, visit in enumerate(info["visits"], start=1):
                rec = visit.copy()
                rec.update({"Merchandiser": merch, "Day": day, "Seq": seq})
                all_rows.append(rec)

    out = pd.DataFrame(all_rows)
    return out, pd.DataFrame(unscheduled)


# =========================
# Merge underutilized merchs (kept for non-fixed mode)
# =========================

def merge_underutilized(schedule_df: pd.DataFrame, *, daily_capacity: float, max_km_same_day: float, distance_model: str, threshold: float, cross_cluster: bool = False) -> pd.DataFrame:
    if threshold <= 0:
        return schedule_df
    df = schedule_df.copy().reset_index(drop=True)
    df["_VisitMin"] = df["Duration"].astype(float) + pd.to_numeric(df.get("ActualTravel", 0.0), errors="coerce").fillna(0.0)

    def day_load(merch, day) -> float:
        sel = (df["Merchandiser"] == merch) & (df["Day"] == day)
        return float(df.loc[sel, "_VisitMin"].sum())

    def ok_same_day(target_merch, target_day, row) -> bool:
        if day_load(target_merch, target_day) + float(row["_VisitMin"]) > daily_capacity:
            return False
        sel = (df["Merchandiser"] == target_merch) & (df["Day"] == target_day)
        same_store_same_city = ((df.loc[sel, "Store"] == row["Store"]) & (df.loc[sel, "City"] == row["City"]))
        if same_store_same_city.any():
            return False
        if max_km_same_day and max_km_same_day > 0:
            coords = df.loc[sel, ["Lat", "Long"]].dropna()
            if not coords.empty and pd.notna(row["Lat"]) and pd.notna(row["Long"]):
                for lat, lon in coords.itertuples(index=False, name=None):
                    if dist_km(lat, lon, row["Lat"], row["Long"], model=distance_model) > max_km_same_day:
                        return False
        return True

    changed_any = False

    def cluster_masks(_df, cross):
        if cross:
            return [("ALL", np.ones(len(_df), dtype=bool))]
        else:
            ids = sorted(_df["Cluster"].dropna().unique().tolist())
            return [(cid, _df["Cluster"] == cid) for cid in ids]

    for cluster_id, c_mask in cluster_masks(df, cross_cluster):
        improved = True
        while improved:
            improved = False
            daily = df.loc[c_mask].groupby(["Merchandiser", "Day"], as_index=False)["_VisitMin"].sum().rename(columns={"_VisitMin": "DailyMin"})
            if daily.empty:
                continue
            util = (daily.groupby("Merchandiser")["DailyMin"].mean() / daily_capacity).sort_values()
            order = util.index.tolist()
            order.sort(key=lambda m: (len(df.loc[c_mask & (df["Merchandiser"] == m)]), util.get(m, 1.0)))
            for low_merch in order:
                if util.get(low_merch, 1.0) >= threshold:
                    continue
                low_rows = df.loc[c_mask & (df["Merchandiser"] == low_merch)]
                if low_rows.empty:
                    continue
                candidates_mask = c_mask & (df["Merchandiser"] != low_merch)
                targets = df.loc[candidates_mask, "Merchandiser"].drop_duplicates().tolist()
                moved_any = False
                for idx in low_rows.index.tolist():
                    row = df.loc[idx]
                    placed = False
                    for tgt in targets:
                        cand_days = df.loc[(df["Merchandiser"] == tgt) & c_mask, "Day"].drop_duplicates().tolist()
                        cand_days.sort(key=lambda d: float(df.loc[(df["Merchandiser"] == tgt) & (df["Day"] == d), "_VisitMin"].sum()))
                        for tgt_day in cand_days:
                            if ok_same_day(tgt, tgt_day, row):
                                df.at[idx, "Merchandiser"] = tgt
                                df.at[idx, "Day"] = tgt_day
                                placed = True
                                moved_any = True
                                changed_any = True
                                break
                        if placed:
                            break
                if moved_any and df.loc[c_mask & (df["Merchandiser"] == low_merch)].empty:
                    improved = True

    if changed_any:
        df = df.sort_values(["Merchandiser", "Day", "Seq"]).copy()
        df["Seq"] = df.groupby(["Merchandiser", "Day"]).cumcount() + 1

    return df.drop(columns=["_VisitMin"], errors="ignore")


# =========================
# Travel km per day & weekly totals
# =========================

def compute_day_km(sched_df: pd.DataFrame, *, distance_model: str):
    df = sched_df.sort_values(["Merchandiser", "Day", "Seq"]).copy()
    df["PrevLat"] = df.groupby(["Merchandiser", "Day"])["Lat"].shift()
    df["PrevLon"] = df.groupby(["Merchandiser", "Day"])["Long"].shift()
    mask = df[["Lat", "Long", "PrevLat", "PrevLon"]].notna().all(axis=1)
    df["HopKM"] = 0.0
    if mask.any():
        if distance_model == "haversine":
            R = 6371.0088
            phi1 = np.radians(df.loc[mask, "PrevLat"].astype(float).to_numpy())
            phi2 = np.radians(df.loc[mask, "Lat"].astype(float).to_numpy())
            dphi = phi2 - phi1
            lam1 = np.radians(df.loc[mask, "PrevLon"].astype(float).to_numpy())
            lam2 = np.radians(df.loc[mask, "Long"].astype(float).to_numpy())
            dlam = lam2 - lam1
            a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2) ** 2
            df.loc[mask, "HopKM"] = 2 * R * np.arcsin(np.sqrt(a))
        else:
            idx = df.index[mask]
            for i in idx:
                df.at[i, "HopKM"] = dist_km(df.at[i, "PrevLat"], df.at[i, "PrevLon"], df.at[i, "Lat"], df.at[i, "Long"], model=distance_model)
    day_km = df.groupby(["Merchandiser", "Day"], as_index=False)["HopKM"].sum().rename(columns={"HopKM": "DayKM"})
    merch_km = day_km.groupby("Merchandiser", as_index=False)["DayKM"].sum().rename(columns={"DayKM": "TotalKM"})
    return df.drop(columns=["PrevLat", "PrevLon"]), day_km, merch_km


# =========================
# Helpers
# =========================

def compute_expected_total_visits(src_df: pd.DataFrame, weeks: int, frequency_period: str, month_weeks: float) -> int:
    df = src_df.copy()
    base = pd.to_numeric(df["Frequency"], errors="coerce").fillna(0.0)
    if "Frequency Period" in df.columns:
        row_period = df["Frequency Period"].astype(str).str.strip().str.lower().where(lambda s: s.isin(["week", "month"]))
    else:
        row_period = pd.Series([None] * len(df))
    mw = float(month_weeks) if month_weeks and month_weeks > 0 else 4.345

    def _factor(i: int) -> float:
        p = row_period.iat[i] if row_period.iat[i] in ("week", "month") else frequency_period
        return (weeks / mw) if p == "month" else float(weeks)

    factors = np.array([_factor(i) for i in range(len(df))], dtype=float)
    total = np.maximum(0, np.rint(base.to_numpy() * factors)).astype(int).sum()
    return int(total)


def compute_effort_based_merchandisers(sched_df: pd.DataFrame, unsched_df: pd.DataFrame, *, daily_capacity: float, workdays: int, weeks: int, default_travel: Optional[float]) -> Tuple[int, int, float, float]:
    scheduled_minutes = float((pd.to_numeric(sched_df.get("Duration", 0.0), errors="coerce").fillna(0.0) + pd.to_numeric(sched_df.get("ActualTravel", 0.0), errors="coerce").fillna(0.0)).sum())
    if unsched_df is not None and not unsched_df.empty:
        unsched_minutes = float(pd.to_numeric(unsched_df.get("Duration", 0.0), errors="coerce").fillna(0.0).sum())
        if default_travel is not None:
            unsched_minutes += float(default_travel) * len(unsched_df)
    else:
        unsched_minutes = 0.0
    total_minutes = scheduled_minutes + unsched_minutes
    per_merch_capacity = float(daily_capacity) * float(workdays) * float(weeks)
    needed_sched = int(math.ceil(scheduled_minutes / per_merch_capacity)) if per_merch_capacity > 0 else 0
    needed_total = int(math.ceil(total_minutes / per_merch_capacity)) if per_merch_capacity > 0 else 0
    return needed_sched, needed_total, scheduled_minutes, total_minutes


def compute_peak_day_merch_need(daily_totals: pd.DataFrame, *, daily_capacity: float) -> int:
    if daily_totals is None or daily_totals.empty:
        return 0
    per_day_total = daily_totals.groupby("Day", as_index=False)["DailyMinutes"].sum()
    per_day_need = np.ceil(per_day_total["DailyMinutes"].to_numpy() / float(daily_capacity))
    return int(per_day_need.max() if len(per_day_need) else 0)


# =========================
# CLI
# =========================

def main():
    p = argparse.ArgumentParser(description=("Merchandiser scheduler with clustering, merge, fixed merch pool, performance knobs, and reporting."))
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--weeks", type=int, default=1)
    p.add_argument("--workdays", type=int, default=5)
    p.add_argument("--daily_capacity", type=float, default=480)
    p.add_argument("--speed", type=float, default=30, help="Avg speed km/h for distance-based travel time")
    p.add_argument("--default_travel", type=float, default=None, help="Fixed travel minutes per hop (overrides distance)")
    p.add_argument("--max_km_same_day", type=float, default=50)
    p.add_argument("--strict_same_merch", action="store_true")
    p.add_argument("--clusters", type=int, default=None)
    p.add_argument("--cluster_mode", choices=["kmeans", "radius"], default="kmeans")
    p.add_argument("--cluster_radius_km", type=float, default=None)
    p.add_argument("--distance_model", choices=["haversine", "geodesic", "euclidean"], default="haversine")
    p.add_argument("--merge_utilization_threshold", type=float, default=0.0)
    p.add_argument("--merge_cross_cluster", action="store_true")
    p.add_argument("--cache_precision", type=int, default=5)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--frequency_period", choices=["week", "month"], default="week")
    p.add_argument("--month_weeks", type=float, default=4.345)
    p.add_argument("--merch_names", type=str, default=None)
    p.add_argument("--merch_count", type=int, default=None)
    p.add_argument("--fixed_strict_capacity", action="store_true", help="(Fixed pool only) never exceed daily_capacity; unplaceable ⇒ Unscheduled")
    p.add_argument("--distance_scope", choices=["all_day", "adjacent"], default="all_day")
    p.add_argument("--fast", action="store_true")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    _set_cache_precision(args.cache_precision)
    np.random.seed(args.seed)

    fixed_names: Optional[List[str]] = None
    if args.merch_names:
        fixed_names = [s.strip() for s in args.merch_names.split(",") if s.strip()]
    elif args.merch_count and args.merch_count > 0:
        fixed_names = [f"Merch_{i+1}" for i in range(int(args.merch_count))]

    if fixed_names:
        logging.info(f"🔒 Fixed merch pool active: using exactly {len(fixed_names)} merchandisers.")
        args.merge_utilization_threshold = 0.0

    df = load_and_prepare_data(args.input, clusters_override=args.clusters, cluster_mode=args.cluster_mode, cluster_radius_km=args.cluster_radius_km, distance_model=args.distance_model, seed=args.seed)

    sched, unsched = schedule_with_constraints(
        df,
        weeks=args.weeks,
        workdays=args.workdays,
        daily_capacity=args.daily_capacity,
        avg_speed_kmph=args.speed,
        default_travel=args.default_travel,
        max_km_same_day=args.max_km_same_day,
        strict_same_merch=args.strict_same_merch,
        distance_model=args.distance_model,
        frequency_period=args.frequency_period,
        month_weeks=args.month_weeks,
        fixed_merch_names=fixed_names,
        fixed_strict_capacity=args.fixed_strict_capacity,
        distance_scope=args.distance_scope,
        fast=args.fast,
    )

    before = sched["Merchandiser"].nunique() if not sched.empty else 0
    if not fixed_names and args.merge_utilization_threshold > 0:
        sched = merge_underutilized(
            sched,
            daily_capacity=args.daily_capacity,
            max_km_same_day=args.max_km_same_day,
            distance_model=args.distance_model,
            threshold=args.merge_utilization_threshold,
            cross_cluster=args.merge_cross_cluster,
        )
        after = sched["Merchandiser"].nunique() if not sched.empty else 0
        if after < before:
            logging.info(f"🔄 Merged merchs: {before} → {after}")
    elif fixed_names:
        logging.info("↪️ Merge step skipped: fixed merch pool in use.")

    expected_total = compute_expected_total_visits(df, weeks=args.weeks, frequency_period=args.frequency_period, month_weeks=args.month_weeks)
    actual_total = len(sched)
    print(f"✅ Expected visits: {expected_total}, Scheduled visits: {actual_total}")
    used_merch = sched["Merchandiser"].nunique() if not sched.empty else 0
    print(f"👥 Total merchandisers used: {used_merch}")
    if unsched is not None and not unsched.empty:
        print(f"🧭 Unscheduled visits: {len(unsched)} (see 'Unscheduled' sheet)")

    sched["PerVisit_Minutes"] = pd.to_numeric(sched["Duration"], errors="coerce").fillna(0.0) + pd.to_numeric(sched.get("ActualTravel", 0.0), errors="coerce").fillna(0.0)

    daily_totals = sched.groupby(["Merchandiser", "Day"], as_index=False)["PerVisit_Minutes"].sum().rename(columns={"PerVisit_Minutes": "DailyMinutes"})
    avg_per_day = daily_totals.groupby("Merchandiser", as_index=False)["DailyMinutes"].mean().rename(columns={"DailyMinutes": "AvgMinutesPerActiveDay"})
    _week = pd.to_numeric(sched["Day"].astype(str).str.extract(r"Week(\d+)", expand=False), errors="coerce").fillna(1).astype(int)
    weekly_totals = sched.assign(Week=_week).groupby(["Merchandiser", "Week"], as_index=False)["PerVisit_Minutes"].sum().rename(columns={"PerVisit_Minutes": "WeeklyMinutes"})
    avg_per_week = weekly_totals.groupby("Merchandiser", as_index=False)["WeeklyMinutes"].mean().rename(columns={"WeeklyMinutes": "AvgMinutesPerWeek"})
    workload_summary = pd.merge(avg_per_day, avg_per_week, on="Merchandiser", how="outer").fillna(0.0)

    print("\n📊 Average workload per merchandiser:")
    for _, r in workload_summary.sort_values("Merchandiser").iterrows():
        print(f"- {r['Merchandiser']}: avg/day={int(round(r['AvgMinutesPerActiveDay']))} min, avg/week={int(round(r['AvgMinutesPerWeek']))} min")

    sched, day_km, merch_km = compute_day_km(sched, distance_model=args.distance_model)
    grand_total_km = float(merch_km["TotalKM"].sum()) if not merch_km.empty else 0.0
    print(f"\n🚗 Total travel distance: {grand_total_km:.1f} km")

    needed_sched, needed_total, sched_min, total_min = compute_effort_based_merchandisers(sched, unsched, daily_capacity=args.daily_capacity, workdays=args.workdays, weeks=args.weeks, default_travel=args.default_travel)
    peak_day_need = compute_peak_day_merch_need(daily_totals=daily_totals, daily_capacity=args.daily_capacity)

    print(f"\n🧮 Effort-based merchandisers (capacity={args.daily_capacity} min/day × {args.workdays} days × {args.weeks} weeks):")
    print(f"   • Global lower bound (scheduled only): {needed_sched}")
    if unsched is not None and not unsched.empty:
        print(f"   • Global lower bound (incl. unscheduled): {needed_total}")
    print(f"   • 📈 Peak-day concurrency need: {peak_day_need}")

    out_cols = ["Cluster", "City", "Store", "Merchandiser", "Day", "Seq", "Duration", "ActualTravel", "PerVisit_Minutes", "Lat", "Long", "HopKM", "OverCapacity"]
    existing = [c for c in out_cols if c in sched.columns]
    sched_sorted = sched.sort_values(["Merchandiser", "Day", "Seq"]) if not sched.empty else sched

    effort_rows = [
        {"Metric": "Daily Capacity (min)", "Value": args.daily_capacity},
        {"Metric": "Workdays", "Value": args.workdays},
        {"Metric": "Weeks", "Value": args.weeks},
        {"Metric": "Total Capacity per Merch (min)", "Value": args.daily_capacity * args.workdays * args.weeks},
        {"Metric": "Scheduled Minutes", "Value": sched_min},
        {"Metric": "Effort-based Merch (Global LB, Scheduled)", "Value": needed_sched},
        {"Metric": "Peak-day Effort Merch (Concurrency)", "Value": peak_day_need},
    ]
    if unsched is not None and not unsched.empty:
        effort_rows.extend([
            {"Metric": "Unscheduled Count", "Value": len(unsched)},
            {"Metric": "Total Minutes incl. Unscheduled", "Value": total_min},
            {"Metric": "Effort-based Merch (Global LB, Incl. Unscheduled)", "Value": needed_total},
        ])
    effort_df = pd.DataFrame(effort_rows)

    try:
        with pd.ExcelWriter(args.output, engine="xlsxwriter") as xw:
            (sched_sorted[existing] if not sched_sorted.empty else pd.DataFrame(columns=existing)).to_excel(xw, index=False, sheet_name="Schedule")
            (daily_totals.assign(UtilizationPct=(daily_totals["DailyMinutes"] / float(args.daily_capacity)) * 100.0).sort_values(["Merchandiser", "Day"])) .to_excel(xw, index=False, sheet_name="Summary")
            day_km.sort_values(["Merchandiser", "Day"]).to_excel(xw, index=False, sheet_name="DailyKM")
            merch_km.sort_values("Merchandiser").to_excel(xw, index=False, sheet_name="Weekly_KM")
            workload_summary.sort_values("Merchandiser").to_excel(xw, index=False, sheet_name="Workload_Averages")
            effort_df.to_excel(xw, index=False, sheet_name="Effort_Summary")
            if unsched is not None and not unsched.empty:
                unsched.to_excel(xw, index=False, sheet_name="Unscheduled")
    except Exception:
        with pd.ExcelWriter(args.output) as xw:
            (sched_sorted[existing] if not sched_sorted.empty else pd.DataFrame(columns=existing)).to_excel(xw, index=False, sheet_name="Schedule")
            (daily_totals.assign(UtilizationPct=(daily_totals["DailyMinutes"] / float(args.daily_capacity)) * 100.0).sort_values(["Merchandiser", "Day"])) .to_excel(xw, index=False, sheet_name="Summary")
            day_km.sort_values(["Merchandiser", "Day"]).to_excel(xw, index=False, sheet_name="DailyKM")
            merch_km.sort_values("Merchandiser").to_excel(xw, index=False, sheet_name="Weekly_KM")
            workload_summary.sort_values("Merchandiser").to_excel(xw, index=False, sheet_name="Workload_Averages")
            effort_df.to_excel(xw, index=False, sheet_name="Effort_Summary")
            if unsched is not None and not unsched.empty:
                unsched.to_excel(xw, index=False, sheet_name="Unscheduled")

    print(f"\n📄 Schedule saved to {args.output} (sheets: Schedule, Summary, DailyKM, Weekly_KM, Workload_Averages, Effort_Summary" + (", Unscheduled" if unsched is not None and not unsched.empty else "") + ")")


if __name__ == "__main__":
    main()
