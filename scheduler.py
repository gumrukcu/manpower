#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import logging
import math
from collections import defaultdict
from functools import lru_cache

import numpy as np
import pandas as pd
from geopy.distance import geodesic
from sklearn.cluster import KMeans, DBSCAN

# =========================
# Distance helpers + caching
# =========================

@lru_cache(maxsize=None)
def cached_distance(lat1, lon1, lat2, lon2) -> float:
    """Geodesic distance in km (cached)."""
    return geodesic((float(lat1), float(lon1)), (float(lat2), float(lon2))).km

def _haversine_km(lat1, lon1, lat2, lon2) -> float:
    R = 6371.0088
    φ1, λ1, φ2, λ2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dφ = φ2 - φ1
    dλ = λ2 - λ1
    a = math.sin(dφ/2)**2 + math.cos(φ1)*math.cos(φ2)*math.sin(dλ/2)**2
    return 2 * R * math.asin(math.sqrt(a))

# cache precision for haversine
CACHED_PREC = 5  # ~1.1 m at equator per 5 decimals

def _set_cache_precision(n: int):
    global CACHED_PREC
    CACHED_PREC = max(0, int(n))

@lru_cache(maxsize=None)
def _cached_haversine_key(lat1_r, lon1_r, lat2_r, lon2_r) -> float:
    return _haversine_km(lat1_r, lon1_r, lat2_r, lon2_r)

def cached_haversine(lat1, lon1, lat2, lon2) -> float:
    l1 = round(float(lat1), CACHED_PREC)
    o1 = round(float(lon1), CACHED_PREC)
    l2 = round(float(lat2), CACHED_PREC)
    o2 = round(float(lon2), CACHED_PREC)
    return _cached_haversine_key(l1, o1, l2, o2)

def dist_km(lat1, lon1, lat2, lon2, model: str) -> float:
    if model == "haversine":
        return cached_haversine(lat1, lon1, lat2, lon2)
    # "euclidean" fallback uses geodesic for realistic km
    return cached_distance(lat1, lon1, lat2, lon2)

# =========================
# Data prep + clustering
# =========================

def load_and_prepare_data(
    file_path: str,
    clusters_override: int | None,
    *,
    cluster_mode: str,
    cluster_radius_km: float | None,
    distance_model: str,  # kept for signature compatibility
) -> pd.DataFrame:
    df = pd.read_excel(file_path)

    # normalize numerics
    for col in ["Lat","Long","Estimated Duration In a store","Estimated Travel Time","Frequency","Cluster"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    required = ["City","Store Name","Estimated Duration In a store","Frequency"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.dropna(subset=["Estimated Duration In a store","Frequency"]).copy()

    # Respect existing clusters if present
    if "Cluster" in df.columns and df["Cluster"].notna().any():
        df["Cluster"] = df["Cluster"].fillna(0).astype(int)
        logging.info("ℹ️ Using existing Cluster column from input.")
        return df

    gps_df = df.dropna(subset=["Lat","Long"]).copy()
    missing_df = df[df["Lat"].isna() | df["Long"].isna()].copy()

    if gps_df.empty:
        df["Cluster"] = 0
        logging.info("⚠️ No GPS rows; assigning all to Cluster 0.")
        return df

    if cluster_mode == "radius":
        if not cluster_radius_km or cluster_radius_km <= 0:
            raise ValueError("--cluster_radius_km must be > 0 for cluster_mode=radius")
        # Fast radius clustering with DBSCAN (haversine)
        coords_deg = gps_df[["Lat","Long"]].to_numpy(dtype=float)
        coords_rad = np.radians(coords_deg)
        eps_rad = float(cluster_radius_km) / 6371.0088  # km -> radians
        db = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine", algorithm="ball_tree")
        labels = db.fit_predict(coords_rad)
        gps_df["Cluster"] = labels
        logging.info(f"📦 Radius clustering (DBSCAN) eps={cluster_radius_km} km → clusters={labels.max()+1}")
    else:
        # K-Means
        if clusters_override is not None and int(clusters_override) > 0:
            k = int(clusters_override)
        else:
            k = max(1, int(round(len(gps_df) ** 0.5)))
        unique_coords = gps_df.drop_duplicates(subset=["Lat","Long"])
        k = max(1, min(k, len(unique_coords)))
        logging.info(f"📦 K-Means clustering with k={k}")
        gps_df["Cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(gps_df[["Lat","Long"]])

    # Map clusters to rows without GPS by city majority
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
    default_travel: float | None,
    max_km_same_day: float,
    strict_same_merch: bool,
    distance_model: str,
    frequency_period: str,     # "week" or "month"
    month_weeks: float,        # weeks per month scalar (e.g., 4.345)
) -> pd.DataFrame:
    df = full_df.copy()

    # ---- visits to plan (weekly/monthly logic) ----
    base_freq = pd.to_numeric(df["Frequency"], errors="coerce").fillna(0.0)

    # Optional per-row override via column 'Frequency Period'
    if "Frequency Period" in df.columns:
        row_period = (
            df["Frequency Period"]
            .astype(str)
            .str.strip()
            .str.lower()
            .where(lambda s: s.isin(["week", "month"]))
        )
    else:
        row_period = pd.Series([None] * len(df))

    mw = float(month_weeks) if month_weeks and month_weeks > 0 else 4.345

    def _factor(idx: int) -> float:
        p = row_period.iat[idx] if row_period.iat[idx] in ("week", "month") else frequency_period
        return (weeks / mw) if p == "month" else float(weeks)

    factors = np.array([_factor(i) for i in range(len(df))], dtype=float)
    df["Total Visits"] = np.maximum(0, np.rint(base_freq.to_numpy() * factors)).astype(int)

    # record effective period per row for placement rules
    eff_period = []
    for i in range(len(df)):
        p = row_period.iat[i] if row_period.iat[i] in ("week","month") else frequency_period
        eff_period.append(p)
    df["_FreqPeriod"] = eff_period  # "week" or "month"

    # ---- calendar labels ----
    weekdays_all = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    workdays = max(1, min(int(workdays), 7))
    day_names = weekdays_all[:workdays]
    days = [f"Week{w+1}-{d}" for w in range(weeks) for d in day_names]
    L = len(days)  # total schedulable slots
    if L == 0:
        return pd.DataFrame(columns=["Store","City","Cluster","Merchandiser","Day","Seq","Duration","ActualTravel","Lat","Long"])

    # helpers
    day_to_index = {d: i for i, d in enumerate(days)}
    def week_of(day_label: str) -> int:
        return int(day_label.split("-")[0].replace("Week",""))

    def evenly_spaced_targets(n_visits: int, total_slots: int, store_key: str) -> list[int]:
        if n_visits <= 0:
            return []
        step = total_slots / float(n_visits)
        frac = (abs(hash(store_key)) % 997) / 997.0
        start = frac * min(step, total_slots)  # jitter
        idxs = []
        for i in range(n_visits):
            x = (i + 0.5) * step + start
            idx = int(min(total_slots - 1, max(0, math.floor(x))))
            idxs.append(idx)
        # strictly increasing (avoid duplicates when rounding)
        for i in range(1, len(idxs)):
            if idxs[i] <= idxs[i-1]:
                idxs[i] = min(total_slots - 1, idxs[i-1] + 1)
        return idxs

    def indices_by_closeness(target_idx: int, total: int) -> list[int]:
        order, used = [], set()
        # exact → ±1 → ±2 ...
        for d in range(total):
            for sgn in (0, -1, 1):
                if d == 0 and sgn != 0:
                    continue
                j = target_idx + (d if sgn == 1 else (-d if sgn == -1 else 0))
                if 0 <= j < total and j not in used:
                    order.append(j); used.add(j)
        return order

    # -------------------------------------------------------

    all_schedules = []
    for cluster_id, group in df.groupby("Cluster"):
        tasks_by_store = defaultdict(list)

        # --------- build tasks with per-visit candidate day lists ----------
        for _, row in group.iterrows():
            total_v = int(row["Total Visits"])
            if total_v <= 0:
                continue

            store_key = f"{row.get('Store Name','')}|{row.get('City','')}"
            is_month = (str(row.get("_FreqPeriod", "week")).lower() == "month")

            if is_month:
                targets = evenly_spaced_targets(total_v, L, store_key)
                for t in targets:
                    ordered_idx = indices_by_closeness(t, L)
                    ordered_days = [days[i] for i in ordered_idx]
                    tasks_by_store[row["Store Name"]].append({
                        "Store": row["Store Name"],
                        "City": row["City"],
                        "Duration": float(row["Estimated Duration In a store"]),
                        "Cluster": int(cluster_id),
                        "Lat": row.get("Lat", np.nan),
                        "Long": row.get("Long", np.nan),
                        "PreferredDaysOrdered": ordered_days,
                        "Monthly": True,
                        "TotalV": total_v,
                    })
            else:
                for _ in range(total_v):
                    tasks_by_store[row["Store Name"]].append({
                        "Store": row["Store Name"],
                        "City": row["City"],
                        "Duration": float(row["Estimated Duration In a store"]),
                        "Cluster": int(cluster_id),
                        "Lat": row.get("Lat", np.nan),
                        "Long": row.get("Long", np.nan),
                        "PreferredDaysOrdered": None,
                        "Monthly": False,
                        "TotalV": total_v,
                    })

        merch_schedules: dict[str, dict] = {}
        store_merch_map: dict[str, str] = {}
        merch_id = 1

        # per-store spacing trackers (for monthly)
        store_assigned_idx: dict[str, list[int]] = defaultdict(list)
        store_week_counts: dict[str, dict[int,int]] = defaultdict(lambda: defaultdict(int))

        for store, visits in tasks_by_store.items():
            base_rot = abs(hash(store)) % L

            preferred_merch = store_merch_map.get(store) if strict_same_merch else None
            merch_pool = [preferred_merch] if (preferred_merch and preferred_merch in merch_schedules) else \
                         list(merch_schedules.keys()) + [f"Merch_{cluster_id}_{merch_id}"]

            # precompute spacing caps for this store (monthly)
            total_v_for_store = len(visits)
            max_per_week_cap = math.ceil(total_v_for_store / max(1, weeks))  # e.g., 6 over 4 -> 2
            min_gap_slots = max(1, L // max(1, total_v_for_store))           # minimum day-slot distance

            for visit in visits:
                assigned = False
                for merch in merch_pool:
                    if merch not in merch_schedules:
                        merch_schedules[merch] = {d: {"total": 0.0, "visits": []} for d in days}
                        merch_id += 1
                    cal = merch_schedules[merch]

                    # candidate lists
                    candidate_day_lists = []
                    if visit.get("PreferredDaysOrdered"):
                        candidate_day_lists.append(visit["PreferredDaysOrdered"])
                    candidate_day_lists.append([days[(base_rot + k) % L] for k in range(L)])

                    # spacing-aware placement
                    def can_place_on(day_label: str, enforce_gap: bool, enforce_week_cap: bool) -> bool:
                        # daily dup store guard
                        if any(v["Store"] == store for v in cal[day_label]["visits"]):
                            return False

                        # monthly spacing checks
                        if visit["Monthly"]:
                            wk = week_of(day_label)
                            if enforce_week_cap and store_week_counts[store][wk] >= max_per_week_cap:
                                return False
                            if enforce_gap:
                                idx = day_to_index[day_label]
                                if store_assigned_idx[store]:
                                    # reject if too close to any existing visit for this store
                                    if min(abs(idx - j) for j in store_assigned_idx[store]) < min_gap_slots:
                                        return False

                        # distance guard across the day's visits
                        if max_km_same_day and cal[day_label]["visits"]:
                            for v in cal[day_label]["visits"]:
                                if all(pd.notna([v["Lat"], v["Long"], visit["Lat"], visit["Long"]])):
                                    if dist_km(v["Lat"], v["Long"], visit["Lat"], visit["Long"], model=distance_model) > max_km_same_day:
                                        return False

                        # travel minutes
                        if default_travel is not None:
                            travel_min = float(default_travel)
                        else:
                            if cal[day_label]["visits"] and all(pd.notna([cal[day_label]["visits"][-1]["Lat"], cal[day_label]["visits"][-1]["Long"], visit["Lat"], visit["Long"]])):
                                km = dist_km(cal[day_label]["visits"][-1]["Lat"], cal[day_label]["visits"][-1]["Long"],
                                             visit["Lat"], visit["Long"], model=distance_model)
                                travel_min = (km / avg_speed_kmph) * 60.0
                            else:
                                travel_min = 0.0

                        total_time = visit["Duration"] + travel_min
                        return cal[day_label]["total"] + total_time <= daily_capacity

                    def place(day_label: str):
                        # compute travel as above
                        if default_travel is not None:
                            travel_min = float(default_travel)
                        else:
                            if merch_schedules[merch][day_label]["visits"] and all(pd.notna([
                                    merch_schedules[merch][day_label]["visits"][-1]["Lat"],
                                    merch_schedules[merch][day_label]["visits"][-1]["Long"],
                                    visit["Lat"], visit["Long"]])):
                                km = dist_km(merch_schedules[merch][day_label]["visits"][-1]["Lat"],
                                             merch_schedules[merch][day_label]["visits"][-1]["Long"],
                                             visit["Lat"], visit["Long"], model=distance_model)
                                travel_min = (km / avg_speed_kmph) * 60.0
                            else:
                                travel_min = 0.0
                        v = {**visit, "ActualTravel": float(travel_min)}
                        merch_schedules[merch][day_label]["visits"].append(v)
                        merch_schedules[merch][day_label]["total"] += visit["Duration"] + float(travel_min)
                        # update spacing trackers
                        if visit["Monthly"]:
                            idx = day_to_index[day_label]
                            store_assigned_idx[store].append(idx)
                            store_week_counts[store][week_of(day_label)] += 1

                    placed = False
                    # Try strict → relax week cap → relax gap
                    for (enf_gap, enf_wcap) in [(True, True), (True, False), (False, False)]:
                        if placed:
                            break
                        for lst in candidate_day_lists:
                            for day_label in lst:
                                if can_place_on(day_label, enforce_gap=enf_gap, enforce_week_cap=enf_wcap):
                                    place(day_label)
                                    store_merch_map[store] = merch
                                    assigned = True
                                    placed = True
                                    break
                            if placed:
                                break

                    if assigned:
                        break

                # Absolute fallback: new merch on lightest day (rare)
                if not assigned:
                    new_merch = f"Merch_{cluster_id}_{merch_id}"; merch_id += 1
                    merch_schedules[new_merch] = {d: {"total": 0.0, "visits": []} for d in days}
                    lightest_day = min(days, key=lambda d: merch_schedules[new_merch][d]["total"])
                    first_travel = float(default_travel) if default_travel is not None else 0.0
                    v = {**visit, "ActualTravel": first_travel}
                    merch_schedules[new_merch][lightest_day]["visits"].append(v)
                    merch_schedules[new_merch][lightest_day]["total"] += visit["Duration"] + first_travel
                    store_merch_map[store] = new_merch
                    if visit["Monthly"]:
                        store_assigned_idx[store].append(day_to_index[lightest_day])
                        store_week_counts[store][week_of(lightest_day)] += 1

        # flatten
        for merch, plans in merch_schedules.items():
            for day, info in plans.items():
                for seq, visit in enumerate(info["visits"], start=1):
                    rec = visit.copy()
                    rec.update({"Merchandiser": merch, "Day": day, "Seq": seq})
                    all_schedules.append(rec)

    return pd.DataFrame(all_schedules)

# =========================
# Merge underutilized merchs (with optional cross-cluster)
# =========================

def merge_underutilized(schedule_df: pd.DataFrame,
                        *,
                        daily_capacity: float,
                        max_km_same_day: float,
                        distance_model: str,
                        threshold: float,
                        cross_cluster: bool = False) -> pd.DataFrame:
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
        if (df.loc[sel, "Store"] == row["Store"]).any():
            return False
        if max_km_same_day and max_km_same_day > 0:
            coords = df.loc[sel, ["Lat","Long"]].dropna()
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

            daily = (df.loc[c_mask]
                       .groupby(["Merchandiser","Day"], as_index=False)["_VisitMin"].sum()
                       .rename(columns={"_VisitMin":"DailyMin"}))
            if daily.empty:
                continue
            util = (daily.groupby("Merchandiser")["DailyMin"].mean() / daily_capacity).sort_values()

            order = util.index.tolist()
            order.sort(key=lambda m: (len(df.loc[c_mask & (df["Merchandiser"] == m)]),
                                      util.get(m, 1.0)))

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
                        # try lightest acceptable day first
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
                    improved = True  # re-evaluate utils

    if changed_any:
        df = df.sort_values(["Merchandiser","Day","Seq"]).copy()
        df["Seq"] = df.groupby(["Merchandiser","Day"]).cumcount() + 1

    return df.drop(columns=["_VisitMin"], errors="ignore")

# =========================
# Travel km per day & weekly totals
# =========================

def compute_day_km(sched_df: pd.DataFrame, *, distance_model: str):
    """
    Computes straight-line hop distance per visit (first hop=0),
    aggregates to DayKM per merch/day and TotalKM per merch (weekly).
    """
    df = sched_df.sort_values(["Merchandiser","Day","Seq"]).copy()
    hop_km = np.zeros(len(df), dtype=float)
    pos = {i: p for p, i in enumerate(df.index)}

    for (m, d), g in df.groupby(["Merchandiser","Day"], sort=False):
        g = g.sort_values("Seq")
        for i in range(1, len(g)):
            prev, cur = g.iloc[i-1], g.iloc[i]
            if all(pd.notna([prev["Lat"], prev["Long"], cur["Lat"], cur["Long"]])):
                km = dist_km(prev["Lat"], prev["Long"], cur["Lat"], cur["Long"], model=distance_model)
                hop_km[pos[cur.name]] += km

    df["HopKM"] = hop_km
    day_km = (df.groupby(["Merchandiser","Day"], as_index=False)["HopKM"]
              .sum().rename(columns={"HopKM":"DayKM"}))
    merch_km = (day_km.groupby("Merchandiser", as_index=False)["DayKM"]
                .sum().rename(columns={"DayKM":"TotalKM"}))
    return df, day_km, merch_km

# =========================
# Helpers
# =========================

def compute_expected_total_visits(src_df: pd.DataFrame, weeks: int, frequency_period: str, month_weeks: float) -> int:
    """Recompute expected total visits (mirrors schedule_with_constraints logic)."""
    df = src_df.copy()
    base = pd.to_numeric(df["Frequency"], errors="coerce").fillna(0.0)

    if "Frequency Period" in df.columns:
        row_period = (
            df["Frequency Period"]
            .astype(str).str.strip().str.lower()
            .where(lambda s: s.isin(["week","month"]))
        )
    else:
        row_period = pd.Series([None]*len(df))

    mw = float(month_weeks) if month_weeks and month_weeks > 0 else 4.345

    def _factor(idx: int) -> float:
        p = row_period.iat[idx] if row_period.iat[idx] in ("week","month") else frequency_period
        return (weeks / mw) if p == "month" else float(weeks)

    factors = np.array([_factor(i) for i in range(len(df))], dtype=float)
    total = np.maximum(0, np.rint(base.to_numpy() * factors)).astype(int).sum()
    return int(total)

# =========================
# CLI
# =========================

def main():
    p = argparse.ArgumentParser(description="Merchandiser scheduler with clustering, merge, and reporting.")
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
    p.add_argument("--cluster_mode", choices=["kmeans","radius"], default="kmeans")
    p.add_argument("--cluster_radius_km", type=float, default=None)
    p.add_argument("--distance_model", choices=["haversine","euclidean"], default="haversine")
    p.add_argument("--merge_utilization_threshold", type=float, default=0.0,
                   help="Merge merchs with avg daily utilization below this fraction of daily capacity (0 disables merging)")
    p.add_argument("--merge_cross_cluster", action="store_true",
                   help="Allow merging low-utilization merchs across clusters (constraints still enforced)")
    p.add_argument("--cache_precision", type=int, default=5,
                   help="Round lat/long decimals for haversine cache (higher=more precise, lower=faster)")
    p.add_argument("--verbose", action="store_true")

    # frequency interpretation controls
    p.add_argument("--frequency_period",
                   choices=["week", "month"],
                   default="week",
                   help="Interpret 'Frequency' as per-week or per-month.")
    p.add_argument("--month_weeks",
                   type=float,
                   default=4.345,
                   help="Number of weeks per month when frequency_period=month (e.g., 4.0 or 4.345).")

    args = p.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING,
                        format="%(message)s")

    _set_cache_precision(args.cache_precision)

    df = load_and_prepare_data(
        args.input,
        clusters_override=args.clusters,
        cluster_mode=args.cluster_mode,
        cluster_radius_km=args.cluster_radius_km,
        distance_model=args.distance_model,
    )

    sched = schedule_with_constraints(
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
    )

    # Optional merge
    before = sched["Merchandiser"].nunique()
    sched = merge_underutilized(
        sched,
        daily_capacity=args.daily_capacity,
        max_km_same_day=args.max_km_same_day,
        distance_model=args.distance_model,
        threshold=args.merge_utilization_threshold,
        cross_cluster=args.merge_cross_cluster,
    )
    after = sched["Merchandiser"].nunique()
    if args.merge_utilization_threshold > 0 and after < before:
        logging.info(f"🔄 Merged merchs: {before} → {after}")

    # Expected vs actual (recomputed safely)
    expected_total = compute_expected_total_visits(
        df, weeks=args.weeks, frequency_period=args.frequency_period, month_weeks=args.month_weeks
    )
    actual_total = len(sched)
    print(f"✅ Expected visits: {expected_total}, Scheduled visits: {actual_total}")
    print(f"👥 Total merchandisers used: {after}")

    # ---- effort-based merchandisers (FTE over the scheduled period) ----
    total_minutes_for_period = float(sched["Duration"].astype(float).sum()) + \
        float(pd.to_numeric(sched.get("ActualTravel", 0.0), errors="coerce").fillna(0.0).sum())

    denom = float(args.daily_capacity) * max(1, args.workdays) * max(1, args.weeks)
    effort_merch = (total_minutes_for_period / denom) if denom > 0 else 0.0

    print(f"🧮 Effort-based merchandisers "
          f"(total_minutes / (daily_capacity×workdays×weeks)): "
          f"{effort_merch:.2f} (ceil={math.ceil(effort_merch)})")

    # ---- reporting: minutes ----
    sched["PerVisit_Minutes"] = sched["Duration"].astype(float) + pd.to_numeric(
        sched.get("ActualTravel", 0.0), errors="coerce"
    ).fillna(0.0)

    daily_totals = (sched.groupby(["Merchandiser","Day"], as_index=False)["PerVisit_Minutes"]
                    .sum().rename(columns={"PerVisit_Minutes":"DailyMinutes"}))
    avg_per_day = (daily_totals.groupby("Merchandiser", as_index=False)["DailyMinutes"]
                   .mean().rename(columns={"DailyMinutes":"AvgMinutesPerActiveDay"}))
    _week = pd.to_numeric(
        sched["Day"].astype(str).str.extract(r"Week(\d+)", expand=False),
        errors="coerce"
    ).fillna(1).astype(int)
    weekly_totals = (sched.assign(Week=_week)
                     .groupby(["Merchandiser","Week"], as_index=False)["PerVisit_Minutes"].sum()
                     .rename(columns={"PerVisit_Minutes":"WeeklyMinutes"}))
    avg_per_week = (weekly_totals.groupby("Merchandiser", as_index=False)["WeeklyMinutes"]
                    .mean().rename(columns={"WeeklyMinutes":"AvgMinutesPerWeek"}))
    summary = pd.merge(avg_per_day, avg_per_week, on="Merchandiser", how="outer").fillna(0.0)

    print("\n📊 Average workload per merchandiser:")
    for _, r in summary.sort_values("Merchandiser").iterrows():
        print(f"- {r['Merchandiser']}: avg/day={int(round(r['AvgMinutesPerActiveDay']))} min, "
              f"avg/week={int(round(r['AvgMinutesPerWeek']))} min")

    # ---- kilometers (compute once at the end; console prints grand total only) ----
    sched, day_km, merch_km = compute_day_km(sched, distance_model=args.distance_model)
    grand_total_km = float(merch_km["TotalKM"].sum())
    print(f"\n🚗 Total weekly travel distance: {grand_total_km:.1f} km")

    # ---- Excel output ----
    out_cols = ["Cluster","City","Store","Merchandiser","Day","Seq",
                "Duration","ActualTravel","PerVisit_Minutes","Lat","Long","HopKM"]
    existing = [c for c in out_cols if c in sched.columns]
    sched_sorted = sched.sort_values(["Merchandiser","Day","Seq"])

    try:
        with pd.ExcelWriter(args.output, engine="xlsxwriter") as xw:
            # 1) Full plan
            sched_sorted[existing].to_excel(xw, index=False, sheet_name="Schedule")
            # 2) Summary (minutes/utilization)
            (daily_totals.assign(UtilizationPct=(daily_totals["DailyMinutes"]/float(args.daily_capacity))*100.0)
             .sort_values(["Merchandiser","Day"])
             .to_excel(xw, index=False, sheet_name="Summary"))
            # 3) DailyKM (per merch/day)
            day_km.sort_values(["Merchandiser","Day"]).to_excel(xw, index=False, sheet_name="DailyKM")
            # 4) Weekly_KM (per merch weekly totals)
            merch_km.sort_values("Merchandiser").to_excel(xw, index=False, sheet_name="Weekly_KM")
    except Exception:
        with pd.ExcelWriter(args.output) as xw:
            sched_sorted[existing].to_excel(xw, index=False, sheet_name="Schedule")
            (daily_totals.assign(UtilizationPct=(daily_totals["DailyMinutes"]/float(args.daily_capacity))*100.0)
             .sort_values(["Merchandiser","Day"])
             .to_excel(xw, index=False, sheet_name="Summary"))
            day_km.sort_values(["Merchandiser","Day"]).to_excel(xw, index=False, sheet_name="DailyKM")
            merch_km.sort_values("Merchandiser").to_excel(xw, index=False, sheet_name="Weekly_KM")

    print(f"\n📄 Schedule saved to {args.output} (sheets: Schedule, Summary, DailyKM, Weekly_KM)")

if __name__ == "__main__":
    main()
