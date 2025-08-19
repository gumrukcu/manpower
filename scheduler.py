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
        coords_deg = gps_df[["Lat","Long"]].to_numpy(dtype=float)
        coords_rad = np.radians(coords_deg)
        eps_rad = float(cluster_radius_km) / 6371.0088  # km -> radians
        db = DBSCAN(eps=eps_rad, min_samples=1, metric="haversine", algorithm="ball_tree")
        labels = db.fit_predict(coords_rad)
        gps_df["Cluster"] = labels
        logging.info(f"📦 Radius clustering (DBSCAN) eps={cluster_radius_km} km → clusters={labels.max()+1}")
    else:
        if clusters_override is not None and int(clusters_override) > 0:
            k = int(clusters_override)
        else:
            k = max(1, int(round(len(gps_df) ** 0.5)))
        unique_coords = gps_df.drop_duplicates(subset=["Lat","Long"])
        k = max(1, min(k, len(unique_coords)))
        logging.info(f"📦 K-Means clustering with k={k}")
        gps_df["Cluster"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(gps_df[["Lat","Long"]])

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
    frequency_period: str,
    month_weeks: float,
    # NEW: fixed merch pool
    fixed_merch_names: list[str] | None = None,
    fixed_strict_capacity: bool = False,
) -> pd.DataFrame:
    df = full_df.copy()

    # ---- visits to plan (weekly/monthly logic) ----
    base_freq = pd.to_numeric(df["Frequency"], errors="coerce").fillna(0.0)

    if "Frequency Period" in df.columns:
        row_period = (df["Frequency Period"].astype(str).str.strip().str.lower()
                      .where(lambda s: s.isin(["week", "month"])))
    else:
        row_period = pd.Series([None] * len(df))

    mw = float(month_weeks) if month_weeks and month_weeks > 0 else 4.345
    def _factor(i: int) -> float:
        p = row_period.iat[i] if row_period.iat[i] in ("week","month") else frequency_period
        return (weeks / mw) if p == "month" else float(weeks)
    factors = np.array([_factor(i) for i in range(len(df))], dtype=float)
    df["Total Visits"] = np.maximum(0, np.rint(base_freq.to_numpy() * factors)).astype(int)

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
    L = len(days)
    if L == 0:
        return pd.DataFrame(columns=["Store","City","Cluster","Merchandiser","Day","Seq","Duration","ActualTravel","Lat","Long"])

    # ==== EVEN SPACING (monthly) ====
    day_to_index = {d: i for i, d in enumerate(days)}
    def week_of(day_label: str) -> int:
        return int(day_label.split("-")[0].replace("Week",""))

    def evenly_spaced_targets(n_visits: int, total_slots: int, store_key: str) -> list[int]:
        if n_visits <= 0: return []
        step = total_slots / float(n_visits)
        frac = (abs(hash(store_key)) % 997) / 997.0
        start = frac * min(step, total_slots)
        idxs = []
        for i in range(n_visits):
            x = (i + 0.5) * step + start
            idx = int(min(total_slots - 1, max(0, math.floor(x))))
            idxs.append(idx)
        for i in range(1, len(idxs)):
            if idxs[i] <= idxs[i-1]:
                idxs[i] = min(total_slots - 1, idxs[i-1] + 1)
        return idxs

    def indices_by_closeness(target_idx: int, total: int) -> list[int]:
        order, used = [], set()
        for d in range(total):
            for sgn in (0, -1, 1):
                if d == 0 and sgn != 0: continue
                j = target_idx + (d if sgn == 1 else (-d if sgn == -1 else 0))
                if 0 <= j < total and j not in used:
                    order.append(j); used.add(j)
        return order

    # ==== merch pool (global) ====
    fixed_mode = bool(fixed_merch_names)
    if fixed_mode:
        merch_schedules: dict[str, dict] = {
            name: {d: {"total": 0.0, "visits": []} for d in days}
            for name in fixed_merch_names
        }
    else:
        merch_schedules = {}  # created on demand
    store_merch_map: dict[str,str] = {}

    # global trackers for monthly spacing
    store_assigned_idx: dict[str, list[int]] = defaultdict(list)
    store_week_counts: dict[str, dict[int,int]] = defaultdict(lambda: defaultdict(int))

    # ---- build per-cluster tasks, but place on the GLOBAL merch pool ----
    all_schedules = []
    for cluster_id, group in df.groupby("Cluster"):
        tasks_by_store = defaultdict(list)

        for _, row in group.iterrows():
            total_v = int(row["Total Visits"])
            if total_v <= 0: continue

            store_key = f"{row.get('Store Name','')}|{row.get('City','')}"
            is_month = (str(row.get("_FreqPeriod", "week")).lower() == "month")
            if is_month:
                targets = evenly_spaced_targets(total_v, L, store_key)
                for t in targets:
                    ordered_idx = indices_by_closeness(t, L)
                    ordered_days = [days[i] for i in ordered_idx]
                    tasks_by_store[row["Store Name"]].append({
                        "Store": row["Store Name"], "City": row["City"], "Cluster": int(cluster_id),
                        "Duration": float(row["Estimated Duration In a store"]),
                        "Lat": row.get("Lat", np.nan), "Long": row.get("Long", np.nan),
                        "PreferredDaysOrdered": ordered_days, "Monthly": True, "TotalV": total_v,
                    })
            else:
                for _ in range(total_v):
                    tasks_by_store[row["Store Name"]].append({
                        "Store": row["Store Name"], "City": row["City"], "Cluster": int(cluster_id),
                        "Duration": float(row["Estimated Duration In a store"]),
                        "Lat": row.get("Lat", np.nan), "Long": row.get("Long", np.nan),
                        "PreferredDaysOrdered": None, "Monthly": False, "TotalV": total_v,
                    })

        merch_id_counter = 1  # only used if not fixed_mode

        for store, visits in tasks_by_store.items():
            base_rot = abs(hash(store)) % L
            store_key = f"{store}|{group['City'].iloc[0] if 'City' in group.columns else ''}"

            preferred_merch = store_merch_map.get(store) if strict_same_merch else None
            if fixed_mode:
                merch_pool = ([preferred_merch] if preferred_merch in (fixed_merch_names or []) else fixed_merch_names)
            else:
                merch_pool = [preferred_merch] if (preferred_merch and preferred_merch in merch_schedules) else \
                             list(merch_schedules.keys()) + [f"Merch_{merch_id_counter}"]

            # spacing caps for this store (monthly)
            total_v_for_store = len(visits)
            max_per_week_cap = math.ceil(total_v_for_store / max(1, weeks))
            min_gap_slots = max(1, L // max(1, total_v_for_store))

            for visit in visits:
                assigned = False

                for merch in merch_pool:
                    # materialize merch if on-demand mode
                    if merch not in merch_schedules:
                        # only allowed in non-fixed mode
                        merch_schedules[merch] = {d: {"total": 0.0, "visits": []} for d in days}
                        merch_id_counter += 1
                    cal = merch_schedules[merch]

                    # candidate day lists (preference then global rotation)
                    candidate_day_lists = []
                    if visit.get("PreferredDaysOrdered"):
                        candidate_day_lists.append(visit["PreferredDaysOrdered"])
                    candidate_day_lists.append([days[(base_rot + k) % L] for k in range(L)])

                    # placement helpers with toggles
                    def can_place(day_label: str, enforce_gap=True, enforce_wcap=True, enforce_dist=True, enforce_cap=True) -> tuple[bool,float]:
                        # same store / day guard
                        if any(v["Store"] == store for v in cal[day_label]["visits"]):
                            return (False, 0.0)

                        # monthly spacing
                        if visit["Monthly"]:
                            if enforce_wcap:
                                wk = week_of(day_label)
                                if store_week_counts[store_key][wk] >= max_per_week_cap:
                                    return (False, 0.0)
                            if enforce_gap and store_assigned_idx[store_key]:
                                idx = day_to_index[day_label]
                                if min(abs(idx - j) for j in store_assigned_idx[store_key]) < min_gap_slots:
                                    return (False, 0.0)

                        # distance guard
                        if enforce_dist and max_km_same_day and cal[day_label]["visits"]:
                            for v in cal[day_label]["visits"]:
                                if all(pd.notna([v["Lat"], v["Long"], visit["Lat"], visit["Long"]])):
                                    if dist_km(v["Lat"], v["Long"], visit["Lat"], visit["Long"], model=distance_model) > max_km_same_day:
                                        return (False, 0.0)

                        # travel time
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
                        if enforce_cap and (cal[day_label]["total"] + total_time > daily_capacity):
                            return (False, travel_min)
                        return (True, travel_min)

                    def place(day_label: str, travel_min: float, overcap_flag: int):
                        v = {**visit, "ActualTravel": float(travel_min), "OverCapacity": overcap_flag}
                        cal = merch_schedules[merch]
                        cal[day_label]["visits"].append(v)
                        cal[day_label]["total"] += visit["Duration"] + float(travel_min)
                        store_merch_map[store] = merch
                        if visit["Monthly"]:
                            idx = day_to_index[day_label]
                            store_assigned_idx[store_key].append(idx)
                            store_week_counts[store_key][week_of(day_label)] += 1

                    placed = False

                    # 1) strict everything
                    for lst in candidate_day_lists:
                        for dlab in lst:
                            ok, tmin = can_place(dlab, True, True, True, True)
                            if ok:
                                place(dlab, tmin, 0); assigned = True; placed = True; break
                        if placed: break
                    if placed: break

                    # 2) relax distance, keep capacity
                    for lst in candidate_day_lists:
                        for dlab in lst:
                            ok, tmin = can_place(dlab, True, True, False, True)
                            if ok:
                                place(dlab, tmin, 0); assigned = True; placed = True; break
                        if placed: break
                    if placed: break

                    # 3) relax weekly cap (keep gap), keep distance+capacity
                    for lst in candidate_day_lists:
                        for dlab in lst:
                            ok, tmin = can_place(dlab, True, False, True, True)
                            if ok:
                                place(dlab, tmin, 0); assigned = True; placed = True; break
                        if placed: break
                    if placed: break

                # 4) FINAL FALLBACKS (fixed_mode only)
                if not assigned and fixed_mode:
                    # try with capacity strictness depending on flag; distance relaxed
                    best = None  # (added_total, merch_name, day_label, travel_min)
                    for merch in fixed_merch_names or []:
                        cal = merch_schedules[merch]
                        for dlab in days:
                            # still avoid same store twice a day
                            if any(v["Store"] == store for v in cal[dlab]["visits"]):
                                continue
                            ok, tmin = can_place(dlab, False, False, False, not fixed_strict_capacity)
                            if not ok and fixed_strict_capacity:
                                continue
                            added = cal[dlab]["total"] + visit["Duration"] + (tmin if not math.isnan(tmin) else 0.0)
                            if (best is None) or (added < best[0]):
                                best = (added, merch, dlab, tmin)

                    if best is not None:
                        _, merch, dlab, tmin = best
                        # mark over-capacity only if we actually exceed it
                        overcap = 1 if (fixed_strict_capacity is False and
                                        merch_schedules[merch][dlab]["total"] + visit["Duration"] + tmin > daily_capacity) else 0
                        cal = merch_schedules[merch]
                        v = {**visit, "ActualTravel": float(tmin), "OverCapacity": overcap}
                        cal[dlab]["visits"].append(v)
                        cal[dlab]["total"] += visit["Duration"] + float(tmin)
                        if visit["Monthly"]:
                            idx = day_to_index[dlab]
                            store_assigned_idx[store_key].append(idx)
                            store_week_counts[store_key][week_of(dlab)] += 1
                        assigned = True

                if not assigned and not fixed_mode:
                    # original behavior: create a new merch and place on lightest day
                    new_merch = f"Merch_auto_{len(merch_schedules)+1}"
                    merch_schedules[new_merch] = {d: {"total": 0.0, "visits": []} for d in days}
                    lightest_day = min(days, key=lambda d: merch_schedules[new_merch][d]["total"])
                    first_travel = float(default_travel) if default_travel is not None else 0.0
                    v = {**visit, "ActualTravel": first_travel, "OverCapacity": 0}
                    merch_schedules[new_merch][lightest_day]["visits"].append(v)
                    merch_schedules[new_merch][lightest_day]["total"] += visit["Duration"] + first_travel
                    store_merch_map[store] = new_merch

    # ---- flatten to dataframe ----
    all_rows = []
    for merch, plans in merch_schedules.items():
        for day, info in plans.items():
            for seq, visit in enumerate(info["visits"], start=1):
                rec = visit.copy()
                rec.update({"Merchandiser": merch, "Day": day, "Seq": seq})
                all_rows.append(rec)

    out = pd.DataFrame(all_rows)
    return out

# =========================
# Merge underutilized merchs (kept for non-fixed mode)
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
        df = df.sort_values(["Merchandiser","Day","Seq"]).copy()
        df["Seq"] = df.groupby(["Merchandiser","Day"]).cumcount() + 1

    return df.drop(columns=["_VisitMin"], errors="ignore")

# =========================
# Travel km per day & weekly totals
# =========================

def compute_day_km(sched_df: pd.DataFrame, *, distance_model: str):
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
    df = src_df.copy()
    base = pd.to_numeric(df["Frequency"], errors="coerce").fillna(0.0)
    if "Frequency Period" in df.columns:
        row_period = (df["Frequency Period"].astype(str).str.strip().str.lower()
                      .where(lambda s: s.isin(["week","month"])))
    else:
        row_period = pd.Series([None]*len(df))
    mw = float(month_weeks) if month_weeks and month_weeks > 0 else 4.345
    def _factor(i: int) -> float:
        p = row_period.iat[i] if row_period.iat[i] in ("week","month") else frequency_period
        return (weeks / mw) if p == "month" else float(weeks)
    factors = np.array([_factor(i) for i in range(len(df))], dtype=float)
    total = np.maximum(0, np.rint(base.to_numpy() * factors)).astype(int).sum()
    return int(total)

# =========================
# CLI
# =========================

def main():
    p = argparse.ArgumentParser(description="Merchandiser scheduler with clustering, merge, fixed merch pool, and reporting.")
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
    p.add_argument("--merge_utilization_threshold", type=float, default=0.0)
    p.add_argument("--merge_cross_cluster", action="store_true")
    p.add_argument("--cache_precision", type=int, default=5)
    p.add_argument("--verbose", action="store_true")

    # frequency
    p.add_argument("--frequency_period", choices=["week","month"], default="week")
    p.add_argument("--month_weeks", type=float, default=4.345)

    # NEW: fixed merch pool
    p.add_argument("--merch_names", type=str, default=None,
                   help="Comma-separated fixed merchandiser names to use (exact).")
    p.add_argument("--merch_count", type=int, default=None,
                   help="Exact number of merchandisers to use (Merch_1..Merch_N) if --merch_names is not given.")
    p.add_argument("--fixed_strict_capacity", action="store_true",
                   help="With fixed merch pool, never exceed daily_capacity; unplaceable visits are reported as UNSCHEDULED.")

    args = p.parse_args()
    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(message)s")
    _set_cache_precision(args.cache_precision)

    # Resolve fixed merch pool
    fixed_names: list[str] | None = None
    if args.merch_names:
        fixed_names = [s.strip() for s in args.merch_names.split(",") if s.strip()]
    elif args.merch_count and args.merch_count > 0:
        fixed_names = [f"Merch_{i+1}" for i in range(int(args.merch_count))]

    if fixed_names:
        logging.info(f"🔒 Fixed merch pool active: using exactly {len(fixed_names)} merchandisers.")
        # When fixed pool is active, ignore merge phase later (we must keep the count).
        args.merge_utilization_threshold = 0.0

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
        fixed_merch_names=fixed_names,
        fixed_strict_capacity=args.fixed_strict_capacity,
    )

    # Optional merge (disabled in fixed mode)
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
        after = sched["Merchandiser"].nunique()
        if after < before:
            logging.info(f"🔄 Merged merchs: {before} → {after}")
    elif fixed_names:
        logging.info("↪️ Merge step skipped: fixed merch pool in use.")

    # Expected vs actual
    expected_total = compute_expected_total_visits(
        df, weeks=args.weeks, frequency_period=args.frequency_period, month_weeks=args.month_weeks
    )
    actual_total = len(sched)
    print(f"✅ Expected visits: {expected_total}, Scheduled visits: {actual_total}")
    print(f"👥 Total merchandisers used: {sched['Merchandiser'].nunique() if not sched.empty else 0}")

    # Flag counts
    if "OverCapacity" in sched.columns:
        over = int(sched["OverCapacity"].fillna(0).sum())
        if over > 0:
            print(f"⚠️ Placements exceeding daily_capacity (OverCapacity=1): {over}")

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

    # ---- kilometers ----
    sched, day_km, merch_km = compute_day_km(sched, distance_model=args.distance_model)
    grand_total_km = float(merch_km["TotalKM"].sum())
    print(f"\n🚗 Total weekly travel distance: {grand_total_km:.1f} km")

    # ---- Excel output ----
    out_cols = ["Cluster","City","Store","Merchandiser","Day","Seq",
                "Duration","ActualTravel","PerVisit_Minutes","Lat","Long","HopKM","OverCapacity"]
    existing = [c for c in out_cols if c in sched.columns]
    sched_sorted = sched.sort_values(["Merchandiser","Day","Seq"])

    try:
        with pd.ExcelWriter(args.output, engine="xlsxwriter") as xw:
            sched_sorted[existing].to_excel(xw, index=False, sheet_name="Schedule")
            (daily_totals.assign(UtilizationPct=(daily_totals["DailyMinutes"]/float(args.daily_capacity))*100.0)
             .sort_values(["Merchandiser","Day"])
             .to_excel(xw, index=False, sheet_name="Summary"))
            day_km.sort_values(["Merchandiser","Day"]).to_excel(xw, index=False, sheet_name="DailyKM")
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
