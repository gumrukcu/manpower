# Merchandiser Scheduling App – User Guide

## Overview
The Merchandiser Scheduling App helps you automatically plan store visits for your merchandisers.  
It takes your store list, visit frequencies, and capacity constraints, and produces an optimized schedule with clusters, assigned merchandisers, and routes.

You upload your store data, choose your scheduling preferences, and the app outputs a ready-to-use Excel file with your optimized plan.

---

## 1. Input Data
You’ll need an Excel file with at least the following columns:

| Column Name | Description |
|-------------|-------------|
| **City** | Name of the city where the store is located |
| **Store Name** | The store’s name |
| **Lat** | Store latitude (decimal degrees) |
| **Long** | Store longitude (decimal degrees) |
| **Estimated Duration In a store** | Time spent in store (minutes) |
| **Estimated Travel Time** *(optional)* | Fixed travel time if not calculating from distances |
| **Frequency per week** | Number of visits required per week |
| **Cluster** *(optional)* | Predefined cluster IDs if you don’t want the app to calculate them |

💡 **Tip:** If `Lat` and `Long` are missing, the system will group them by city.

---

## 2. Main Settings
When you start the scheduler, you can adjust the following:

| Setting | Description |
|---------|-------------|
| **Weeks** | Number of weeks to plan for |
| **Workdays** | Number of working days per week (1–7) |
| **Daily Capacity** | Total available working minutes per merchandiser per day |
| **Average Speed (km/h)** | Used for calculating travel time from distances |
| **Default Travel Minutes** | Fixed travel time for each hop (overrides calculated travel time) |
| **Max KM Same Day** | Maximum allowed distance between any two stops in the same day |
| **Strict Same Merchandiser** | If on, the same store will always be visited by the same merchandiser |
| **Cluster Mode** | How to group stores:  |
| | - **KMeans** = Group by similar location automatically  |
| | - **Radius** = Group by maximum distance between stores |
| **KMeans k** | If using KMeans, number of clusters. Leave blank for automatic calculation |
| **Cluster Radius KM** | If using Radius mode, maximum distance in km between stores in a cluster |
| **Distance Model** | How distances are measured:  |
| | - **Haversine** = More accurate on Earth’s curve  |
| | - **Euclidean** = Straight-line “as the crow flies” |
| **Frequency is Total** | Treat the frequency column as total visits instead of weekly |
| **Merge Utilization Threshold** | Merge merchandisers with less daily work than this percentage |
| **Merge Across Clusters** | Allow merging of merchandisers even if they work in different clusters |
| **Verbose Mode** | Show detailed scheduling logs |

---

## 3. How Scheduling Works
1. **Load Data** → Reads your Excel file and prepares store details.
2. **Cluster Stores** → Groups stores into geographical clusters.
3. **Assign Merchandisers** → Allocates visits to merchandisers based on capacity and constraints.
4. **Optimize** → Merges underutilized merchandisers if desired.
5. **Output** → Creates an Excel file with:
   - **Schedule**: All store visits, ordered by merchandiser, day, and sequence
   - **Summary**: Workload per day and utilization
   - **DailyKM**: Travel distance per day
   - **Weekly_KM**: Total travel distance per merchandiser per week

---

## 4. Output Excel
The output Excel file contains 4 sheets:

### Schedule
| Cluster | City | Store | Merchandiser | Day | Seq | Duration | ActualTravel | PerVisit_Minutes | Lat | Long | HopKM |
|---------|------|-------|--------------|-----|-----|----------|--------------|------------------|-----|------|-------|

- **Seq** = Visit order in the day
- **HopKM** = Distance from previous stop

### Summary
- Minutes worked per merchandiser per day
- Utilization % of daily capacity

### DailyKM
- Kilometers traveled per merchandiser per day

### Weekly_KM
- Total kilometers traveled per merchandiser per week

---

## 5. Example Run
```bash
python3 scheduler.py     --input "Coverage_Max.xlsx"     --output "Schedule.xlsx"     --weeks 1     --workdays 6     --daily_capacity 480     --cluster_mode radius     --cluster_radius_km 100     --distance_model haversine     --max_km_same_day 100     --merge_utilization_threshold 0.3     --merge_cross_cluster     --verbose
```

---

## 6. Tips for Best Results
- Always provide accurate **Lat/Long** for better distance calculations.
- Set **daily capacity** to match real working hours (e.g., 480 minutes for 8 hours).
- If your regions are far apart, use **cluster_mode radius** with a sensible `cluster_radius_km`.
- Use **merge_utilization_threshold** to avoid part-time merchandisers unless needed.
- Use **strict_same_merch** if you want consistent store-merchandiser relationships.
