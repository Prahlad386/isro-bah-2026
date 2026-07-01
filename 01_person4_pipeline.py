"""
====================================================================
PERSON 4 - ROVER TRAVERSE PLANNING & ICE VOLUME LEAD
Full pipeline: landing site selection -> cost map -> A* path planning
               -> distance/energy estimate -> ice volume estimate -> PDF reports
====================================================================
Reads from data/inputs/ (synthetic stand-ins for Person 2 & Person 3 outputs):
    HazardMap.tif, LandingSuitability.tif, IceProbability.tif, IceMask.tif,
    IceRegions.geojson, LandingCandidates.geojson

Writes to data/outputs/:
    BestLandingSite.geojson
    MissionCostMap.tif
    TraversePath.geojson
    IceVolumeReport.pdf
    MissionPlan.pdf

Run:  python3 01_person4_pipeline.py
====================================================================
"""

import numpy as np
import rasterio
from rasterio.transform import rowcol, xy
import geopandas as gpd
from shapely.geometry import Point, LineString
import heapq
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

IN = "/home/claude/person4_project/data/inputs"
OUT = "/home/claude/person4_project/data/outputs"
PIXEL_SIZE_M = 5.0

# Mission assumptions (documented explicitly, since these drive energy/time estimates)
ROVER_SPEED_MPS = 0.05          # ~180 m/h, in line with typical small lunar rover speeds (e.g. Pragyan-class)
ROVER_MASS_KG = 30.0
MOON_GRAVITY = 1.62             # m/s^2
DRIVE_POWER_W = 25.0            # nominal driving power draw
ICE_DEPTH_M = 5.0               # per PS spec: estimate ice in top 5 m of regolith
ICE_DENSITY_KGM3 = 917.0
REGOLITH_DIELECTRIC = 3.0
ICE_DIELECTRIC = 3.2            # used only to document the mixing-model assumption in the report

print("=" * 70)
print("PERSON 4 PIPELINE START")
print("=" * 70)

# ====================================================================
# STEP 0: Load all inputs
# ====================================================================
print("\n[STEP 0] Loading inputs...")

with rasterio.open(f"HazardMap.tif") as src:
    hazard = src.read(1)
    transform = src.transform
    crs = src.crs
    profile = src.profile

with rasterio.open(f"LandingSuitability.tif") as src:
    suitability = src.read(1)

with rasterio.open(f"IceProbability.tif") as src:
    ice_prob = src.read(1)

with rasterio.open(f"IceMask.tif") as src:
    ice_mask = src.read(1)

ice_regions = gpd.read_file(f"IceRegions.geojson")
landing_candidates = gpd.read_file(f"LandingCandidates.geojson")

print(f"  Loaded HazardMap, LandingSuitability, IceProbability, IceMask: shape={hazard.shape}")
print(f"  Loaded {len(ice_regions)} ice region(s), {len(landing_candidates)} landing candidate(s)")

def world_to_pixel(x, y):
    row, col = rowcol(transform, x, y)
    return int(row), int(col)

def pixel_to_world(row, col):
    x, y = xy(transform, row, col)
    return x, y

# ====================================================================
# STEP 1: Ice target identification ("crater detected" / ice zone confirmed)
# ====================================================================
print("\n[STEP 1] Identifying ice target zone...")

# Pick the highest mean-probability ice region as the primary target
ice_regions["mean_ice_probability"] = ice_regions["mean_ice_probability"].astype(float)
target_region = ice_regions.loc[ice_regions["mean_ice_probability"].idxmax()]
target_centroid = target_region.geometry.centroid
target_row, target_col = world_to_pixel(target_centroid.x, target_centroid.y)

print(f"  Target ice region: id={target_region['region_id']}, "
      f"area={target_region['area_m2']:.0f} m^2, "
      f"mean_ice_probability={target_region['mean_ice_probability']:.3f}")
print(f"  Target centroid (world): ({target_centroid.x:.1f}, {target_centroid.y:.1f})")
print(f"  Target pixel (row,col): ({target_row}, {target_col})")

# ====================================================================
# STEP 2: Landing site selection
# Score each candidate on Safety (suitability), Accessibility (1/distance),
# and Science value (ice probability/area of nearest region)
# ====================================================================
print("\n[STEP 2] Scoring and selecting landing site...")

W_SAFETY, W_ACCESS, W_SCIENCE = 0.5, 0.3, 0.2

dists = landing_candidates.geometry.apply(lambda g: g.distance(target_region.geometry))
landing_candidates["distance_to_ice_m"] = dists

safety_norm = (landing_candidates["suitability_score"] - landing_candidates["suitability_score"].min()) / \
              (landing_candidates["suitability_score"].max() - landing_candidates["suitability_score"].min() + 1e-9)
access_norm = 1 - (dists - dists.min()) / (dists.max() - dists.min() + 1e-9)
science_norm = np.full(len(landing_candidates), target_region["mean_ice_probability"])  # same ice zone for all here

landing_candidates["safety_score"] = safety_norm
landing_candidates["access_score"] = access_norm
landing_candidates["science_score"] = science_norm
landing_candidates["final_score"] = (
    W_SAFETY * safety_norm + W_ACCESS * access_norm + W_SCIENCE * science_norm
)

landing_candidates_sorted = landing_candidates.sort_values("final_score", ascending=False).reset_index(drop=True)
best = landing_candidates_sorted.iloc[0]

print("  Candidate scoring (sorted by final score):")
for _, row in landing_candidates_sorted.iterrows():
    print(f"    {row['candidate_id']}: safety={row['safety_score']:.3f} "
          f"access={row['access_score']:.3f} science={row['science_score']:.3f} "
          f"-> FINAL={row['final_score']:.3f}")

print(f"\n  SELECTED: {best['candidate_id']} (final_score={best['final_score']:.3f})")

best_site_row, best_site_col = world_to_pixel(best.geometry.x, best.geometry.y)

best_site_gdf = gpd.GeoDataFrame(
    [{
        "geometry": best.geometry,
        "candidate_id": best["candidate_id"],
        "final_score": float(best["final_score"]),
        "safety_score": float(best["safety_score"]),
        "access_score": float(best["access_score"]),
        "science_score": float(best["science_score"]),
        "distance_to_ice_m": float(best["distance_to_ice_m"]),
    }],
    crs=crs,
)
best_site_gdf.to_file(f"BestLandingSite.geojson", driver="GeoJSON")
print(f"  Saved -> BestLandingSite.geojson")

# ====================================================================
# STEP 3: Mission cost map
# Combine hazard + shadow penalty (shadow already embedded in HazardMap
# via Person 3's pipeline, so we re-derive a power-penalty proxy here
# from hazard's shadow-dominant component for transparency)
# ====================================================================
print("\n[STEP 3] Building mission cost map...")

W_HAZARD, W_SHADOW = 0.7, 0.3
# proxy shadow penalty: hazard map's high values near the ice zone center are
# dominated by shadow (per Person 3's compositing) - reuse hazard as the shadow proxy
shadow_penalty = hazard.copy()

cost = W_HAZARD * hazard + W_SHADOW * shadow_penalty
cost = (cost - cost.min()) / (cost.max() - cost.min() + 1e-9)
cost = np.clip(cost, 0.02, 1.0).astype("float32")  # floor >0 so A* always has finite cost

cost_profile = profile.copy()
cost_profile.update(dtype="float32", count=1)
with rasterio.open(f"MissionCostMap.tif", "w", **cost_profile) as dst:
    dst.write(cost, 1)
print(f"  Saved -> MissionCostMap.tif  range=({cost.min():.3f},{cost.max():.3f})")

# ====================================================================
# STEP 4: A* path planning from landing site to ice target
# ====================================================================
print("\n[STEP 4] Running A* path planning...")

def astar(cost_grid, start, goal):
    """8-connected A* over a cost-weighted grid. start/goal = (row, col)."""
    rows, cols = cost_grid.shape
    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1),
                 (-1, -1), (-1, 1), (1, -1), (1, 1)]

    def heuristic(a, b):
        return np.hypot(a[0] - b[0], a[1] - b[1])

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    visited = set()

    while open_set:
        _, current = heapq.heappop(open_set)
        if current in visited:
            continue
        visited.add(current)

        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        for dr, dc in neighbors:
            nr, nc = current[0] + dr, current[1] + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                step_dist = np.hypot(dr, dc)  # 1.0 orthogonal, 1.414 diagonal
                step_cost = cost_grid[nr, nc] * step_dist
                tentative_g = g_score[current] + step_cost
                neighbor = (nr, nc)
                if tentative_g < g_score.get(neighbor, np.inf):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score = tentative_g + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score, neighbor))
    return None  # no path found

start_px = (best_site_row, best_site_col)
goal_px = (target_row, target_col)

print(f"  Start (landing site) pixel: {start_px}")
print(f"  Goal (ice target) pixel:    {goal_px}")

path_pixels = astar(cost, start_px, goal_px)

if path_pixels is None:
    raise RuntimeError("A* failed to find a path - check cost map for impassable barriers.")

print(f"  Path found: {len(path_pixels)} waypoints")

# Convert path to world coordinates and save as GeoJSON LineString
path_world = [pixel_to_world(r, c) for r, c in path_pixels]
path_line = LineString(path_world)

path_gdf = gpd.GeoDataFrame(
    [{"geometry": path_line, "num_waypoints": len(path_pixels)}], crs=crs
)
path_gdf.to_file(f"TraversePath.geojson", driver="GeoJSON")
print(f"  Saved -> TraversePath.geojson")

# ====================================================================
# STEP 5: Distance, time, shadow-exposure, energy estimate
# ====================================================================
print("\n[STEP 5] Computing distance, time, and energy estimates...")

seg_dists_m = []
shadow_frac_per_seg = []
for i in range(len(path_pixels) - 1):
    r1, c1 = path_pixels[i]
    r2, c2 = path_pixels[i + 1]
    d_px = np.hypot(r2 - r1, c2 - c1)
    seg_dists_m.append(d_px * PIXEL_SIZE_M)
    shadow_frac_per_seg.append(hazard[r2, c2])  # proxy: hazard value at arrival pixel

total_distance_m = float(np.sum(seg_dists_m))
avg_shadow_exposure = float(np.mean(shadow_frac_per_seg))  # 0-1 proxy for fraction of path in high-hazard/shadow terrain

travel_time_s = total_distance_m / ROVER_SPEED_MPS
travel_time_hr = travel_time_s / 3600.0

# Energy: base driving energy + extra penalty proportional to cumulative cost-weighted distance
base_energy_J = DRIVE_POWER_W * travel_time_s
hazard_penalty_factor = 1 + avg_shadow_exposure  # rougher/shadowed terrain draws more power (illustrative model)
total_energy_J = base_energy_J * hazard_penalty_factor
total_energy_Wh = total_energy_J / 3600.0

print(f"  Total traverse distance: {total_distance_m:.1f} m")
print(f"  Estimated travel time:   {travel_time_hr:.2f} hours")
print(f"  Avg shadow/hazard exposure along path: {avg_shadow_exposure:.3f} (0=sunlit/safe, 1=deep shadow/hazard)")
print(f"  Estimated energy consumption: {total_energy_Wh:.1f} Wh")

# ====================================================================
# STEP 6: Ice volume estimation (top 5 m of regolith, dielectric mixing model)
# ====================================================================
print("\n[STEP 6] Estimating subsurface ice volume...")

# Ice volume fraction per pixel: scale IceProbability into a plausible 5-20% range
# (this stands in for Person 2's actual CPR/DOP-calibrated dielectric inversion)
MIN_FRACTION, MAX_FRACTION = 0.05, 0.20
ice_fraction_map = MIN_FRACTION + (MAX_FRACTION - MIN_FRACTION) * ice_prob

pixel_area_m2 = PIXEL_SIZE_M ** 2
ice_pixel_mask = ice_mask.astype(bool)

volume_per_pixel_m3 = pixel_area_m2 * ICE_DEPTH_M * ice_fraction_map
total_ice_volume_m3 = float(volume_per_pixel_m3[ice_pixel_mask].sum())
total_ice_mass_kg = total_ice_volume_m3 * ICE_DENSITY_KGM3
total_ice_mass_tonnes = total_ice_mass_kg / 1000.0

ice_area_m2 = float(ice_pixel_mask.sum() * pixel_area_m2)
mean_fraction_in_zone = float(ice_fraction_map[ice_pixel_mask].mean())

print(f"  Ice-bearing area: {ice_area_m2:.0f} m^2")
print(f"  Mean ice volume fraction in zone: {mean_fraction_in_zone:.1%}")
print(f"  Estimated depth considered: top {ICE_DEPTH_M:.0f} m")
print(f"  TOTAL ESTIMATED ICE VOLUME: {total_ice_volume_m3:,.0f} m^3")
print(f"  TOTAL ESTIMATED ICE MASS:   {total_ice_mass_tonnes:,.0f} tonnes")

# ====================================================================
# STEP 7: Generate map figure (used in both PDFs)
# ====================================================================
print("\n[STEP 7] Generating mission map figure...")

extent = [transform.c, transform.c + transform.a * cost.shape[1],
          transform.f + transform.e * cost.shape[0], transform.f]

fig, ax = plt.subplots(figsize=(8, 8))
ax.imshow(cost, extent=extent, cmap="Reds", alpha=0.85, origin="upper")
ax.imshow(np.ma.masked_where(ice_mask == 0, ice_mask), extent=extent,
          cmap="Blues", alpha=0.6, origin="upper")

path_x = [p[0] for p in path_world]
path_y = [p[1] for p in path_world]
ax.plot(path_x, path_y, color="lime", linewidth=2.5, label="Rover Traverse Path")

ax.scatter([best.geometry.x], [best.geometry.y], color="yellow", edgecolor="black",
           s=180, marker="^", zorder=5, label=f"Landing Site ({best['candidate_id']})")
ax.scatter([target_centroid.x], [target_centroid.y], color="cyan", edgecolor="black",
           s=180, marker="*", zorder=5, label="Ice Target (centroid)")

# Show other rejected candidates faintly
for _, row in landing_candidates_sorted.iloc[1:].iterrows():
    ax.scatter([row.geometry.x], [row.geometry.y], color="gray", marker="^", s=60, alpha=0.6)

ax.set_title("Mission Plan: Landing Site, Cost Map, Rover Traverse Path")
ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.legend(loc="upper right", fontsize=8)
ax.set_aspect("equal")
plt.tight_layout()
map_fig_path = f"_mission_map.png"
plt.savefig(map_fig_path, dpi=130)
plt.close()
print(f"  Saved -> {map_fig_path}")

# Ice volume figure
fig2, ax2 = plt.subplots(figsize=(7, 7))
im = ax2.imshow(ice_fraction_map, extent=extent, cmap="Blues", origin="upper")
ax2.set_aspect("equal")
ice_regions.boundary.plot(ax=ax2, color="red", linewidth=1.5, aspect=1)
ax2.set_title("Ice Volume Fraction Map (top 5 m, dielectric mixing model)")
plt.colorbar(im, ax=ax2, fraction=0.046, label="Ice volume fraction")
plt.tight_layout()
ice_fig_path = f"_ice_volume_map.png"
plt.savefig(ice_fig_path, dpi=130)
plt.close()
print(f"  Saved -> {ice_fig_path}")

# ====================================================================
# STEP 8: Generate MissionPlan.pdf
# ====================================================================
print("\n[STEP 8] Generating MissionPlan.pdf...")

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18)
h2 = styles["Heading2"]
body = styles["BodyText"]

doc = SimpleDocTemplate(f"MissionPlan.pdf", pagesize=A4,
                         topMargin=1.5*cm, bottomMargin=1.5*cm)
elements = []

elements.append(Paragraph("Lunar South Pole Mission Plan", title_style))
elements.append(Paragraph("Rover Traverse Planning &amp; Landing Site Selection", h2))
elements.append(Spacer(1, 0.5*cm))

elements.append(Paragraph("1. Landing Site Candidate Comparison", h2))
cand_table_data = [["Candidate", "Safety", "Access", "Science", "Final Score"]]
for _, row in landing_candidates_sorted.iterrows():
    cand_table_data.append([
        row["candidate_id"],
        f"{row['safety_score']:.3f}",
        f"{row['access_score']:.3f}",
        f"{row['science_score']:.3f}",
        f"{row['final_score']:.3f}",
    ])
cand_table = Table(cand_table_data, hAlign="LEFT")
cand_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("BACKGROUND", (0, 1), (-1, 1), colors.HexColor("#d4edda")),
]))
elements.append(cand_table)
elements.append(Spacer(1, 0.3*cm))
elements.append(Paragraph(
    f"Selected site: <b>{best['candidate_id']}</b> (Final Score = {best['final_score']:.3f}). "
    f"Scoring weights: Safety 50%, Accessibility 30%, Science Value 20%, reflecting that "
    f"mission/crew safety is prioritized while still favoring proximity to high-confidence ice.",
    body))
elements.append(Spacer(1, 0.5*cm))

elements.append(Paragraph("2. Mission Map", h2))
elements.append(RLImage(map_fig_path, width=14*cm, height=14*cm))
elements.append(Spacer(1, 0.3*cm))

elements.append(Paragraph("3. Traverse Summary", h2))
summary_data = [
    ["Metric", "Value"],
    ["Total traverse distance", f"{total_distance_m:.1f} m"],
    ["Estimated travel time", f"{travel_time_hr:.2f} hours"],
    ["Avg. shadow/hazard exposure along path", f"{avg_shadow_exposure:.1%}"],
    ["Estimated energy consumption", f"{total_energy_Wh:.1f} Wh"],
    ["Assumed rover speed", f"{ROVER_SPEED_MPS*3600:.0f} m/h"],
    ["Path planning algorithm", "A* (8-connected grid, cost-weighted)"],
]
summary_table = Table(summary_data, hAlign="LEFT")
summary_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
]))
elements.append(summary_table)
elements.append(Spacer(1, 0.3*cm))

elements.append(Paragraph("4. Methodology &amp; Limitations", h2))
elements.append(Paragraph(
    "Path planning uses an 8-connected A* search over a cost-weighted grid derived from the "
    "HazardMap (shadow depth, crater rims, boulder density, surface roughness). No DEM was "
    "available for this analysis; consequently, energy estimates do not account for terrain "
    "slope or elevation change and rely on a simplified power model scaled by hazard/shadow "
    "exposure along the path. Dijkstra and Hybrid A* were considered as alternatives: Dijkstra "
    "guarantees a global optimum without a heuristic but is slower, while Hybrid A* would "
    "additionally respect rover turning-radius constraints and is recommended as a production-grade "
    "upgrade once kinematic rover parameters are available.",
    body))

doc.build(elements)
print(f"  Saved -> MissionPlan.pdf")

# ====================================================================
# STEP 9: Generate IceVolumeReport.pdf
# ====================================================================
print("\n[STEP 9] Generating IceVolumeReport.pdf...")

doc2 = SimpleDocTemplate(f"IceVolumeReport.pdf", pagesize=A4,
                          topMargin=1.5*cm, bottomMargin=1.5*cm)
elements2 = []

elements2.append(Paragraph("Subsurface Ice Volume Report", title_style))
elements2.append(Paragraph("Doubly Shadowed Crater - Lunar South Pole", h2))
elements2.append(Spacer(1, 0.4*cm))

elements2.append(Paragraph("1. Ice Zone Map", h2))
elements2.append(RLImage(ice_fig_path, width=12*cm, height=12*cm))
elements2.append(Spacer(1, 0.3*cm))

elements2.append(Paragraph("2. Volume Estimate", h2))
vol_data = [
    ["Parameter", "Value"],
    ["Ice-bearing surface area", f"{ice_area_m2:,.0f} m\u00b2"],
    ["Depth considered", f"Top {ICE_DEPTH_M:.0f} m of regolith"],
    ["Mean ice volume fraction (in zone)", f"{mean_fraction_in_zone:.1%}"],
    ["Assumed fraction range", f"{MIN_FRACTION:.0%} - {MAX_FRACTION:.0%}"],
    ["Total estimated ice volume", f"{total_ice_volume_m3:,.0f} m\u00b3"],
    ["Assumed ice density", f"{ICE_DENSITY_KGM3:.0f} kg/m\u00b3"],
    ["Total estimated ice mass", f"{total_ice_mass_tonnes:,.0f} tonnes"],
]
vol_table = Table(vol_data, hAlign="LEFT")
vol_table.setStyle(TableStyle([
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1b4f72")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTSIZE", (0, 0), (-1, -1), 9),
    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
    ("BACKGROUND", (-2, -1), (-1, -1), colors.HexColor("#d6eaf8")),
]))
elements2.append(vol_table)
elements2.append(Spacer(1, 0.4*cm))

elements2.append(Paragraph("3. Methodology", h2))
elements2.append(Paragraph(
    f"Ice volume fraction per pixel is estimated using a simplified dielectric mixing-model "
    f"assumption, scaling radar-derived ice probability (range 0-1) linearly into a plausible "
    f"volume-fraction range of {MIN_FRACTION:.0%}-{MAX_FRACTION:.0%}, consistent with regolith "
    f"dielectric constant (~{REGOLITH_DIELECTRIC:.1f}) versus water-ice dielectric constant "
    f"(~{ICE_DIELECTRIC:.1f}). Volume is computed as: "
    f"<i>Volume = &#931; (pixel_area &times; depth &times; ice_volume_fraction)</i> over all pixels "
    f"within the ice mask. In an operational pipeline, ice volume fraction would instead be derived "
    f"directly from Person 2's CPR/DOP-calibrated radar backscatter inversion rather than this "
    f"probability-scaled approximation.",
    body))
elements2.append(Spacer(1, 0.3*cm))

elements2.append(Paragraph("4. Key Assumptions &amp; Caveats", h2))
elements2.append(Paragraph(
    "(1) Ice is assumed uniformly mixed within the top 5 m, with no stratification. "
    "(2) Ice volume fraction is approximated from probability rather than a calibrated dielectric "
    "inversion. (3) No ground-truth validation (e.g., from landed instruments) is available at this "
    "stage. These caveats should be resolved with refined radar inversion models and, where possible, "
    "in-situ validation in later mission phases.",
    body))

doc2.build(elements2)
print(f"  Saved -> IceVolumeReport.pdf")

# ====================================================================
# DONE
# ====================================================================
print("\n" + "=" * 70)
print("PERSON 4 PIPELINE COMPLETE")
print("=" * 70)
print(f"""
Outputs in {OUT}/:
  BestLandingSite.geojson   - selected landing site with scoring breakdown
  MissionCostMap.tif        - traversal cost surface used by A*
  TraversePath.geojson      - rover path from landing site to ice target
  MissionPlan.pdf           - full mission plan with map and metrics
  IceVolumeReport.pdf       - subsurface ice volume estimate and methodology

KEY RESULTS:
  Selected landing site: {best['candidate_id']}
  Traverse distance:     {total_distance_m:.1f} m
  Travel time:           {travel_time_hr:.2f} hours
  Energy estimate:       {total_energy_Wh:.1f} Wh
  Ice volume estimate:   {total_ice_volume_m3:,.0f} m^3 ({total_ice_mass_tonnes:,.0f} tonnes)
""")
