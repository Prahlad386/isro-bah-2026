"""
====================================================================
SYNTHETIC INPUT GENERATOR - Person 4 (Rover Traverse & Ice Volume Lead)
====================================================================
Purpose:
  Simulates the files Person 4 would normally RECEIVE from Person 2
  (ice detection) and Person 3 (terrain hazard / landing site analysis),
  so Person 4 can build and demo the rover-planning + ice-volume pipeline
  end-to-end without waiting for real upstream data.

Run this once:  python3 00_generate_synthetic_inputs.py

Produces 6 files in data/inputs/ :
  1. HazardMap.tif            - 0-1 terrain danger raster        (from Person 3)
  2. LandingSuitability.tif   - 0-1 safe-landing-score raster    (from Person 3)
  3. IceProbability.tif       - 0-1 confidence raster            (from Person 2)
  4. IceMask.tif              - binary ice/no-ice raster        (from Person 2)
  5. IceRegions.geojson       - ice zone as a polygon            (from Person 2)
  6. LandingCandidates.geojson- ranked candidate landing points  (from Person 3)
====================================================================
"""

import numpy as np
import rasterio
from rasterio.transform import from_origin
from scipy.ndimage import gaussian_filter, maximum_filter, label
import geopandas as gpd
from shapely.geometry import Point, MultiPoint

np.random.seed(7)

# --------------------------------------------------------------
# Common spatial grid - every file shares this exact grid so all
# rasters/vectors line up pixel-for-pixel and coordinate-for-coordinate
# --------------------------------------------------------------
SIZE = 500                      # 500 x 500 pixel scene
PIXEL_SIZE_M = 5.0              # 5 m/pixel -> 2.5 km x 2.5 km area
CRS = "ESRI:104903"             # Moon 2000 Geographic CRS
ORIGIN_X, ORIGIN_Y = -200000.0, -200000.0
transform = from_origin(ORIGIN_X, ORIGIN_Y, PIXEL_SIZE_M, PIXEL_SIZE_M)

OUT = "/home/claude/person4_project/data/inputs"

def save_tif(path, arr, dtype="float32"):
    with rasterio.open(
        path, "w", driver="GTiff", height=arr.shape[0], width=arr.shape[1],
        count=1, dtype=dtype, crs=CRS, transform=transform,
    ) as dst:
        dst.write(arr.astype(dtype), 1)
    print(f"  -> {path.split('/')[-1]:28s} range=({arr.min():.3f}, {arr.max():.3f})")

def pixel_to_world(row, col):
    x, y = transform * (col, row)
    return x, y

print("Generating shared crater scene geometry...")
y, x = np.mgrid[0:SIZE, 0:SIZE]
cx, cy = SIZE / 2, SIZE / 2
outer_r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
icx, icy = cx - 60, cy + 40          # inner "doubly shadowed" crater, offset from center
inner_r = np.sqrt((x - icx) ** 2 + (y - icy) ** 2)

# ================================================================
# Build hazard components (shadow, craters, boulders, roughness)
# ================================================================
print("\nBuilding hazard components (shadow / craters / boulders / roughness)...")

shadow_intensity = np.clip(1.2 - inner_r / 50, 0, 1) ** 2
shadow_intensity += 0.3 * np.clip(1.0 - outer_r / 220, 0, 1)
shadow_intensity = np.clip(shadow_intensity, 0, 1)

crater_field = np.zeros((SIZE, SIZE))
for _ in range(18):
    rx, ry = np.random.randint(40, SIZE - 40, 2)
    r = np.random.uniform(8, 20)
    d = np.sqrt((x - rx) ** 2 + (y - ry) ** 2)
    crater_field += np.exp(-((d - r) ** 2) / (2 * 3 ** 2)) * 0.6

boulder_field = np.zeros((SIZE, SIZE))
n_boulders = 300
bx = np.random.randint(0, SIZE, n_boulders)
by = np.random.randint(0, SIZE, n_boulders)
for i in range(n_boulders):
    r = np.random.uniform(1, 3)
    h = np.random.uniform(0.3, 1.0)
    boulder_field += h * np.exp(-(((x - bx[i]) ** 2 + (y - by[i]) ** 2)) / (2 * r ** 2))

roughness = np.zeros((SIZE, SIZE))
for octave in range(1, 5):
    freq = octave * 3
    roughness += (1 / octave) * np.abs(np.sin(2 * np.pi * freq * x / SIZE) * np.cos(2 * np.pi * freq * y / SIZE))
roughness = (roughness - roughness.min()) / (roughness.max() - roughness.min())

# ================================================================
# FILE 1: HazardMap.tif
# Purpose: tells Person 4 which areas are dangerous to land/drive on
# (0 = perfectly safe, 1 = extremely hazardous)
# Combines shadow depth + crater rims + boulders + surface roughness
# ================================================================
hazard = (
    0.45 * shadow_intensity +
    0.20 * np.clip(crater_field, 0, 1) +
    0.20 * np.clip(boulder_field, 0, 1) +
    0.15 * roughness
)
hazard = gaussian_filter(hazard, sigma=1.5)
hazard = np.clip(hazard, 0, 1).astype("float32")

print("\n[1/6] HazardMap.tif")
print("  Purpose: 0-1 danger score per pixel (shadow+craters+boulders+roughness).")
print("           Used to penalize unsafe areas in the rover cost map & site scoring.")
save_tif(f"{OUT}/HazardMap.tif", hazard)

# ================================================================
# FILE 2: LandingSuitability.tif
# Purpose: tells Person 4 how good each pixel is as a landing spot
# (inverse of hazard, with a bonus for being a reasonable distance from the ice)
# ================================================================
dist_to_inner_m = inner_r * PIXEL_SIZE_M
proximity_bonus = np.clip(1 - np.abs(dist_to_inner_m - 250) / 400, 0, 1) * 0.25
suitability = np.clip((1 - hazard) * 0.8 + proximity_bonus, 0, 1).astype("float32")

print("\n[2/6] LandingSuitability.tif")
print("  Purpose: 0-1 composite safety+accessibility score per pixel.")
print("           Used to generate and rank candidate landing points.")
save_tif(f"{OUT}/LandingSuitability.tif", suitability)

# ================================================================
# FILE 3 & 4: IceProbability.tif  &  IceMask.tif
# Purpose: tells Person 4 WHERE the ice is and HOW confident Person 2 is
# IceProbability = continuous 0-1 confidence; IceMask = thresholded binary version
# ================================================================
ice_prob = np.clip(1.3 - inner_r / 35, 0, 1) ** 1.5
ice_prob = gaussian_filter(ice_prob, sigma=1.0).astype("float32")
ice_mask = (ice_prob > 0.5).astype("uint8")

print("\n[3/6] IceProbability.tif")
print("  Purpose: 0-1 ice confidence per pixel, from radar CPR/DOP analysis.")
print("           Used as the ROVER'S TARGET and for ice volume calculation.")
save_tif(f"{OUT}/IceProbability.tif", ice_prob)

print("\n[4/6] IceMask.tif")
print("  Purpose: binary (0/1) thresholded version of IceProbability.")
print("           Used for quick area calculations and masking.")
save_tif(f"{OUT}/IceMask.tif", ice_mask, dtype="uint8")

# ================================================================
# FILE 5: IceRegions.geojson
# Purpose: vector polygon(s) outlining the high-confidence ice zone
# Easier to do distance/containment calculations with a polygon than a raster
# ================================================================
labeled, n = label(ice_mask)
regions = []
for region_id in range(1, n + 1):
    rows, cols = np.where(labeled == region_id)
    if len(rows) < 5:
        continue
    pts = [pixel_to_world(r, c) for r, c in zip(rows, cols)]
    hull = MultiPoint(pts).convex_hull
    regions.append({
        "geometry": hull,
        "region_id": region_id,
        "mean_ice_probability": float(ice_prob[rows, cols].mean()),
        "area_m2": float(len(rows) * PIXEL_SIZE_M ** 2),
    })
gdf_ice = gpd.GeoDataFrame(regions, crs=CRS)
gdf_ice.to_file(f"{OUT}/IceRegions.geojson", driver="GeoJSON")

print("\n[5/6] IceRegions.geojson")
print(f"  Purpose: vector polygon(s) of the ice zone ({n} region found).")
print("           Used for distance-to-ice calculations and as the path target.")
if regions:
    print(f"  -> IceRegions.geojson    {n} region(s), area={regions[0]['area_m2']:.0f} m^2")

# ================================================================
# FILE 6: LandingCandidates.geojson
# Purpose: a handful of ranked candidate landing points for Person 4 to choose from
# Found via local-maxima detection on the suitability raster, spaced apart, outside hazard zones
# ================================================================
local_max = (maximum_filter(suitability, size=25) == suitability) & (suitability > 0.55) & (hazard < 0.35)
cand_rows, cand_cols = np.where(local_max)
candidates_sorted = sorted(zip(cand_rows, cand_cols, suitability[cand_rows, cand_cols]), key=lambda t: -t[2])

selected = []
min_sep_px = 40
for r, c, s in candidates_sorted:
    if all(np.hypot(r - sr, c - sc) > min_sep_px for sr, sc, _ in selected):
        selected.append((r, c, s))
    if len(selected) >= 5:
        break

cand_records = []
for i, (r, c, s) in enumerate(selected):
    wx, wy = pixel_to_world(r, c)
    dist_to_ice_m = float(inner_r[r, c] * PIXEL_SIZE_M)
    cand_records.append({
        "geometry": Point(wx, wy),
        "candidate_id": f"LC-{i+1:02d}",
        "suitability_score": float(s),
        "distance_to_ice_m": dist_to_ice_m,
        "hazard_value": float(hazard[r, c]),
    })
gdf_cand = gpd.GeoDataFrame(cand_records, crs=CRS)
gdf_cand.to_file(f"{OUT}/LandingCandidates.geojson", driver="GeoJSON")

print("\n[6/6] LandingCandidates.geojson")
print(f"  Purpose: {len(cand_records)} ranked candidate landing sites with safety/")
print("           distance/hazard attributes. Person 4 selects the best one from these.")
for rec in cand_records:
    print(f"    {rec['candidate_id']}: suitability={rec['suitability_score']:.3f}, "
          f"dist_to_ice={rec['distance_to_ice_m']:.0f}m, hazard={rec['hazard_value']:.3f}")

print("\n" + "=" * 60)
print("DONE - all 6 synthetic input files generated in data/inputs/")
print("=" * 60)
