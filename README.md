# ISRO BAH 2026 - Lunar Rover Autonomous Mission Planning

This repository contains an autonomous lunar rover planning framework developed for the **ISRO BAH 2026 challenge**.

The project focuses on end-to-end rover mission planning on lunar terrain, combining:

1. **3D terrain-aware path planning using mesh geometry**
2. **Mission-level planning using hazard maps, landing suitability maps, and ice probability maps**

The system can select a suitable landing location, generate a safe rover traverse path, estimate mission requirements, and generate scientific mission reports.

---

# Project Overview

The complete workflow:

```
Lunar Terrain Data
        |
        |
        v
Terrain Analysis
        |
        |
        +-----------------------------+
        |                             |
        v                             v
3D Mesh Path Planning          Mission Planning Pipeline
(moon_obj_pathplan.py)         (01_person4_pipeline.py)
        |                             |
        |                             |
        v                             v
A* Terrain Traversal          Landing Site Selection
        |                             |
        v                             v
Safe Rover Path              Cost Map Generation
        |                             |
        v                             v
CSV Waypoints                A* Traverse Planning
                                      |
                                      v
                          Distance / Energy Estimate
                                      |
                                      v
                          Ice Volume Estimation
                                      |
                                      v
                          PDF Mission Reports
```

---

# Repository Structure

```
isro-bah-2026/

│
├── moon_obj_pathplan.py
│       3D terrain mesh based rover path planner
│
├── 01_person4_pipeline.py
│       Complete rover mission planning pipeline
│
├── hazardmap.obj
│       3D lunar terrain mesh input
│
│       Inputs
│── HazardMap.tif
│── LandingSuitability.tif
│── IceProbability.tif
│── IceMask.tif
│── IceRegions.geojson
│── LandingCandidates.geojson
│       Outputs
│── BestLandingSite.geojson
│── MissionCostMap.tif
│── TraversePath.geojson
│── MissionPlan.pdf
│── IceVolumeReport.pdf
│
└── README.md
```

---

# Module 1: 3D Terrain Path Planning

## `moon_obj_pathplan.py`

This module performs autonomous rover path planning directly on a 3D lunar terrain mesh.

The terrain is represented as an OBJ file containing:

- Vertices
- Triangle faces
- Surface geometry

Open3D is used to process the mesh and extract terrain information.

---

# Algorithm Pipeline

## 1. Terrain Mesh Processing

The OBJ terrain file is loaded:

```
hazardmap.obj
```

The mesh is converted into a graph:

- Each vertex → graph node
- Connected triangle vertices → graph edges

This allows graph-search based navigation.

---

# 2. Terrain Slope Analysis

Surface normals are calculated for every terrain vertex.

The slope angle is computed using:

```
angle = acos(normal_z)
```

where:

- Smaller angle → flatter terrain
- Larger angle → steeper terrain

A slope threshold is defined:

```python
THRESHOLD_ANGLE = 30
```

Flat regions are considered safer rover traversal areas.

---

# 3. Terrain Traversal Cost

Each vertex receives a cost based on terrain steepness.

The cost function:

```python
cost =
1 +
ROUGHNESS_PENALTY *
(angle / 90)^2
```

Current parameter:

```python
ROUGHNESS_PENALTY = 20.0
```

Steeper terrain receives a higher traversal penalty.

---

# 4. A* Path Planning

A* search is performed over the terrain graph.

The movement cost:

```
distance × terrain cost
```

The heuristic:

```
Euclidean distance to goal
```

The resulting path minimizes:

- Travel distance
- Terrain difficulty

---

# Output

The generated rover path is saved as:

```
astar_path.csv
```

Format:

| index | x | y | z |
|-|-|-|-|
|0|...|...|...|

Each row represents a rover waypoint.

---

# Visualization

The Open3D viewer displays:

- Lunar terrain mesh
- Start location
- Goal location
- Planned rover path

The camera can be projected onto the XY plane for horizontal navigation analysis.

---

# Module 2: Full Rover Mission Planning Pipeline

## `01_person4_pipeline.py`

This module performs mission-level rover planning using synthetic mission data products.

The pipeline combines:

- Landing site selection
- Hazard-aware navigation
- Rover traverse planning
- Energy estimation
- Ice resource estimation
- Report generation

---

# Input Data

Located in:

```
data/inputs/
```

---

## HazardMap.tif

Represents terrain hazards:

- Rough terrain
- Craters
- Boulder regions
- Shadow hazards

Used for rover traversal cost.

---

## LandingSuitability.tif

Represents landing safety.

Higher values indicate better landing conditions.

---

## IceProbability.tif

Probability map showing possible subsurface ice regions.

---

## IceMask.tif

Binary map identifying detected ice-bearing regions.

---

## IceRegions.geojson

Contains detected ice regions:

- Region ID
- Area
- Mean ice probability
- Geometry

---

## LandingCandidates.geojson

Contains possible landing locations.

Each candidate contains:

- Candidate ID
- Position
- Suitability score

---

# Mission Pipeline

## 1. Ice Target Selection

The pipeline identifies the most promising ice region.

The region with the highest mean ice probability is selected as the mission target.

---

# 2. Landing Site Selection

Candidate landing locations are scored using:

```
Final Score =
0.5 × Safety
+
0.3 × Accessibility
+
0.2 × Science Value
```

Priority:

1. Safe landing
2. Short distance to ice
3. Scientific value

The best candidate is selected.

Output:

```
BestLandingSite.geojson
```

---

# 3. Mission Cost Map Generation

A rover traversal cost map is created:

```
Cost =
0.7 × Hazard
+
0.3 × Shadow Penalty
```

High-cost regions represent difficult terrain.

Output:

```
MissionCostMap.tif
```

---

# 4. Rover Traverse Planning

A* search is performed on the cost map.

The planner uses an 8-connected grid:

```
Up
Down
Left
Right
Diagonal movement
```

The path is optimized between:

```
Landing Site
       |
       v
Ice Target Region
```

Output:

```
TraversePath.geojson
```

---

# 5. Distance, Time and Energy Estimation

Mission parameters:

```python
ROVER_SPEED_MPS = 0.05
ROVER_MASS_KG = 30
DRIVE_POWER_W = 25
MOON_GRAVITY = 1.62
```

The pipeline estimates:

## Traverse Distance

Total rover travel distance.

---

## Travel Time

Calculated:

```
time = distance / rover speed
```

---

## Energy Consumption

Estimated using:

```
Energy =
Power × Time × Hazard Penalty
```

Higher hazard exposure increases energy consumption.

---

# 6. Subsurface Ice Volume Estimation

Ice resources are estimated using:

Assumptions:

```
Depth considered = 5 m
Ice density = 917 kg/m³
```

Ice fraction model:

```
5% - 20% volume fraction
```

Volume calculation:

```
Volume =
pixel area × depth × ice fraction
```

Outputs:

- Ice-bearing area
- Estimated ice volume
- Estimated ice mass

---

# Generated Outputs

The pipeline generates:

```
data/outputs/

├── BestLandingSite.geojson
│
├── MissionCostMap.tif
│
├── TraversePath.geojson
│
├── MissionPlan.pdf
│
└── IceVolumeReport.pdf
```

---

# Reports

## MissionPlan.pdf

Contains:

- Landing candidate comparison
- Selected landing site
- Mission map
- Rover traverse path
- Distance estimate
- Energy estimate
- Planning methodology

---

## IceVolumeReport.pdf

Contains:

- Ice distribution map
- Ice volume estimation
- Ice mass estimation
- Methodology
- Assumptions and limitations

---

# Installation

Requirements:

- Python 3.x
- Ubuntu Linux recommended

Install dependencies:

```bash
pip install numpy
pip install open3d
pip install rasterio
pip install geopandas
pip install shapely
pip install matplotlib
pip install reportlab
```

---

# Running

## 3D Mesh Path Planner

Ensure:

```
hazardmap.obj
```

is present.

Run:

```bash
python3 moon_obj_pathplan.py
```

---

## Mission Planning Pipeline

Ensure:

```
data/inputs/
```

contains all required files.

Run:

```bash
python3 01_person4_pipeline.py
```

---

# Coordinate System

The system uses:

```
X axis → Horizontal direction
Y axis → Horizontal direction
Z axis → Terrain elevation
```

The mesh planner preserves the terrain elevation while planning mainly in the XY plane.

---

# Limitations

Current implementation uses simplified assumptions for:

- Energy estimation
- Ice fraction calculation
- Terrain difficulty

Future improvements:

- DEM based slope-aware energy model
- Hybrid A* rover kinematics
- Real radar dielectric inversion
- ROS2 rover integration
- Real-time obstacle avoidance
- Autonomous localization

---

# Author

Prahlad

ISRO BAH 2026
