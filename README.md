# ISRO BAH 2026 - Lunar Terrain Path Planning

This repository contains a terrain-aware autonomous path planning implementation developed for the ISRO BAH 2026 challenge.

The project focuses on autonomous rover navigation over a lunar terrain surface represented as a 3D mesh. The planner processes terrain geometry, identifies steep regions, assigns traversal costs, and computes a safe path between a given start point and goal point using the A* search algorithm.

The main implementation file is:

```
moon_obj_pathplan.py
```

---

# Features

- 3D lunar terrain mesh processing
- Terrain slope estimation using surface normals
- Mesh connectivity graph generation
- A* based path planning
- Terrain-aware traversal cost calculation
- Slope/roughness penalty handling
- Automatic nearest vertex selection for start and goal points
- XY plane navigation support
- CSV waypoint generation
- Open3D based terrain and path visualization

---

# Algorithm Overview

## 1. Terrain Mesh Loading

The input terrain is provided as an OBJ mesh file.

Example:

```
hazardmap.obj
```

The mesh contains:

- Vertices
- Triangle faces
- Surface geometry information

The terrain mesh is loaded and processed using Open3D.

---

# 2. Terrain Steepness Calculation

Terrain steepness is estimated using surface normals.

For every vertex, the angle between the terrain normal and the vertical direction is calculated:

```
angle = acos(normal_z)
```

A smaller angle indicates flatter terrain, while a larger angle indicates steeper terrain.

The slope threshold is defined using:

```python
THRESHOLD_ANGLE = 30
```

Vertices below this threshold are considered safer regions.

---

# 3. Mesh Graph Generation

The triangular mesh is converted into a graph representation.

Conversion:

- Each mesh vertex becomes a graph node
- Connected triangle vertices become graph edges

This allows graph search algorithms to operate directly on the terrain surface.

---

# 4. Terrain Cost Function

Each vertex is assigned a traversal cost based on terrain steepness.

The cost function used is:

```python
cost =
1 +
ROUGHNESS_PENALTY *
(angle / 90)^2
```

Higher slope regions receive higher costs, causing the planner to avoid steep terrain.

Current value:

```python
ROUGHNESS_PENALTY = 20.0
```

---

# 5. A* Path Planning

The generated terrain graph is searched using the A* algorithm.

The movement cost between connected vertices is:

```
distance × average terrain cost
```

The heuristic used is Euclidean distance:

```
distance(current node, goal node)
```

The final path is optimized considering:

- Short travel distance
- Lower terrain steepness

---

# Installation

## Requirements

Tested with:

- Ubuntu Linux
- Python 3.x

Install dependencies:

```bash
pip install numpy
pip install open3d
```

or:

```bash
pip install -r requirements.txt
```

---

# Running the Code

Place the terrain mesh file in the same directory as the Python script.

Example:

```
isro-bah-2026/

├── moon_obj_pathplan.py
├── hazardmap.obj
```

Run:

```bash
python3 moon_obj_pathplan.py
```

---

# Input Configuration

The start and goal coordinates can be modified inside:

```python
START_POINT = np.array([
    x,
    y,
    z
])


GOAL_POINT = np.array([
    x,
    y,
    z
])
```

Example:

```python
START_POINT = np.array([
    -0.664,
    -0.696,
    0.0
])


GOAL_POINT = np.array([
    -0.48,
    -0.32,
    0.0
])
```

The algorithm automatically finds the nearest terrain mesh vertices corresponding to these coordinates.

---

# Output

## Visualization

The Open3D window displays:

- Lunar terrain mesh
- Start point (green)
- Goal point (blue)
- Planned path (red)

The camera view is projected onto the XY plane to analyze horizontal rover navigation.

---

## Generated Waypoints

The planned path coordinates are saved as:

```
astar_path.csv
```

CSV format:

| index | x | y | z |
|------|---|---|---|
|0|...|...|...|
|1|...|...|...|

Each row represents a rover navigation waypoint.

---

# Folder Structure

Recommended structure:

```
isro-bah-2026/

│
├── moon_obj_pathplan.py
├── hazardmap.obj
├── astar_path.csv
├── requirements.txt
└── README.md
```

---

# Coordinate System

The planner uses:

```
X axis → Horizontal direction
Y axis → Horizontal direction
Z axis → Terrain height
```

Path planning is performed primarily on the XY plane, while the Z coordinate is preserved from the terrain mesh so that the generated path follows the actual surface elevation.

---

# Visualization Debug Options

The script supports visualization of:

- Surface normals
- X-axis vectors
- Y-axis vectors
- Z-up reference vectors

These can be enabled by uncommenting:

```python
vis.add_geometry(normal_line_set)
vis.add_geometry(z_line_set)
vis.add_geometry(x_line_set)
vis.add_geometry(y_line_set)
```

---

# Future Improvements

Possible extensions:

- ROS2 integration
- Real rover localization
- Dynamic obstacle avoidance
- Terrain costmap generation
- Path smoothing
- Rover kinematic constraints
- Real-time terrain updates

---

# Author

Prahlad

ISRO BAH 2026
