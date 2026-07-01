import heapq
from collections import defaultdict
import os
import open3d as o3d
import numpy as np
import csv

# --- FORCE X11 BACKEND FOR WAYLAND COMPATIBILITY ---
os.environ["GDK_BACKEND"] = "x11"
os.environ["XDG_SESSION_TYPE"] = "x11"
if "WAYLAND_DISPLAY" in os.environ:
    del os.environ["WAYLAND_DISPLAY"]

THRESHOLD_ANGLE = 30

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

# START_VERTEX = 20
# GOAL_VERTEX = 25

ROUGHNESS_PENALTY = 20.0

def build_graph(triangles):
    graph = defaultdict(set)

    for tri in triangles:
        a, b, c = tri

        graph[a].add(b)
        graph[a].add(c)

        graph[b].add(a)
        graph[b].add(c)

        graph[c].add(a)
        graph[c].add(b)

    return graph


def heuristic(vertices, current, goal):
    return np.linalg.norm(
        vertices[current] - vertices[goal]
    )


def astar(graph,
          vertices,
          vertex_costs,
          start,
          goal):

    open_set = []

    heapq.heappush(open_set, (0, start))

    came_from = {}

    g_score = {start: 0}

    while open_set:

        _, current = heapq.heappop(open_set)

        if current == goal:

            path = [current]

            while current in came_from:
                current = came_from[current]
                path.append(current)

            return path[::-1]

        for neighbor in graph[current]:

            edge_distance = np.linalg.norm(
                vertices[current][[0, 1]] - vertices[neighbor][[0, 1]]
            )

            move_cost = (
                edge_distance *
                (vertex_costs[current] +
                 vertex_costs[neighbor]) * 0.5
            )

            tentative_g = (
                g_score[current] +
                move_cost
            )

            if (
                neighbor not in g_score
                or tentative_g < g_score[neighbor]
            ):

                came_from[neighbor] = current

                g_score[neighbor] = tentative_g

                f_score = (
                    tentative_g +
                    heuristic(
                        vertices,
                        neighbor,
                        goal
                    )
                )

                heapq.heappush(
                    open_set,
                    (f_score, neighbor)
                )

    return None

def find_nearest_in_component(vertices, point, component):
    component_list = list(component)
    component_vertices = vertices[component_list]
    distances = np.linalg.norm(
        component_vertices[:, [0, 1]] - point[[0, 1]],
        axis=1
    )
    nearest_local_idx = np.argmin(distances)
    return component_list[nearest_local_idx]

def get_all_components(graph):
    visited = set()
    components = []
    for node in graph:
        if node not in visited:
            component = set()
            stack = [node]
            while stack:
                n = stack.pop()
                if n in visited: continue
                visited.add(n)
                component.add(n)
                stack.extend(graph[n] - visited)
            components.append(component)
    return sorted(components, key=len, reverse=True)

def extract_and_visualize_steepness(obj_file_path, output_png="moon_env.png", line_length=0.1, sample_rate=12):
    mesh = o3d.io.read_triangle_mesh(obj_file_path)
    mesh.merge_close_vertices(0.001)  # weld duplicate vertices
    
    if not mesh.has_vertex_normals():
        print("No vertex normals found in file. Estimating them now...")
        mesh.compute_vertex_normals()
        
    if not mesh.has_triangle_normals():
        mesh.compute_triangle_normals()

    vertices = np.asarray(mesh.vertices)

    start_idx = np.argmin(
        np.linalg.norm(
            vertices[:, [0, 1]] -
            START_POINT[[0, 1]],
            axis=1
        )
    )

    goal_idx = np.argmin(
        np.linalg.norm(
            vertices[:, [0, 1]] -
            GOAL_POINT[[0, 1]],
            axis=1
        )
    )

    print(
        "Start vertex:",
        start_idx,
        vertices[start_idx]
    )

    print(
        "Goal vertex:",
        goal_idx,
        vertices[goal_idx]
    )

    vertex_normals = np.asarray(mesh.vertex_normals)

    print(f"\n--- Open3D Extraction Complete ---")
    print(f"Vertex normals shape: {vertex_normals.shape}")
    
    # --- 4. CALCULATE STEEPNESS AND COLOR VERTICES ---
    # Use Y-axis as "up"
    y_components = vertex_normals[:, 2]

    y_components = np.clip(
        y_components,
        -1.0,
        1.0
    )

    angles_from_up = np.degrees(
        np.arccos(y_components)
    )

    num_vertices = len(vertex_normals)

    vertex_colors = np.zeros(
        (num_vertices, 3)
    )

    flat_vertices_mask = (
        angles_from_up < THRESHOLD_ANGLE
    )

    vertex_colors[flat_vertices_mask] = [
        1.0,
        1.0,
        1.0
    ]

    mesh.vertex_colors = (
        o3d.utility.Vector3dVector(
            vertex_colors
        )
    )

    # Build graph from mesh connectivity
    triangles = np.asarray(mesh.triangles)
    graph = build_graph(triangles)

    # After building the graph, find the largest component
    components = get_all_components(graph)
    largest_component = components[0]  # already sorted by size
    start_idx = find_nearest_in_component(vertices, START_POINT, largest_component)
    goal_idx = find_nearest_in_component(vertices, GOAL_POINT, largest_component)

    print(f"After welding - vertices: {len(vertices)}, triangles: {len(triangles)}")
    print(f"Components after welding: {len(components)}")

    # Terrain traversal costs
    vertex_costs = (
        1.0 +
        ROUGHNESS_PENALTY *
        (angles_from_up / 90.0) ** 2
    )

    # Run A*
    path = astar(
        graph,
        vertices,
        vertex_costs,
        start_idx,
        goal_idx
    )
    if path is None:
        raise RuntimeError("No path found")
    for idx in path:
        vertex_colors[idx] = [1.0, 0.0, 0.0]
    mesh.vertex_colors = o3d.utility.Vector3dVector(
        vertex_colors
    )

    print(f"Path found with {len(path)} vertices")

    #Storing the coordinates in csv file
    if path is None:
        raise RuntimeError("No path found")

    # Extract 3D coordinates for each vertex in path
    path_coordinates = vertices[path]  # shape: (N, 3)

    # Save to CSV
    csv_path = "astar_path.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'x', 'y', 'z'])  # header
        for i, (idx, coord) in enumerate(zip(path, path_coordinates)):
            writer.writerow([i, coord[0], coord[1], coord[2]])

    print(f"Path saved to {csv_path} with {len(path)} waypoints")

    start_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.05
    )
    start_sphere.paint_uniform_color([0,1,0])
    start_sphere.translate(vertices[start_idx])

    goal_sphere = o3d.geometry.TriangleMesh.create_sphere(
        radius=0.05
    )
    goal_sphere.paint_uniform_color([0,0,1])
    goal_sphere.translate(vertices[goal_idx])
    
    # --- 4.5. DOWNSAMPLE FOR ACCELERATED RENDERING ---
    sampled_vertices = vertices[::sample_rate]
    sampled_normals = vertex_normals[::sample_rate]
    num_sampled = len(sampled_vertices)
    
    # --- 4.6. GENERATE GREEN SURFACE NORMALS ---
    normal_ends = sampled_vertices + (sampled_normals * line_length)
    all_normal_points = np.vstack((sampled_vertices, normal_ends))
    normal_lines = [[i, i + num_sampled] for i in range(num_sampled)]
    
    normal_line_set = o3d.geometry.LineSet()
    normal_line_set.points = o3d.utility.Vector3dVector(all_normal_points)
    normal_line_set.lines = o3d.utility.Vector2iVector(normal_lines)
    normal_line_set.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 0.0] for _ in range(num_sampled)]) # Pure Green
    
    # --- 4.7. GENERATE RED VERTICAL Z-DIRECTION VECTORS ---
    z_up_directions = np.zeros_like(sampled_vertices)
    z_up_directions[:, 2] = 1.0  
    
    z_ends = sampled_vertices + (z_up_directions * line_length)
    all_z_points = np.vstack((sampled_vertices, z_ends))
    z_lines = [[i, i + num_sampled] for i in range(num_sampled)]
    
    z_line_set = o3d.geometry.LineSet()
    z_line_set.points = o3d.utility.Vector3dVector(all_z_points)
    z_line_set.lines = o3d.utility.Vector2iVector(z_lines)
    z_line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.3, 0.0] for _ in range(num_sampled)]) # Bright Red/Orange
    
    # --- 4.8. GENERATE X-DIRECTION VECTORS (CYAN) ---
    x_directions = np.zeros_like(sampled_vertices)
    x_directions[:, 0] = 1.0  # Force X component to 1
    
    x_ends = sampled_vertices + (x_directions * line_length)
    all_x_points = np.vstack((sampled_vertices, x_ends))
    x_lines = [[i, i + num_sampled] for i in range(num_sampled)]
    
    x_line_set = o3d.geometry.LineSet()
    x_line_set.points = o3d.utility.Vector3dVector(all_x_points)
    x_line_set.lines = o3d.utility.Vector2iVector(x_lines)
    x_line_set.colors = o3d.utility.Vector3dVector([[0.0, 1.0, 1.0] for _ in range(num_sampled)]) # Cyan
    
    # --- 4.9. GENERATE Y-DIRECTION VECTORS (MAGENTA) ---
    y_directions = np.zeros_like(sampled_vertices)
    y_directions[:, 1] += 1.0  # Force Y component to 1
    
    y_ends = sampled_vertices + (y_directions * line_length)
    all_y_points = np.vstack((sampled_vertices, y_ends))
    y_lines = [[i, i + num_sampled] for i in range(num_sampled)]
    
    y_line_set = o3d.geometry.LineSet()
    y_line_set.points = o3d.utility.Vector3dVector(all_y_points)
    y_line_set.lines = o3d.utility.Vector2iVector(y_lines)
    y_line_set.colors = o3d.utility.Vector3dVector([[1.0, 0.0, 1.0] for _ in range(num_sampled)]) # Magentaz

    # Make the path
    path_points = vertices[path]#.copy()
    # path_points[:,2] += 0.2

    path_lines = [
        [i, i + 1]
        for i in range(len(path_points) - 1)
    ]

    path_line_set = o3d.geometry.LineSet()
    path_line_set.points = o3d.utility.Vector3dVector(path_points)
    path_line_set.lines = o3d.utility.Vector2iVector(path_lines)
    path_line_set.colors = o3d.utility.Vector3dVector(
        [[1,0,0] for _ in path_lines]
    )
    
    # --- 5. VISUALIZER WINDOW SETUP ---
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Terrain Steepness Map", width=1024, height=768)
    
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5, origin=[0, 0, 0])

    vis.add_geometry(mesh)
    vis.add_geometry(start_sphere)
    vis.add_geometry(goal_sphere)
    vis.add_geometry(path_line_set)
    # vis.add_geometry(normal_line_set)  # Green surface normals
    # vis.add_geometry(z_line_set)       # Red Z-up reference lines
    
    # --- COMMENTED GEOMETRIES: Uncomment these lines to show X and Y axis fields ---
    # vis.add_geometry(x_line_set)     # Cyan X-axis vectors
    # vis.add_geometry(y_line_set)     # Magenta Y-axis vectors
    
    vis.add_geometry(coord_frame)
    
    # Configure render options
    render_options = vis.get_render_option()
    if render_options is not None:
        render_options.mesh_show_wireframe = False     
        render_options.mesh_show_back_face = True     
        render_options.mesh_shade_option = o3d.visualization.MeshShadeOption.Color
        render_options.background_color = np.array([0.2, 0.4, 0.8])
        render_options.line_width = 5.0 
    
    # --- 6. PROJECT CAM TO XY PLANE ---
    vis.poll_events()
    vis.update_renderer()
    
    ctr = vis.get_view_control()
    if ctr is not None:
        ctr.set_front([0.0, 0.0, -1.0])
        ctr.set_up([0.0, 1.0, 0.0])
    
    vis.reset_view_point(True)
    vis.run()
    
    # --- 7. CAPTURE IMAGE ---
    print(f"Projecting onto XY plane and saving image to: {output_png}...")
    vis.poll_events()
    vis.update_renderer()
    vis.capture_screen_image(output_png)
    
    vis.destroy_window()
    return vertex_normals

# Run the script
v_norms = extract_and_visualize_steepness("hazardmap.obj", line_length=0.12, sample_rate=15)