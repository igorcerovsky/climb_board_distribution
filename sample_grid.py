import numpy as np
import matplotlib.pyplot as plt

# --- 1. Generate Original Grid Points ---
np.random.seed(42)

len_x_mm = 3600
len_y_mm = 3200
distance_mm = 200

num_points_x = int(len_x_mm / distance_mm)
num_points_y = int(len_y_mm / distance_mm)

distance_m = distance_mm / 1000.0
x_coords = np.arange(0, num_points_x * distance_m, distance_m)
y_coords = np.arange(0, num_points_y * distance_m, distance_m)

X, Y = np.meshgrid(x_coords, y_coords)
grid_points = np.column_stack([X.ravel(), Y.ravel()])

# --- 2. Farthest Point Sampling (maximize min distance) ---
def farthest_point_sampling(points, N, seed=42):
    rng = np.random.default_rng(seed)

    # start with one random point
    first_idx = rng.integers(len(points))
    selected = [first_idx]

    # track distance to nearest selected point
    dist = np.full(len(points), np.inf)

    for _ in range(1, N):
        last = points[selected[-1]]
        d = np.linalg.norm(points - last, axis=1)
        dist = np.minimum(dist, d)

        # pick point with maximum distance to nearest selected
        next_idx = np.argmax(dist)
        selected.append(next_idx)

    selected = np.array(selected)
    free = np.setdiff1d(np.arange(len(points)), selected)
    return selected, free

# --- 3. Sample Multiple Layers of Points ---
# Define how many points to add per iteration (edit this list)
N_list = [32, 32, 44]  # e.g., first iteration 32, second iteration 32
# Define colors per iteration (must be at least as long as N_list)
layer_names = ["Jug", "Bid Edge Rotated", "Pinch Rotated"]  # names for legend
draw_triang = [False, False, True]  # whether to draw Delaunay triangulation for each layer
rot_limits = [(0, 350), (45, 135), (45, 135)]  # rotation limits (degrees) for each layer
layer_colors = ["red", "green", "blue", "purple", "orange"]
rng = np.random.default_rng(42)

def sample_additional(points, free_idx, existing_selected_idx, N, rng,
                      respect_existing=True, top_k=5):
    if N <= 0 or free_idx.size == 0:
        return np.array([], dtype=int)

    free_points = points[free_idx]
    if respect_existing and existing_selected_idx.size > 0:
        selected_points = points[existing_selected_idx]
        dist_to_selected = np.min(
            np.linalg.norm(free_points[:, None, :] - selected_points[None, :, :], axis=2),
            axis=1
        )
    else:
        dist_to_selected = np.full(len(free_idx), np.inf)

    chosen = []
    available = np.ones(len(free_idx), dtype=bool)

    for _ in range(min(N, len(free_idx))):
        candidates = np.where(available)[0]
        if candidates.size == 0:
            break
        cand_scores = dist_to_selected[candidates]
        k = min(top_k, candidates.size)
        top_idx = np.argpartition(cand_scores, -k)[-k:]
        idx_local = candidates[rng.choice(top_idx)]
        if dist_to_selected[idx_local] < 0:
            break
        chosen.append(free_idx[idx_local])
        available[idx_local] = False

        # Update distances using the newly chosen point
        new_point = points[free_idx[idx_local]]
        d = np.linalg.norm(free_points - new_point, axis=1)
        dist_to_selected = np.minimum(dist_to_selected, d)

    return np.array(chosen, dtype=int)

selected_layers = []
free_idx = np.arange(len(grid_points))
selected_all = np.array([], dtype=int)

for layer_i, n in enumerate(N_list):
    # For later layers, prioritize uniformity within remaining free points
    respect_existing = (layer_i == 0)
    new_sel = sample_additional(
        grid_points,
        free_idx,
        selected_all,
        n,
        rng,
        respect_existing=respect_existing,
        top_k=8
    )
    selected_layers.append(new_sel)
    if new_sel.size > 0:
        selected_all = np.concatenate([selected_all, new_sel])
        free_idx = np.setdiff1d(free_idx, new_sel, assume_unique=False)

# --- 4. Rotation Assignment (optimize differences on triangulation graph) ---

def angular_diff(a, b):
    """Smallest absolute angular difference (radians) in [-pi, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)

# --- 5. Build Planar Graphs (Delaunay + MST) ---
selected_points = grid_points[selected_all]

# Toggle graph visibility
show_graph = True
show_mst = False

# Delaunay triangulation (planar, no intersecting edges)
tri = None
mtri = None
try:
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(selected_points[:, 0], selected_points[:, 1])
except Exception:
    tri = None
    mtri = None

def build_neighbors_from_tri(tri, n):
    neighbors = [set() for _ in range(n)]
    if tri is None:
        return neighbors
    for a, b, c in tri.triangles:
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])
    return neighbors

def optimize_rotations(neighbors, rot_min_deg, rot_max_deg, rng,
                       iterations=60, candidates_per_point=80):
    rot_min = np.deg2rad(rot_min_deg)
    rot_max = np.deg2rad(rot_max_deg)
    n = len(neighbors)
    angles = initialize_spread_angles(rng, rot_min, rot_max, n)

    for _ in range(iterations):
        order = rng.permutation(n)
        for i in order:
            nbrs = list(neighbors[i])
            if not nbrs:
                continue
            nbr_angles = angles[nbrs]
            candidates = rng.uniform(rot_min, rot_max, size=candidates_per_point)
            best_angle = angles[i]
            best_score = -1.0
            for a in candidates:
                diffs = angular_diff(a, nbr_angles)
                score = diffs.min()
                if score > best_score:
                    best_score = score
                    best_angle = a
            angles[i] = best_angle
    return angles

def initialize_spread_angles(rng, rot_min, rot_max, n):
    if n == 0:
        angles = np.array([])
    else:
        # Farthest-point sampling on linear range to get well-spread initial angles
        num_candidates = max(1000, n * 5)
        candidates = np.linspace(rot_min, rot_max, num_candidates)
        selected_idxs = []
        first_idx = int(rng.integers(num_candidates))
        selected_idxs.append(first_idx)

        dist = np.full(num_candidates, np.inf)
        for _ in range(1, n):
            last = candidates[selected_idxs[-1]]
            # Use regular 1D distance for linear sampling (not circular/angular)
            d = np.abs(candidates - last)
            dist = np.minimum(dist, d)
            next_idx = int(np.argmax(dist))
            selected_idxs.append(next_idx)

        angles = candidates[np.array(selected_idxs)]
    return angles

def euclidean_mst(points):
    """Prim's algorithm for Euclidean MST. Returns list of edges (i, j)."""
    n = len(points)
    in_tree = np.zeros(n, dtype=bool)
    min_dist = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=int)

    min_dist[0] = 0.0
    edges = []

    for _ in range(n):
        u = np.argmin(np.where(in_tree, np.inf, min_dist))
        in_tree[u] = True
        if parent[u] != -1:
            edges.append((parent[u], u))

        d = np.linalg.norm(points - points[u], axis=1)
        update = (~in_tree) & (d < min_dist)
        min_dist[update] = d[update]
        parent[update] = u

    return edges

mst_edges = euclidean_mst(selected_points) if len(selected_points) > 1 else []

# --- 6. Optimize rotations based on triangulation neighbors (per-layer) ---
rot_rad = np.full(len(selected_all), np.nan)

for layer_i, layer in enumerate(selected_layers):
    if layer.size == 0:
        continue
    
    # Get rotation limits for this layer
    if layer_i < len(rot_limits):
        rot_min, rot_max = rot_limits[layer_i]
    
    # Build neighbors graph for this layer only
    if layer.size > 1:
        layer_points = grid_points[layer]
        try:
            layer_tri = mtri.Triangulation(layer_points[:, 0], layer_points[:, 1])
            layer_neighbors = build_neighbors_from_tri(layer_tri, len(layer))
        except:
            layer_neighbors = [set() for _ in range(len(layer))]
    else:
        layer_neighbors = [set() for _ in range(len(layer))]
    
    # Optimize rotations for this layer
    layer_angles = optimize_rotations(
        layer_neighbors,
        rot_min,
        rot_max,
        rng,
        iterations=120,
        candidates_per_point=100
    ) if layer.size > 0 else np.array([])
    
    # Map back to global indices
    global_indices = np.where(np.isin(selected_all, layer))[0]
    rot_rad[global_indices] = layer_angles

# Small line segment direction for visualization (both sides of point)
seg_len = distance_m * 0.8
u = np.cos(rot_rad) * seg_len
v = np.sin(rot_rad) * seg_len

# Map global point index -> rotation vector
u_map = np.full(len(grid_points), np.nan)
v_map = np.full(len(grid_points), np.nan)
u_map[selected_all] = u
v_map[selected_all] = v

plt.figure(figsize=(8, 7))
plt.scatter(grid_points[free_idx, 0], grid_points[free_idx, 1],
            color="lightgrey", s=2, label="Free grid points")
for i, layer in enumerate(selected_layers):
    if layer.size == 0:
        continue
    color = layer_colors[i % len(layer_colors)]
    layer_name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
    plt.scatter(grid_points[layer, 0], grid_points[layer, 1],
                color=color, s=10, label=f"{layer_name} (N={len(layer)})")
for i, layer in enumerate(selected_layers):
    if layer.size == 0:
        continue
    color = layer_colors[i % len(layer_colors)]
    layer_name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
    layer_points = grid_points[layer]
    plt.quiver(
        layer_points[:, 0] - u_map[layer],
        layer_points[:, 1] - v_map[layer],
        2 * u_map[layer], 2 * v_map[layer],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.005,
        color=color,
        alpha=0.85,
        label=f"{layer_name} Rotation"
    )

if show_graph and mtri is not None:
    for i, layer in enumerate(selected_layers):
        if layer.size < 3:
            continue
        if i < len(draw_triang) and not draw_triang[i]:
            continue
        color = layer_colors[i % len(layer_colors)]
        layer_name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
        layer_points = grid_points[layer]
        layer_tri = mtri.Triangulation(layer_points[:, 0], layer_points[:, 1])
        # Draw triangulation with transparency
        for tri_idx in layer_tri.triangles:
            triangle = layer_points[tri_idx]
            triangle_closed = np.vstack([triangle, triangle[0]])
            plt.plot(triangle_closed[:, 0], triangle_closed[:, 1], 
                    color=color, linewidth=0.9, alpha=0.1)
        # Add label once per layer
        plt.plot([], [], color=color, linewidth=0.9, label=f"{layer_name} Delaunay")

if show_mst:
    for i, j in mst_edges:
        xi, yi = selected_points[i]
        xj, yj = selected_points[j]
        plt.plot([xi, xj], [yi, yj], color="blue", linewidth=1.2, alpha=0.85, label="_nolegend_")

plt.gca().set_aspect("equal")
plt.title("Farthest-Point Sampling on Grid with Rotations, Delaunay, and MST")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
plt.tight_layout(rect=[0, 0, 0.82, 1])
plt.show()
