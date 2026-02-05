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
N_list = [32, 32]  # e.g., first iteration 32, second iteration 32
# Define colors per iteration (must be at least as long as N_list)
layer_colors = ["red", "green", "orange", "blue", "purple"]

def sample_additional(points, free_idx, existing_selected_idx, N):
    if N <= 0 or free_idx.size == 0:
        return np.array([], dtype=int)

    free_points = points[free_idx]
    if existing_selected_idx.size > 0:
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
        idx_local = np.argmax(np.where(available, dist_to_selected, -1.0))
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

for n in N_list:
    new_sel = sample_additional(grid_points, free_idx, selected_all, n)
    selected_layers.append(new_sel)
    if new_sel.size > 0:
        selected_all = np.concatenate([selected_all, new_sel])
        free_idx = np.setdiff1d(free_idx, new_sel, assume_unique=False)

# --- 4. Rotation Assignment (optimize differences on triangulation graph) ---
rot_min_deg = -45
rot_max_deg = 45
rng = np.random.default_rng(42)

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
                       iterations=60, candidates_per_point=50):
    rot_min = np.deg2rad(rot_min_deg)
    rot_max = np.deg2rad(rot_max_deg)
    n = len(neighbors)
    angles = rng.uniform(rot_min, rot_max, size=n)

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

# --- 6. Optimize rotations based on triangulation neighbors ---
neighbors = build_neighbors_from_tri(tri, len(selected_points))
rot_rad = optimize_rotations(
    neighbors,
    rot_min_deg,
    rot_max_deg,
    rng,
    iterations=60,
    candidates_per_point=50
) if len(selected_points) > 0 else np.array([])

# Small line segment direction for visualization (both sides of point)
seg_len = distance_m * 0.35
u = np.cos(rot_rad) * seg_len
v = np.sin(rot_rad) * seg_len

# Map global point index -> rotation vector
u_map = np.full(len(grid_points), np.nan)
v_map = np.full(len(grid_points), np.nan)
u_map[selected_all] = u
v_map[selected_all] = v

plt.figure(figsize=(8, 7))
plt.scatter(grid_points[free_idx, 0], grid_points[free_idx, 1],
            color="lightgrey", s=20, label="Free grid points")
for i, layer in enumerate(selected_layers):
    if layer.size == 0:
        continue
    color = layer_colors[i % len(layer_colors)]
    plt.scatter(grid_points[layer, 0], grid_points[layer, 1],
                color=color, s=40, label=f"Selected layer {i+1} (N={len(layer)})")
for i, layer in enumerate(selected_layers):
    if layer.size == 0:
        continue
    color = layer_colors[i % len(layer_colors)]
    layer_points = grid_points[layer]
    plt.quiver(
        layer_points[:, 0] - u_map[layer],
        layer_points[:, 1] - v_map[layer],
        2 * u_map[layer], 2 * v_map[layer],
        angles="xy",
        scale_units="xy",
        scale=1,
        width=0.004,
        color=color,
        alpha=0.85,
        label=f"Rotation {i+1}"
    )

if show_graph and mtri is not None:
    for i, layer in enumerate(selected_layers):
        if layer.size < 3:
            continue
        color = layer_colors[i % len(layer_colors)]
        layer_points = grid_points[layer]
        layer_tri = mtri.Triangulation(layer_points[:, 0], layer_points[:, 1])
        plt.triplot(layer_tri, color=color, linewidth=0.9, alpha=0.7, label=f"Delaunay {i+1}")

if show_mst:
    for i, j in mst_edges:
        xi, yi = selected_points[i]
        xj, yj = selected_points[j]
        plt.plot([xi, xj], [yi, yj], color="blue", linewidth=1.2, alpha=0.85, label="_nolegend_")

plt.gca().set_aspect("equal")
plt.title("Farthest-Point Sampling on Grid with Rotations, Delaunay, and MST")
plt.xlabel("X (m)")
plt.ylabel("Y (m)")
plt.legend()
plt.tight_layout()
plt.show()
