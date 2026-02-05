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

# --- 3. Sample and Plot ---
N = 32
selected_idx, free_idx = farthest_point_sampling(grid_points, N, seed=42)

# --- 4. Rotation Assignment (optimize differences on triangulation graph) ---
rot_min_deg = -45
rot_max_deg = 45
rng = np.random.default_rng(42)

def angular_diff(a, b):
    """Smallest absolute angular difference (radians) in [-pi, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)

# --- 5. Build Planar Graphs (Delaunay + MST) ---
selected_points = grid_points[selected_idx]

# Toggle graph visibility
show_graph = True
show_mst = False

# Delaunay triangulation (planar, no intersecting edges)
tri = None
try:
    import matplotlib.tri as mtri
    tri = mtri.Triangulation(selected_points[:, 0], selected_points[:, 1])
except Exception:
    tri = None

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

mst_edges = euclidean_mst(selected_points)

# --- 6. Optimize rotations based on triangulation neighbors ---
neighbors = build_neighbors_from_tri(tri, len(selected_points))
rot_rad = optimize_rotations(
    neighbors,
    rot_min_deg,
    rot_max_deg,
    rng,
    iterations=60,
    candidates_per_point=50
)

# Small line segment direction for visualization (both sides of point)
seg_len = distance_m * 0.35
u = np.cos(rot_rad) * seg_len
v = np.sin(rot_rad) * seg_len

plt.figure(figsize=(8, 7))
plt.scatter(grid_points[free_idx, 0], grid_points[free_idx, 1],
            color="lightgrey", s=20, label="Free grid points")
plt.scatter(grid_points[selected_idx, 0], grid_points[selected_idx, 1],
            color="red", s=40, label=f"Selected points (N={N})")
plt.quiver(
    grid_points[selected_idx, 0] - u,
    grid_points[selected_idx, 1] - v,
    2 * u, 2 * v,
    angles="xy",
    scale_units="xy",
    scale=1,
    width=0.004,
    color="red",
    alpha=0.85,
    label="Rotation"
)

if show_graph and tri is not None:
    plt.triplot(tri, color="black", linewidth=0.9, alpha=0.7, label="Delaunay")

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
