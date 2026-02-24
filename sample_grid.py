import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import argparse
import sys
from pathlib import Path
from point_distribution_optimizer import optimize_point_distribution

try:
    import matplotlib.tri as mtri
except ImportError:
    mtri = None

# --- Configuration Loading ---
def load_config(config_file=None):
    """Load configuration from JSON file, with defaults."""
    defaults = {
        "layers": [
            {"name": "Jug", "N": 44, "draw_triang": False, "color": "red", "rot_limits": [0, 350], "visible": True},
            {"name": "Pinch", "N": 44, "draw_triang": False, "color": "orange", "rot_limits": [45, 135], "visible": True},
            {"name": "Sloper", "N": 44, "draw_triang": False, "color": "purple", "rot_limits": [50, 120], "visible": True},
            {"name": "Volume", "N": 44, "draw_triang": False, "color": "indigo", "rot_limits": [20, 160], "visible": True},
            {"name": "Edge", "N": 44, "draw_triang": False, "color": "green", "rot_limits": [20, 160], "visible": True},
            {"name": "Hold", "N": 48, "draw_triang": False, "color": "magenta", "rot_limits": [20, 160], "visible": True},
        ]
    }
    
    if config_file and Path(config_file).exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
            # Merge with defaults, preferring loaded values
            defaults.update(config)
        print(f"Loaded configuration from {config_file}")
    else:
        if config_file:
            print(f"Config file {config_file} not found. Using defaults.")
        else:
            print("Using default configuration")
    
    return defaults

# Disable matplotlib's default quit keybinds (e.g., 'q') before any figures are created
mpl.rcParams['keymap.quit'] = []


# --- 1. Generate Original Grid Points ---

# Global grid points (precomputed)
LEN_X_MM = 3400
LEN_Y_MM = 3200
DISTANCE_MM = 200
NUM_POINTS_X = int(LEN_X_MM / DISTANCE_MM)
NUM_POINTS_Y = int(LEN_Y_MM / DISTANCE_MM)
DISTANCE_M = DISTANCE_MM / 1000.0
X_COORDS = np.arange(0, NUM_POINTS_X * DISTANCE_M, DISTANCE_M)
Y_COORDS = np.arange(0, NUM_POINTS_Y * DISTANCE_M, DISTANCE_M)
X_GRID, Y_GRID = np.meshgrid(X_COORDS, Y_COORDS)
GRID_POINTS = np.column_stack([X_GRID.ravel(), Y_GRID.ravel()])

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

def sample_layer_mitchell(points, free_idx, N, rng, num_candidates=10):
    """
    Mitchell's Best-Candidate Algorithm for simple, uniform, random discrete point placement.
    For each of the N points, we generate `num_candidates` valid candidates from the free_idx pool.
    We choose the candidate that is furthest away from all previously selected points in this layer.
    """
    if N <= 0 or free_idx.size == 0:
        return np.array([], dtype=int)
    
    # We will need N points, or as many as are available
    N_actual = min(N, free_idx.size)
    available_mask = np.ones(free_idx.size, dtype=bool)
    free_points = points[free_idx]
    
    # Pick the first point uniformly at random
    first_idx_local = rng.integers(free_idx.size)
    chosen = [free_idx[first_idx_local]]
    available_mask[first_idx_local] = False
    
    # Track the distance from each free point to the *closest* chosen point
    # Initially, it's just the distance to the first chosen point
    min_dist_to_chosen = np.linalg.norm(free_points - free_points[first_idx_local], axis=1)

    for _ in range(1, N_actual):
        valid_indices = np.where(available_mask)[0]
        if valid_indices.size == 0:
            break
            
        # Draw random candidates
        k = min(num_candidates, valid_indices.size)
        candidates_local = rng.choice(valid_indices, size=k, replace=False)
        
        # We want the candidate that MAXIMIZES the minimum distance to existing points
        best_candidate_local = None
        best_dist = -1.0
        
        for cand in candidates_local:
            dist = min_dist_to_chosen[cand]
            if dist > best_dist:
                best_dist = dist
                best_candidate_local = cand
                
        # Commit the best candidate
        chosen.append(free_idx[best_candidate_local])
        available_mask[best_candidate_local] = False
        
        # Update min_dist_to_chosen array for all points
        new_point_pos = free_points[best_candidate_local]
        dist_to_new = np.linalg.norm(free_points - new_point_pos, axis=1)
        min_dist_to_chosen = np.minimum(min_dist_to_chosen, dist_to_new)
        
    return np.array(chosen, dtype=int)

def distribute_layers_sequential(points, N_list, rng, candidates_multiplier=2):
    """
    Distribute multiple layers sequentially using Mitchell's Best-Candidate.
    """
    total_points = len(points)
    available_idx = np.arange(total_points)
    
    selected_layers = []
    
    for layer_i, N_needed in enumerate(N_list):
        if N_needed <= 0 or available_idx.size == 0:
            selected_layers.append(np.array([], dtype=int))
            continue
            
        # Scale number of candidates: as grid gets full, we need fewer tests
        k = max(5, int(candidates_multiplier * N_needed))
        
        layer_indices = sample_layer_mitchell(points, available_idx, N_needed, rng, num_candidates=k)
        selected_layers.append(layer_indices)
        
        # Remove selected from available pool
        available_idx = np.setdiff1d(available_idx, layer_indices, assume_unique=False)
        
    return selected_layers, available_idx


# --- 4. Rotation Assignment (optimize differences on triangulation graph) ---
def angular_diff(a, b):
    """Smallest absolute angular difference (radians) in [-pi, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)

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
            # Randomly select from all candidates with max distance to avoid bias
            max_dist = dist.max()
            candidates_max = np.where(dist == max_dist)[0]
            next_idx = rng.choice(candidates_max)
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

# === MAIN FUNCTION ===
def main():
    """Main interactive analysis function."""
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Interactive grid analysis with toggleable triangulations"
    )
    parser.add_argument("-c", "--config", help="Path to JSON configuration file", default=None)
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    layers_config = config["layers"]
    
    # Extract properties from layers config
    layer_names = [layer["name"] for layer in layers_config]
    N_list = [layer["N"] for layer in layers_config]
    draw_triang = [layer["draw_triang"] for layer in layers_config]
    rot_limits = [layer["rot_limits"] for layer in layers_config]
    layer_colors = [layer["color"] for layer in layers_config]
    initial_visible = [layer.get("visible", True) for layer in layers_config]
    use_optimizer = config.get("use_optimizer", True)
    optimizer_params = config.get(
        "optimizer_params",
        {
            "max_iterations": 25,
            "point_variance_tolerance": 0.0,
            "improvement_threshold": 0.001,
            "patience": 5        }
    )
    use_custom_refinement = config.get("use_custom_refinement", False)
    
    # Print grid size summary
    print(f"Grid size: {NUM_POINTS_X} x {NUM_POINTS_Y} = {len(GRID_POINTS)} points")

    # Adjust layer counts to fill the grid evenly (approximately)
    total_points = len(GRID_POINTS)
    target_sum = total_points
    current_sum = int(np.sum(N_list))
    if current_sum != target_sum and len(N_list) > 0:
        diff = target_sum - current_sum
        step = 1 if diff > 0 else -1
        diff = abs(diff)
        for i in range(diff):
            N_list[i % len(N_list)] += step
        # Prevent negative counts
        N_list = [max(0, n) for n in N_list]
        print(f"ℹ Adjusted layer counts to fill grid: total {sum(N_list)} (grid {total_points})")

    # Initialize RNG
    rng = np.random.default_rng(42)
    
    num_layers = len(layer_names)
    
    # Sample layers using Mitchell's Best-Candidate Algorithm
    print("\nℹ Sampling layers with Mitchell's Best-Candidate...")
    
    # Sort indices strictly by size to assign largest layers first (best distribution)
    # Actually, often it's better to assign the most critical/sparse layers first.
    # We will distribute them in the order they are defined for predictability.
    
    selected_layers, free_idx = distribute_layers_sequential(
        GRID_POINTS, 
        N_list[:num_layers], 
        rng, 
        candidates_multiplier=3
    )
    
    selected_all = np.concatenate(selected_layers) if selected_layers else np.array([], dtype=int)

    # Optimizer from demo (layer-to-layer distribution improvement)
    # No highly complex refinements needed with Mitchell's Best-Candidate
    if use_optimizer and len(selected_layers) > 1:
        print("ℹ Running point distribution optimizer (demo strategy) as requested...")
        selected_layers = optimize_point_distribution(
            GRID_POINTS, selected_layers, **optimizer_params
        )
        selected_all = np.concatenate(selected_layers) if selected_layers else np.array([], dtype=int)
        free_idx = np.setdiff1d(np.arange(len(GRID_POINTS)), selected_all, assume_unique=False)
        print("✓ Optimizer done")

    # Final validation: all grid points should be assigned
    if len(selected_layers) > 0:
        assigned_all = np.concatenate(selected_layers) if selected_layers else np.array([], dtype=int)
        assigned_set = set(assigned_all.tolist())
        missing = len(GRID_POINTS) - len(assigned_set)
        if missing > 0:
            print(f"⚠ Warning: {missing} grid points still unassigned; force-assigning...")
            free_idx = np.setdiff1d(np.arange(len(GRID_POINTS)), np.array(list(assigned_set), dtype=int), assume_unique=False)
            # Force-assign remaining points evenly by layer size
            free_points = free_idx.copy()
            for i, p in enumerate(free_points):
                layer_idx = i % len(selected_layers)
                selected_layers[layer_idx] = np.append(selected_layers[layer_idx], p)
            assigned_all = np.concatenate(selected_layers)
            assigned_set = set(assigned_all.tolist())
            selected_all = assigned_all
        # Recompute free_idx after force-assign
        free_idx = np.setdiff1d(np.arange(len(GRID_POINTS)), np.array(list(assigned_set), dtype=int), assume_unique=False)
    
    print(f"✓ Sampled {len(selected_layers)} layers with {len(selected_all)} total points")
    
    # Compute rotations
    print("ℹ Computing rotations...")
    selected_points = GRID_POINTS[selected_all]
    rot_map = np.full(len(GRID_POINTS), np.nan)
    
    for layer_i, layer in enumerate(selected_layers):
        if layer.size == 0:
            continue
        
        rot_min = None
        rot_max = None
        if layer_i < len(rot_limits) and rot_limits[layer_i] is not None:
            rot_min, rot_max = rot_limits[layer_i]
        
        if layer.size > 1:
            layer_points = GRID_POINTS[layer]
            try:
                layer_tri = mtri.Triangulation(layer_points[:, 0], layer_points[:, 1])
                layer_neighbors = build_neighbors_from_tri(layer_tri, len(layer))
            except (ValueError, RuntimeError, FloatingPointError) as err:
                # Keep fallback behavior but surface expected triangulation failures.
                print(f"⚠ Triangulation failed for layer {layer_i + 1}: {err}")
                layer_neighbors = [set() for _ in range(len(layer))]
        else:
            layer_neighbors = [set() for _ in range(len(layer))]
        
        if rot_min is not None and rot_max is not None and layer.size > 0:
            layer_angles = optimize_rotations(
                layer_neighbors, rot_min, rot_max, rng,
                iterations=60, candidates_per_point=120
            )
            rot_map[layer] = layer_angles
    
    # Fill any missing rotations for layers with limits
    for layer_i, layer in enumerate(selected_layers):
        if layer.size == 0:
            continue
        if layer_i < len(rot_limits) and rot_limits[layer_i] is not None:
            rot_min, rot_max = rot_limits[layer_i]
            missing = np.isnan(rot_map[layer])
            if np.any(missing):
                rot_map[layer[missing]] = rng.uniform(
                    np.deg2rad(rot_min), np.deg2rad(rot_max), size=np.sum(missing)
                )
    # Default any remaining NaNs to 0
    rot_map[np.isnan(rot_map)] = 0.0
    rot_rad = rot_map[selected_all]
    print("✓ Rotations computed")
    
    # Compute visualization
    seg_len = DISTANCE_M * 0.8
    u = np.cos(rot_rad) * seg_len
    v = np.sin(rot_rad) * seg_len
    u_map = np.full(len(GRID_POINTS), np.nan)
    v_map = np.full(len(GRID_POINTS), np.nan)
    u_map[selected_all] = u
    v_map[selected_all] = v
    
    mst_edges = euclidean_mst(selected_points) if len(selected_points) > 1 else []
    show_graph = True
    show_mst = False
    
    # Track which layers show colored symbols (initialized from config)
    show_symbols = {i: initial_visible[i] for i in range(len(selected_layers))}
    
    # Keyboard mapping for qwertyuiop row (10 keys for up to 10 layers)
    symbol_keys = {
        'q': 0, 'w': 1, 'e': 2, 'r': 3, 't': 4,
        'y': 5, 'u': 6, 'i': 7, 'o': 8, 'p': 9
    }
    # Reverse mapping for display
    key_labels = {v: k for k, v in symbol_keys.items()}
    
    # Interactive visualization using terminal key events (no Enter required)
    print("\n" + "="*60)
    print("CONTROLS:")
    print("  Numbers (1-N): Toggle Delaunay triangulation")
    print("  Letters (q-p):  Toggle colored symbols (show/hide as gray cross)")
    print("  'x':            Quit")
    print("="*60)
    print("TRIANGULATION LAYERS:")
    for i in range(len(selected_layers)):
        status = "ON " if (i < len(draw_triang) and draw_triang[i]) else "OFF"
        name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
        symbol_key = key_labels.get(i, '?')
        symbol_status = "✓" if show_symbols.get(i, True) else "✗"
        print(f"  {i+1}. {name}: Tri={status} {symbol_key}={symbol_status}")
    print("="*60)

    fig, ax = plt.subplots(figsize=(8, 7))

    def render():
        ax.clear()

        ax.scatter(GRID_POINTS[free_idx, 0], GRID_POINTS[free_idx, 1],
                   color="lightgrey", s=2, label="Free grid points")

        for i, layer in enumerate(selected_layers):
            if layer.size == 0:
                continue
            layer_points = GRID_POINTS[layer]

            if show_symbols.get(i, True):
                color = layer_colors[i % len(layer_colors)]
                layer_name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
                ax.scatter(layer_points[:, 0], layer_points[:, 1],
                           color=color, s=10, label=f"{layer_name} (N={len(layer)})")
            else:
                ax.scatter(layer_points[:, 0], layer_points[:, 1],
                           marker='x', color='gray', s=30, linewidths=0.4, alpha=0.6,
                           label=f"Layer {i+1} (hidden)")

        for i, layer in enumerate(selected_layers):
            if layer.size == 0:
                continue
            if i < len(rot_limits) and rot_limits[i] is None:
                continue
            if not show_symbols.get(i, True):
                continue
            color = layer_colors[i % len(layer_colors)]
            layer_name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
            layer_points = GRID_POINTS[layer]
            scale_factor = 0.2 if i < len(rot_limits) and rot_limits[i] is None else 0.6
            ax.quiver(
                layer_points[:, 0] - scale_factor * u_map[layer],
                layer_points[:, 1] - scale_factor * v_map[layer],
                2*scale_factor*u_map[layer], 2*scale_factor*v_map[layer],
                angles="xy", scale_units="xy", scale=1,
                width=0.003, color=color, alpha=0.85,
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
                layer_points = GRID_POINTS[layer]
                layer_tri = mtri.Triangulation(layer_points[:, 0], layer_points[:, 1])

                # Draw only interior edges (shared by two triangles)
                edge_counts = {}
                for tri_idx in layer_tri.triangles:
                    a, b, c = tri_idx
                    edges = [(a, b), (b, c), (c, a)]
                    for u, v in edges:
                        e = tuple(sorted((u, v)))
                        edge_counts[e] = edge_counts.get(e, 0) + 1
                for (u, v), count in edge_counts.items():
                    if count == 2:
                        p1 = layer_points[u]
                        p2 = layer_points[v]
                        ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                color=color, linewidth=1.2, alpha=0.6)

                ax.plot([], [], color=color, linewidth=1.8, label=f"{layer_name} Delaunay")

        if show_mst:
            for i, j in mst_edges:
                xi, yi = selected_points[i]
                xj, yj = selected_points[j]
                ax.plot([xi, xj], [yi, yj], color="blue", linewidth=1.2, alpha=0.85, label="_nolegend_")

        ax.set_aspect("equal")
        ax.set_title("Farthest-Point Sampling on Grid with Rotations, Delaunay, and MST")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
        fig.tight_layout(rect=[0, 0, 0.82, 1])
        fig.canvas.draw_idle()

    def on_key_press(event):
        if not event.key:
            return
        key = event.key.lower()
        if key == "escape" or key == "x":
            print("✓ Exiting.")
            plt.close(fig)
            return
            
        if key in symbol_keys:
            layer_idx = symbol_keys[key]
            if layer_idx < len(selected_layers):
                show_symbols[layer_idx] = not show_symbols[layer_idx]
                status = "shown" if show_symbols[layer_idx] else "hidden"
                print(f"✓ Layer {layer_idx + 1} symbols {status}")
            render()
            return
            
        if key.isdigit():
            layer_idx = int(key) - 1
            if 0 <= layer_idx < len(selected_layers):
                draw_triang[layer_idx] = not draw_triang[layer_idx]
                print(f"✓ Toggled triangulation for layer {key}")
                render()
            return

    fig.canvas.mpl_connect('key_press_event', on_key_press)
    render()
    plt.show()

if __name__ == "__main__":
    main()
