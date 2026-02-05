import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
import sys
from pathlib import Path

# --- Configuration Loading ---
def load_config(config_file=None):
    """Load configuration from JSON file, with defaults."""
    defaults = {
        "layer_names": ["Jug", "Bid Edge Rotated", "Pinch Rotated", "Sloper", "Layer 5", "Layer 6", "Layer 7", "Layer 8", "Layer 9"],
        "rot_limits": [[0, 350], [45, 135], [45, 135], [0, 180], [90, 90], [90, 90], [90, 90], [90, 90], [90, 90]],
        "N_list": [32, 32, 44, 32, 32, 32, 32, 32, 32],
        "draw_triang": [False, False, True, False, False, False, False, False, False],
        "layer_colors": ["red", "green", "blue", "purple", "orange", "cyan", "magenta", "brown", "pink"]
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

# --- 1. Generate Original Grid Points ---
np.random.seed(42)

# Global grid points (precomputed)
LEN_X_MM = 3600
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
    layer_names = config["layer_names"]
    N_list = config["N_list"]
    draw_triang = list(config["draw_triang"])
    rot_limits = config["rot_limits"]
    layer_colors = config["layer_colors"]
    
    # Initialize RNG
    rng = np.random.default_rng(42)
    
    # Sample layers (only for defined layer_names)
    print("\nℹ Sampling layers...")
    selected_layers = []
    free_idx = np.arange(len(GRID_POINTS))
    selected_all = np.array([], dtype=int)
    
    num_layers = len(layer_names)
    
    for layer_i in range(num_layers):
        if layer_i >= len(N_list):
            break
        n = N_list[layer_i]
        respect_existing = (layer_i == 0)
        new_sel = sample_additional(
            GRID_POINTS, free_idx, selected_all, n, rng,
            respect_existing=respect_existing, top_k=8
        )
        selected_layers.append(new_sel)
        if new_sel.size > 0:
            selected_all = np.concatenate([selected_all, new_sel])
            free_idx = np.setdiff1d(free_idx, new_sel, assume_unique=False)
    
    print(f"✓ Sampled {len(selected_layers)} layers with {len(selected_all)} total points")
    
    # Compute rotations
    print("ℹ Computing rotations...")
    selected_points = GRID_POINTS[selected_all]
    rot_rad = np.full(len(selected_all), np.nan)
    
    try:
        import matplotlib.tri as mtri
    except ImportError:
        mtri = None
    
    for layer_i, layer in enumerate(selected_layers):
        if layer.size == 0:
            continue
        
        if layer_i < len(rot_limits):
            rot_min, rot_max = rot_limits[layer_i]
        else:
            rot_min, rot_max = -45, 45
        
        if layer.size > 1:
            layer_points = GRID_POINTS[layer]
            try:
                layer_tri = mtri.Triangulation(layer_points[:, 0], layer_points[:, 1])
                layer_neighbors = build_neighbors_from_tri(layer_tri, len(layer))
            except:
                layer_neighbors = [set() for _ in range(len(layer))]
        else:
            layer_neighbors = [set() for _ in range(len(layer))]
        
        layer_angles = optimize_rotations(
            layer_neighbors, rot_min, rot_max, rng,
            iterations=120, candidates_per_point=100
        ) if layer.size > 0 else np.array([])
        
        global_indices = np.where(np.isin(selected_all, layer))[0]
        rot_rad[global_indices] = layer_angles
    
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
    
    # Track which layers show colored symbols (default: all True)
    show_symbols = {i: True for i in range(len(selected_layers))}
    
    # Keyboard mapping for qwertyuiop row (10 keys for up to 10 layers)
    symbol_keys = {
        'q': 0, 'w': 1, 'e': 2, 'r': 3, 't': 4,
        'y': 5, 'u': 6, 'i': 7, 'o': 8, 'p': 9
    }
    # Reverse mapping for display
    key_labels = {v: k for k, v in symbol_keys.items()}
    
    # Interactive loop
    while True:
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  Numbers (1-N): Toggle Delaunay triangulation")
        print("  Letters (q-p):  Toggle colored symbols (show/hide as gray cross)")
        print("  'q':            Quit")
        print("="*60)
        print("TRIANGULATION LAYERS:")
        for i in range(len(selected_layers)):
            status = "ON " if (i < len(draw_triang) and draw_triang[i]) else "OFF"
            name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
            symbol_key = key_labels.get(i, '?')
            symbol_status = "✓" if show_symbols.get(i, True) else "✗"
            print(f"  {i+1}. {name}: Tri={status} {symbol_key}={symbol_status}")
        print("="*60)
        
        # Create visualization
        plt.figure(figsize=(8, 7))
        
        # Plot free points
        plt.scatter(GRID_POINTS[free_idx, 0], GRID_POINTS[free_idx, 1],
                    color="lightgrey", s=2, label="Free grid points")
        
        # Plot layers
        for i, layer in enumerate(selected_layers):
            if layer.size == 0:
                continue
            layer_points = GRID_POINTS[layer]
            
            if show_symbols.get(i, True):
                # Draw colored scatter points
                color = layer_colors[i % len(layer_colors)]
                layer_name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
                plt.scatter(layer_points[:, 0], layer_points[:, 1],
                            color=color, s=10, label=f"{layer_name} (N={len(layer)})")
            else:
                # Draw small gray crosses instead
                plt.scatter(layer_points[:, 0], layer_points[:, 1],
                            marker='x', color='gray', s=30, linewidths=0.4, alpha=0.6, label=f"Layer {i+1} (hidden)")
        
        # Plot rotations
        for i, layer in enumerate(selected_layers):
            if layer.size == 0:
                continue
            # Skip rotation arrows if limits are [90, 90]
            if i < len(rot_limits) and rot_limits[i] == [90, 90]:
                continue
            # Skip rotation arrows if layer symbols are hidden
            if not show_symbols.get(i, True):
                continue
            color = layer_colors[i % len(layer_colors)]
            layer_name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
            layer_points = GRID_POINTS[layer]
            scale_factor = 0.2 if i < len(rot_limits) and rot_limits[i] == [90, 90] else 0.6
            plt.quiver(
                layer_points[:, 0] - scale_factor * u_map[layer],
                layer_points[:, 1] - scale_factor * v_map[layer],
                2*scale_factor*u_map[layer], 2*scale_factor*v_map[layer],
                angles="xy", scale_units="xy", scale=1,
                width=0.003, color=color, alpha=0.85,
                label=f"{layer_name} Rotation"
            )
        
        # Plot triangulations
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
                
                for tri_idx in layer_tri.triangles:
                    triangle = layer_points[tri_idx]
                    triangle_closed = np.vstack([triangle, triangle[0]])
                    plt.plot(triangle_closed[:, 0], triangle_closed[:, 1], 
                            color=color, linewidth=0.2, alpha=0.4)
                
                plt.plot([], [], color=color, linewidth=0.9, label=f"{layer_name} Delaunay")
        
        # Plot MST
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
        plt.show(block=False)
        
        # Get user input
        user_input = input(f"\nEnter layer number to toggle triangulation (1-{len(selected_layers)}),\n"
                          f"letter (q-p) to toggle symbol visibility, or 'quit' to exit: ").strip().lower()
        
        if user_input in ['quit', 'exit', 'x']:
            print("✓ Exiting.")
            plt.close('all')
            break
        
        # Check if input is a symbol key (q-p for layers 0-9)
        if user_input in symbol_keys:
            layer_idx = symbol_keys[user_input]
            if layer_idx < len(selected_layers):
                show_symbols[layer_idx] = not show_symbols[layer_idx]
                status = "shown" if show_symbols[layer_idx] else "hidden"
                print(f"✓ Layer {layer_idx + 1} symbols {status}")
                plt.close('all')
            else:
                print(f"✗ No layer assigned to key '{user_input}'")
            continue
        
        # Check if input is a triangulation toggle (numeric)
        try:
            layer_idx = int(user_input) - 1
            if 0 <= layer_idx < len(selected_layers):
                draw_triang[layer_idx] = not draw_triang[layer_idx]
                print(f"✓ Toggled triangulation for layer {user_input}")
                plt.close('all')
            else:
                print(f"✗ Invalid layer number. Please enter 1-{len(selected_layers)}")
        except ValueError:
            print("✗ Invalid input. Please enter a number (1-N), letter (q-p), or 'quit'.")

if __name__ == "__main__":
    main()
