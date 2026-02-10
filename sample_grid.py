import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tty
import termios
import select
import json
import argparse
import sys
from pathlib import Path
from point_distribution_optimizer import optimize_point_distribution

# --- Configuration Loading ---
def load_config(config_file=None):
    """Load configuration from JSON file, with defaults."""
    defaults = {
        "layers": [
            {"name": "Jug", "N": 44, "draw_triang": False, "color": "red", "rot_limits": [0, 350], "visible": True},
            {"name": "Pinch", "N": 24, "draw_triang": False, "color": "blue", "rot_limits": [45, 135], "visible": True},
            {"name": "PinchBig", "N": 24, "draw_triang": False, "color": "orange", "rot_limits": [45, 135], "visible": True},
            {"name": "Sloper (round)", "N": 44, "draw_triang": False, "color": "purple", "rot_limits": [50, 120], "visible": True},
            {"name": "Sloper (flat)", "N": 44, "draw_triang": False, "color": "purple", "rot_limits": [50, 120], "visible": True},
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
np.random.seed(1)

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

def _feasible_min_per(n, desired=2):
    if n <= 0:
        return 0
    max_per_row = n // max(NUM_POINTS_Y, 1)
    max_per_col = n // max(NUM_POINTS_X, 1)
    return min(desired, max_per_row, max_per_col)

def _layer_score_from_dist(dist_mat, idxs, penalty_weight=0.0, min_per=2):
    if idxs.size < 2:
        return 0.0
    d = dist_mat[np.ix_(idxs, idxs)].copy()
    np.fill_diagonal(d, np.inf)
    nn = np.min(d, axis=1)
    base = float(nn.min() + 0.2 * nn.mean())
    if penalty_weight > 0.0 and min_per > 0:
        base -= penalty_weight * rowcol_penalty(idxs, min_per=min_per)
    return base

def _global_score(dist_mat, layers, cross_weight=0.2, penalty_weight=0.0, min_per=2):
    if not layers:
        return 0.0
    scores = []
    for idxs in layers:
        layer_min = _feasible_min_per(len(idxs), desired=min_per)
        scores.append(_layer_score_from_dist(dist_mat, idxs, penalty_weight=penalty_weight, min_per=layer_min))
    intra = float(np.mean(scores))
    # Cross-layer repulsion: encourage layers to avoid each other's points
    if len(layers) > 1:
        cross_mins = []
        for i in range(len(layers)):
            for j in range(i + 1, len(layers)):
                a = layers[i]
                b = layers[j]
                if a.size == 0 or b.size == 0:
                    continue
                d = dist_mat[np.ix_(a, b)]
                cross_mins.append(np.min(d))
        if cross_mins:
            inter = float(np.mean(cross_mins))
        else:
            inter = 0.0
    else:
        inter = 0.0
    return intra + cross_weight * inter

def _layer_score(points_or_dist, idxs, penalty_weight=0.0, min_per=2):
    # Backward-compatible helper
    if isinstance(points_or_dist, np.ndarray) and points_or_dist.ndim == 2:
        layer_min = _feasible_min_per(len(idxs), desired=min_per)
        return _layer_score_from_dist(points_or_dist, idxs, penalty_weight=penalty_weight, min_per=layer_min)
    dist_mat = np.linalg.norm(points_or_dist[:, None, :] - points_or_dist[None, :, :], axis=2)
    layer_min = _feasible_min_per(len(idxs), desired=min_per)
    return _layer_score_from_dist(dist_mat, idxs, penalty_weight=penalty_weight, min_per=layer_min)

def _greedy_farthest(points, free_idx, N, rng, top_k=8):
    if N <= 0 or free_idx.size == 0:
        return np.array([], dtype=int)
    free_points = points[free_idx]
    chosen = []
    available = np.ones(len(free_idx), dtype=bool)
    dist_to_selected = np.full(len(free_idx), np.inf)

    # random start
    first = rng.integers(len(free_idx))
    chosen.append(free_idx[first])
    available[first] = False
    dist_to_selected = np.minimum(
        dist_to_selected,
        np.linalg.norm(free_points - free_points[first], axis=1)
    )

    for _ in range(1, min(N, len(free_idx))):
        candidates = np.where(available)[0]
        if candidates.size == 0:
            break
        cand_scores = dist_to_selected[candidates]
        k = min(top_k, candidates.size)
        top_idx = np.argpartition(cand_scores, -k)[-k:]
        idx_local = candidates[rng.choice(top_idx)]
        chosen.append(free_idx[idx_local])
        available[idx_local] = False

        new_point = free_points[idx_local]
        d = np.linalg.norm(free_points - new_point, axis=1)
        dist_to_selected = np.minimum(dist_to_selected, d)

    return np.array(chosen, dtype=int)

def _poisson_disk_select(points, free_idx, N, rng,
                         tries=30, iters=20):
    """Maximize minimum distance via discrete Poisson-disk selection with binary search."""
    if N <= 0 or free_idx.size == 0:
        return np.array([], dtype=int)

    free_points = points[free_idx]
    dist = np.linalg.norm(free_points[:, None, :] - free_points[None, :, :], axis=2)
    max_dist = dist.max() if dist.size else 0.0

    best_sel = None
    best_r = -1.0

    def attempt(r):
        best_local = []
        for _ in range(tries):
            order = rng.permutation(len(free_idx))
            blocked = np.zeros(len(free_idx), dtype=bool)
            selected = []
            for i in order:
                if blocked[i]:
                    continue
                selected.append(i)
                if len(selected) >= N:
                    return True, selected
                blocked |= (dist[i] < r)
            if len(selected) > len(best_local):
                best_local = selected
        return False, best_local

    lo, hi = 0.0, max_dist
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        ok, sel = attempt(mid)
        if ok:
            lo = mid
            best_r = mid
            best_sel = sel
        else:
            hi = mid

    if best_sel is None or len(best_sel) < N:
        # fallback to greedy farthest if binary search fails to reach N
        return _greedy_farthest(points, free_idx, N, rng, top_k=10)

    return free_idx[np.array(best_sel, dtype=int)]

def _pam_select(points, free_idx, N, rng, iters=25):
    """PAM (k-medoids) on remaining free points."""
    if N <= 0 or free_idx.size == 0:
        return np.array([], dtype=int)
    free_points = points[free_idx]
    m = len(free_idx)
    if N >= m:
        return free_idx.copy()

    # Precompute distances
    dist = np.linalg.norm(free_points[:, None, :] - free_points[None, :, :], axis=2)

    # Initialize medoids with farthest-point greedy
    medoids = _greedy_farthest(points, free_idx, N, rng, top_k=10)
    medoid_idx = np.searchsorted(free_idx, medoids)

    for _ in range(iters):
        # Assign each point to nearest medoid
        d_to_medoids = dist[:, medoid_idx]
        assignment = np.argmin(d_to_medoids, axis=1)

        new_medoids = medoid_idx.copy()
        for k in range(N):
            cluster = np.where(assignment == k)[0]
            if cluster.size == 0:
                continue
            # Choose point minimizing sum of distances within cluster
            sub = dist[np.ix_(cluster, cluster)]
            costs = sub.sum(axis=1)
            new_medoids[k] = cluster[np.argmin(costs)]

        if np.array_equal(new_medoids, medoid_idx):
            break
        medoid_idx = new_medoids

    return free_idx[medoid_idx]

def rowcol_penalty(idxs, min_per=2):
    """Soft penalty for rows/cols with fewer than min_per points."""
    if idxs.size == 0:
        return 0.0
    row_counts = np.zeros(NUM_POINTS_Y, dtype=int)
    col_counts = np.zeros(NUM_POINTS_X, dtype=int)
    for idx in idxs:
        r = idx // NUM_POINTS_X
        c = idx % NUM_POINTS_X
        row_counts[r] += 1
        col_counts[c] += 1
    deficit_rows = np.clip(min_per - row_counts, 0, None)
    deficit_cols = np.clip(min_per - col_counts, 0, None)
    return float(deficit_rows.sum() + deficit_cols.sum())

def _swap_improve(points, free_idx, idxs, rng, steps=200, candidate_pool=80,
                  penalty_weight=0.0, min_per=2):
    if idxs.size < 2:
        return idxs
    current = idxs.copy()
    free_set = set(free_idx.tolist())
    current_set = set(current.tolist())
    free_set -= current_set
    free_list = np.array(list(free_set), dtype=int)
    if free_list.size == 0:
        return current

    dist_mat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    best_score = _layer_score_from_dist(dist_mat, current, penalty_weight=penalty_weight, min_per=min_per)
    for _ in range(steps):
        # pick a point to replace
        out_idx = rng.integers(len(current))
        out_point = current[out_idx]
        # sample candidates to swap in
        k = min(candidate_pool, free_list.size)
        cand = rng.choice(free_list, size=k, replace=False)
        improved = False
        for in_point in cand:
            trial = current.copy()
            trial[out_idx] = in_point
            score = _layer_score_from_dist(dist_mat, trial, penalty_weight=penalty_weight, min_per=min_per)
            if score > best_score:
                best_score = score
                current = trial
                # update free_list (swap)
                free_list = np.array([p for p in free_list if p != in_point] + [out_point], dtype=int)
                improved = True
                break
        if not improved:
            continue
    return current

def sample_layer_best(points, free_idx, N, rng,
                      trials=12, top_k=8, swap_steps=200,
                      use_poisson=True, use_pam=False):
    if N <= 0 or free_idx.size == 0:
        return np.array([], dtype=int)
    best = None
    best_score = -1.0
    dist_mat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    for _ in range(trials):
        if use_pam:
            idxs = _pam_select(points, free_idx, N, rng, iters=20)
        elif use_poisson:
            idxs = _poisson_disk_select(points, free_idx, N, rng, tries=25, iters=18)
        else:
            idxs = _greedy_farthest(points, free_idx, N, rng, top_k=top_k)
        layer_min = _feasible_min_per(len(idxs), desired=2)
        idxs = _swap_improve(points, free_idx, idxs, rng, steps=swap_steps,
                             penalty_weight=0.15, min_per=layer_min)
        score = _layer_score_from_dist(dist_mat, idxs, penalty_weight=0.15, min_per=layer_min)
        if score > best_score:
            best_score = score
            best = idxs
    if best is None or best.size == 0:
        # Fallback: random sample to avoid empty layer
        if free_idx.size >= N:
            return rng.choice(free_idx, size=N, replace=False)
        return free_idx.copy()
    if best.size < N and free_idx.size >= N:
        # Ensure full layer size if possible
        return rng.choice(free_idx, size=N, replace=False)
    return best

def sample_layers_global_greedy(points, N_list, rng):
    """Global greedy sampler that allocates points to all layers simultaneously."""
    num_layers = len(N_list)
    total_needed = int(np.sum(N_list))
    if total_needed == 0:
        return [np.array([], dtype=int) for _ in range(num_layers)], np.arange(len(points))

    total_points = len(points)
    if total_needed > total_points:
        raise ValueError("Requested more points than available in grid.")

    # Precompute distance matrix (small grid sizes make this ok)
    dist_mat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    # Expected spacing per layer for normalization
    x_min, x_max = points[:, 0].min(), points[:, 0].max()
    y_min, y_max = points[:, 1].min(), points[:, 1].max()
    area = (x_max - x_min) * (y_max - y_min)
    target_spacing = [
        np.sqrt(area / max(n, 1)) if n > 0 else 1.0
        for n in N_list
    ]

    available_mask = np.ones(total_points, dtype=bool)
    selected_layers = [np.array([], dtype=int) for _ in range(num_layers)]
    remaining = N_list.copy()

    # Seed each non-empty layer with one random point to avoid degeneracy
    for i in range(num_layers):
        if remaining[i] > 0:
            candidates = np.where(available_mask)[0]
            if candidates.size == 0:
                break
            pick = rng.choice(candidates)
            selected_layers[i] = np.array([pick], dtype=int)
            available_mask[pick] = False
            remaining[i] -= 1

    # Initialize min-distance arrays for each layer
    min_dist = []
    for i in range(num_layers):
        if selected_layers[i].size > 0:
            md = dist_mat[:, selected_layers[i]].min(axis=1)
        else:
            md = np.full(total_points, np.inf)
        min_dist.append(md)

    while any(r > 0 for r in remaining):
        best_layer = None
        best_point = None
        best_score = -np.inf

        for i in range(num_layers):
            if remaining[i] <= 0:
                continue
            candidates = np.where(available_mask)[0]
            if candidates.size == 0:
                break
            # pick farthest point for this layer
            md = min_dist[i][candidates]
            idx_local = np.argmax(md)
            point_idx = candidates[idx_local]
            score = md[idx_local] / max(target_spacing[i], 1e-6)
            if score > best_score:
                best_score = score
                best_layer = i
                best_point = point_idx

        if best_layer is None or best_point is None:
            break

        # assign
        selected_layers[best_layer] = np.append(selected_layers[best_layer], best_point)
        available_mask[best_point] = False
        remaining[best_layer] -= 1

        # update min_dist for all layers
        for i in range(num_layers):
            min_dist[i] = np.minimum(min_dist[i], dist_mat[:, best_point])

    free_idx = np.where(available_mask)[0]
    return selected_layers, free_idx

def refine_layers_global(points, layers, free_idx, rng,
                         steps=3000, candidate_pool=120, temp_start=0.5, temp_end=0.02,
                         cross_weight=0.2, penalty_weight=0.15, min_per=2):
    """Global refinement using simulated annealing swaps between layers and free points."""
    if not layers:
        return layers, free_idx

    layers = [layer.copy() for layer in layers]
    free_set = set(free_idx.tolist())
    layer_sizes = [len(layer) for layer in layers]
    dist_mat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    current_score = _global_score(dist_mat, layers, cross_weight=cross_weight,
                                  penalty_weight=penalty_weight, min_per=min_per)
    best_layers = [l.copy() for l in layers]
    best_score = current_score

    for step in range(steps):
        t = temp_start * ((temp_end / temp_start) ** (step / max(1, steps - 1)))
        move_type = rng.random()

        # Choose swap between layers
        if move_type < 0.5 and len(layers) > 1:
            idxs = [i for i, s in enumerate(layer_sizes) if s > 0]
            if len(idxs) < 2:
                continue
            a, b = rng.choice(idxs, size=2, replace=False)
            la = layers[a]
            lb = layers[b]
            ia = rng.integers(len(la))
            ib = rng.integers(len(lb))
            pa = la[ia]
            pb = lb[ib]

            la_trial = la.copy()
            lb_trial = lb.copy()
            la_trial[ia] = pb
            lb_trial[ib] = pa

            trial_layers = layers.copy()
            trial_layers[a] = la_trial
            trial_layers[b] = lb_trial
            trial_score = _global_score(dist_mat, trial_layers, cross_weight=cross_weight,
                                        penalty_weight=penalty_weight, min_per=min_per)

            delta = trial_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-6)):
                layers = trial_layers
                current_score = trial_score

        # Swap with free points
        else:
            idxs = [i for i, s in enumerate(layer_sizes) if s > 0]
            if not idxs or not free_set:
                continue
            a = rng.choice(idxs)
            la = layers[a]
            ia = rng.integers(len(la))
            pa = la[ia]

            free_list = np.array(list(free_set), dtype=int)
            k = min(candidate_pool, free_list.size)
            cand = rng.choice(free_list, size=k, replace=False)
            in_point = cand[rng.integers(len(cand))]

            la_trial = la.copy()
            la_trial[ia] = in_point

            trial_layers = layers.copy()
            trial_layers[a] = la_trial
            trial_score = _global_score(dist_mat, trial_layers, cross_weight=cross_weight,
                                        penalty_weight=penalty_weight, min_per=min_per)

            delta = trial_score - current_score
            if delta > 0 or rng.random() < np.exp(delta / max(t, 1e-6)):
                layers = trial_layers
                current_score = trial_score
                free_set.remove(in_point)
                free_set.add(pa)

        if current_score > best_score:
            best_score = current_score
            best_layers = [l.copy() for l in layers]

    free_idx = np.array(list(free_set), dtype=int)
    return best_layers, free_idx

def refine_last_layer(points, layers, free_idx, rng,
                      steps=4000, candidate_pool=200, preserve_other=0.85):
    """Targeted refinement for the last layer using swaps with free points and other layers."""
    if not layers:
        return layers, free_idx
    layers = [l.copy() for l in layers]
    free_set = set(free_idx.tolist())
    dist_mat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    last_idx = len(layers) - 1
    last = layers[last_idx]
    if last.size == 0:
        return layers, free_idx

    for _ in range(steps):
        move_type = rng.random()

        # Prefer swapping with free points for last layer
        if move_type < 0.7 and free_set:
            ia = rng.integers(len(last))
            pa = last[ia]
            free_list = np.array(list(free_set), dtype=int)
            k = min(candidate_pool, free_list.size)
            cand = rng.choice(free_list, size=k, replace=False)
            current_score = _layer_score_from_dist(dist_mat, last)
            best_score = current_score
            best_in = None
            for in_point in cand:
                trial = last.copy()
                trial[ia] = in_point
                score = _layer_score_from_dist(dist_mat, trial)
                if score > best_score:
                    best_score = score
                    best_in = in_point
            if best_in is not None:
                last[ia] = best_in
                layers[last_idx] = last
                free_set.remove(best_in)
                free_set.add(pa)
            continue

        # Swap with another layer (keep other layers from degrading too much)
        other_idx = rng.integers(len(layers) - 1)
        other = layers[other_idx]
        if other.size == 0:
            continue
        ia = rng.integers(len(last))
        ib = rng.integers(len(other))
        pa = last[ia]
        pb = other[ib]

        last_trial = last.copy()
        other_trial = other.copy()
        last_trial[ia] = pb
        other_trial[ib] = pa

        last_score = _layer_score_from_dist(dist_mat, last)
        last_score_trial = _layer_score_from_dist(dist_mat, last_trial)
        other_score = _layer_score_from_dist(dist_mat, other)
        other_score_trial = _layer_score_from_dist(dist_mat, other_trial)

        if last_score_trial > last_score and other_score_trial >= preserve_other * other_score:
            layers[last_idx] = last_trial
            layers[other_idx] = other_trial
            last = layers[last_idx]

    free_idx = np.array(list(free_set), dtype=int)
    return layers, free_idx

def ensure_layers_disjoint(layers, free_idx, rng):
    """Ensure no point index appears in multiple layers by swapping with free points."""
    layers = [l.copy() for l in layers]
    free_set = set(free_idx.tolist())
    seen = set()
    for i in range(len(layers)):
        layer = layers[i]
        for j in range(len(layer)):
            idx = layer[j]
            if idx in seen:
                if not free_set:
                    continue
                replacement = rng.choice(list(free_set))
                free_set.remove(replacement)
                free_set.add(idx)
                layer[j] = replacement
            seen.add(layer[j])
        layers[i] = layer
    free_idx = np.array(list(free_set), dtype=int)
    return layers, free_idx

def assign_remaining_points(points, layers, free_idx, rng,
                            penalty_weight=0.15, min_per=2):
    """Assign any remaining free points to layers to fully occupy the grid."""
    if free_idx.size == 0 or not layers:
        return layers, free_idx

    layers = [l.copy() for l in layers]
    dist_mat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    free = free_idx.copy()
    rng.shuffle(free)

    for p in free:
        best_layer = None
        best_score = -np.inf
        for i in range(len(layers)):
            trial = np.append(layers[i], p)
            score = _layer_score_from_dist(dist_mat, trial,
                                           penalty_weight=penalty_weight,
                                           min_per=min_per)
            if score > best_score:
                best_score = score
                best_layer = i
        layers[best_layer] = np.append(layers[best_layer], p)

    free_idx = np.array([], dtype=int)
    return layers, free_idx

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
            "patience": 5
        }
    )
    use_custom_refinement = config.get("use_custom_refinement", False)
    
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
    
    # Sample layers (only for defined layer_names)
    print("\nℹ Sampling layers...")
    selected_layers = []
    free_idx = np.arange(len(GRID_POINTS))
    selected_all = np.array([], dtype=int)
    
    num_layers = len(layer_names)
    # Prioritize smaller layers first to improve their regularity
    layer_order = list(range(num_layers))
    layer_order.sort(key=lambda i: N_list[i])
    temp_layers = [None] * num_layers
    for layer_i in layer_order:
        if layer_i >= len(N_list):
            continue
        n = N_list[layer_i]
        # Optimize each layer for regularity on the remaining free points
        new_sel = sample_layer_best(
            GRID_POINTS, free_idx, n, rng,
            trials=8, top_k=10, swap_steps=250, use_poisson=False, use_pam=True
        )
        temp_layers[layer_i] = new_sel
        if new_sel.size > 0:
            selected_all = np.concatenate([selected_all, new_sel])
            free_idx = np.setdiff1d(free_idx, new_sel, assume_unique=False)
    selected_layers = temp_layers

    # Optimizer from demo (layer-to-layer distribution improvement)
    if use_optimizer and len(selected_layers) > 1:
        print("ℹ Running point distribution optimizer (demo strategy)...")
        selected_layers = optimize_point_distribution(
            GRID_POINTS, selected_layers, **optimizer_params
        )
        selected_all = np.concatenate(selected_layers) if selected_layers else np.array([], dtype=int)
        free_idx = np.setdiff1d(np.arange(len(GRID_POINTS)), selected_all, assume_unique=False)
        print("✓ Optimizer done")

    if use_custom_refinement:
        # Global refinement: swap points between layers to improve overall regularity
        if len(selected_layers) > 1:
            print("ℹ Global refinement across layers...")
            selected_layers, free_idx = refine_layers_global(
                GRID_POINTS, selected_layers, free_idx, rng,
                steps=3500, candidate_pool=140, cross_weight=0.35,
                penalty_weight=0.15, min_per=2
            )
            # Recompute selected_all
            selected_all = np.concatenate(selected_layers) if selected_layers else np.array([], dtype=int)
            print("✓ Global refinement done")

        # Targeted refinement for the last layer (improve its regularity)
        if len(selected_layers) > 0:
            print("ℹ Targeted refinement for last layer...")
            selected_layers, free_idx = refine_last_layer(
                GRID_POINTS, selected_layers, free_idx, rng,
                steps=3500, candidate_pool=160, preserve_other=0.92
            )
            selected_all = np.concatenate(selected_layers) if selected_layers else np.array([], dtype=int)
            print("✓ Last layer refinement done")

    # Soft row/col constraint handled via penalty in scoring (no hard enforcement)

    # Ensure layers are disjoint after all refinements
    if len(selected_layers) > 0:
        selected_layers, free_idx = ensure_layers_disjoint(selected_layers, free_idx, rng)
        selected_all = np.concatenate(selected_layers) if selected_layers else np.array([], dtype=int)

    # Assign any remaining free points to layers to fully occupy the grid
    if free_idx.size > 0 and len(selected_layers) > 0:
        print("ℹ Assigning remaining free points to layers...")
        selected_layers, free_idx = assign_remaining_points(
            GRID_POINTS, selected_layers, free_idx, rng,
            penalty_weight=0.15, min_per=2
        )
        selected_all = np.concatenate(selected_layers) if selected_layers else np.array([], dtype=int)
        print("✓ All grid points assigned to layers")

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
    
    try:
        import matplotlib.tri as mtri
    except ImportError:
        mtri = None
    
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
            except:
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

    def getch_nonblocking(timeout=0.1):
        """Read a single keypress from terminal without Enter. Returns None if no key."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            rlist, _, _ = select.select([sys.stdin], [], [], timeout)
            if rlist:
                ch = sys.stdin.read(1)
                return ch
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    render()
    plt.show(block=False)

    while plt.fignum_exists(fig.number):
        key = getch_nonblocking(0.1)
        if not key:
            plt.pause(0.01)
            continue
        key = key.lower()
        if key == "x":
            print("✓ Exiting.")
            plt.close(fig)
            break
        if key in symbol_keys:
            layer_idx = symbol_keys[key]
            if layer_idx < len(selected_layers):
                show_symbols[layer_idx] = not show_symbols[layer_idx]
                status = "shown" if show_symbols[layer_idx] else "hidden"
                print(f"✓ Layer {layer_idx + 1} symbols {status}")
            render()
            continue
        if key.isdigit():
            layer_idx = int(key) - 1
            if 0 <= layer_idx < len(selected_layers):
                draw_triang[layer_idx] = not draw_triang[layer_idx]
                print(f"✓ Toggled triangulation for layer {key}")
                render()
            continue

if __name__ == "__main__":
    main()
