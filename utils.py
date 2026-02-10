"""Shared utility functions for grid analysis and point distribution."""

import numpy as np
import matplotlib.tri as mtri


def angular_diff(a, b):
    """Smallest absolute angular difference (radians) in [-pi, pi]."""
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def build_neighbors_from_tri(tri, n):
    """Build neighbor graph from Delaunay triangulation."""
    neighbors = [set() for _ in range(n)]
    if tri is None:
        return neighbors
    for a, b, c in tri.triangles:
        neighbors[a].update([b, c])
        neighbors[b].update([a, c])
        neighbors[c].update([a, b])
    return neighbors


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


def compute_layer_metrics(layer_indices, all_points, expected_spacing=None):
    """
    Compute distribution metrics for a layer.
    
    Args:
        layer_indices: array of point indices in this layer
        all_points: all grid points array
    
    Returns:
        dict with metrics: n, avg_dist, min_dist, max_dist, density, regularity_score
    """
    if len(layer_indices) == 0:
        return {"n": 0, "avg_dist": np.nan, "min_dist": np.nan, "max_dist": np.nan, 
                "density": 0, "regularity_score": 0}
    
    layer_points = all_points[layer_indices]
    n = len(layer_indices)
    
    # Compute pairwise distances
    if n < 2:
        return {"n": n, "avg_dist": np.nan, "min_dist": np.nan, "max_dist": np.nan,
                "density": 1.0, "regularity_score": 0}
    
    # Build Delaunay to get neighbors
    if n >= 3:
        try:
            tri = mtri.Triangulation(layer_points[:, 0], layer_points[:, 1])
            neighbors = build_neighbors_from_tri(tri, n)
        except:
            neighbors = [set() for _ in range(n)]
    else:
        # For < 3 points, just compute distance to nearest neighbor
        dists = np.linalg.norm(layer_points[:, None, :] - layer_points[None, :, :], axis=2)
        neighbors = [set(np.argsort(dists[i])[1:3]) for i in range(n)]
    
    # Compute neighbor distances
    neighbor_dists = []
    for i, nbrs in enumerate(neighbors):
        if nbrs:
            dists_to_neighbors = np.linalg.norm(
                layer_points[list(nbrs)] - layer_points[i], axis=1
            )
            neighbor_dists.extend(dists_to_neighbors)
    
    if neighbor_dists:
        neighbor_dists = np.array(neighbor_dists)
        avg_dist = neighbor_dists.mean()
        min_dist = neighbor_dists.min()
        max_dist = neighbor_dists.max()
        # Regularity: coefficient of variation of neighbor distances (lower is more regular)
        std_dist = neighbor_dists.std()
        regularity_score = std_dist / avg_dist if avg_dist > 0 else 0
        # Size-normalized regularity (optional): compare to expected spacing
        if expected_spacing is not None and expected_spacing > 0:
            normalized = neighbor_dists / expected_spacing
            regularity_score = normalized.std() / normalized.mean() if normalized.mean() > 0 else regularity_score
    else:
        avg_dist = min_dist = max_dist = regularity_score = np.nan
    
    # Density estimate: average distance to k-nearest neighbors averaged over points
    hull_area = compute_convex_hull_area(layer_points) if n >= 3 else 0
    density = n / (hull_area + 1e-6) if hull_area > 0 else 0
    
    return {
        "n": n,
        "avg_dist": avg_dist,
        "min_dist": min_dist,
        "max_dist": max_dist,
        "density": density,
        "regularity_score": regularity_score
    }


def compute_convex_hull_area(points):
    """Compute area of convex hull of points using bounding box as fallback."""
    if len(points) < 3:
        return 0
    
    # Try using Delaunay to estimate area
    try:
        tri = mtri.Triangulation(points[:, 0], points[:, 1])
        # Sum areas of all triangles
        area = 0
        for simplex in tri.triangles:
            triangle = points[simplex]
            # Shoelace formula for triangle area
            a, b, c = triangle
            area += abs((b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])) / 2
        return area if area > 0 else 0
    except:
        # Fallback: bounding box area
        x_min, x_max = points[:, 0].min(), points[:, 0].max()
        y_min, y_max = points[:, 1].min(), points[:, 1].max()
        return (x_max - x_min) * (y_max - y_min)


def find_point_to_swap(from_layer_idx, to_layer_idx, all_point_indices, all_points, all_layers):
    """
    Find a point in from_layer that would benefit moving to to_layer.
    
    Heuristic: choose point most distant from its layer's centroid,
    closest to target layer's centroid.
    """
    from_layer = all_layers[from_layer_idx]
    to_layer = all_layers[to_layer_idx]
    
    if len(from_layer) == 0 or len(to_layer) == 0:
        return None
    
    from_points = all_points[from_layer]
    to_points = all_points[to_layer]
    
    from_centroid = from_points.mean(axis=0)
    to_centroid = to_points.mean(axis=0)
    
    # Score each point in from_layer by distance to its own centroid
    dist_to_from_centroid = np.linalg.norm(from_points - from_centroid, axis=1)
    # And by distance to target centroid
    dist_to_to_centroid = np.linalg.norm(from_points - to_centroid, axis=1)
    
    # Higher score = candidate for moving out of from_layer
    # (far from own centroid, close to target centroid)
    score = dist_to_from_centroid - dist_to_to_centroid
    
    best_idx_in_layer = np.argmax(score)
    global_idx = from_layer[best_idx_in_layer]
    
    return global_idx


def extract_layers_from_indices(layer_indices_list, all_point_count):
    """Convert list of layer index arrays to numpy array format."""
    selected_layers = []
    for indices in layer_indices_list:
        selected_layers.append(np.array(indices, dtype=int))
    return selected_layers
