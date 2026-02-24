#!/usr/bin/env python3
"""
Nested layer generator:
- Each layer contains multiple sub-layers.
- Each sub-layer has its own regular distribution + triangulation.
- Each top-level layer also has a triangulation over all its sub-layer points.
"""

import json
import argparse
import sys
import tty
import termios
import select
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

from utils import build_neighbors_from_tri
from point_distribution_optimizer import optimize_point_distribution


def load_config(config_file=None):
    defaults = {
        "grid": {
            "len_x_mm": 3400,
            "len_y_mm": 3200,
            "distance_mm": 200
        },
        "layers": [
            {
                "name": "Jug",
                "color": "red",
                "draw_triang": True,
                "rot_limits": [0, 350],
                "sublayers": [
                    {"name": "Jug-A", "N": 22, "color": "red", "marker": "o", "draw_triang": True},
                    {"name": "Jug-B", "N": 22, "color": "darkred", "marker": "x", "draw_triang": True}
                ]
            },
            {
                "name": "Pinch",
                "color": "blue",
                "draw_triang": True,
                "rot_limits": [45, 135],
                "sublayers": [
                    {"name": "Pinch-A", "N": 24, "color": "blue", "marker": "o", "draw_triang": True},
                    {"name": "Pinch-B", "N": 24, "color": "navy", "marker": "x", "draw_triang": True}
                ]
            },
            {
                "name": "Sloper (flat)",
                "color": "purple",
                "draw_triang": True,
                "rot_limits": [50, 120],
                "sublayers": [
                    {"name": "Sloper-A", "N": 22, "color": "purple", "marker": "o", "draw_triang": True},
                    {"name": "Sloper-B", "N": 22, "color": "mediumorchid", "marker": "x", "draw_triang": True}
                ]
            },
            {
                "name": "Volume",
                "color": "indigo",
                "draw_triang": True,
                "rot_limits": [20, 160],
                "sublayers": [
                    {"name": "Volume-A", "N": 22, "color": "indigo", "marker": "o", "draw_triang": True},
                    {"name": "Volume-B", "N": 22, "color": "slateblue", "marker": "s", "draw_triang": True}
                ]
            },
            {
                "name": "Edge",
                "color": "green",
                "draw_triang": True,
                "rot_limits": [20, 160],
                "sublayers": [
                    {"name": "Edge-A", "N": 22, "color": "green", "marker": "o", "draw_triang": True},
                    {"name": "Edge-B", "N": 22, "color": "seagreen", "marker": "x", "draw_triang": True}
                ]
            },
            {
                "name": "Hold",
                "color": "magenta",
                "draw_triang": True,
                "rot_limits": [20, 160],
                "sublayers": [
                    {"name": "Hold-A", "N": 24, "color": "magenta", "marker": "o", "draw_triang": True},
                    {"name": "Hold-B", "N": 24, "color": "deeppink", "marker": "x", "draw_triang": True}
                ]
            }
        ]
    }

    if config_file and Path(config_file).exists():
        with open(config_file, "r") as f:
            cfg = json.load(f)
        # shallow merge
        defaults.update(cfg)
        if "grid" in cfg:
            defaults["grid"].update(cfg["grid"])
        if "layers" in cfg:
            defaults["layers"] = cfg["layers"]
        print(f"Loaded configuration from {config_file}")
    else:
        if config_file:
            print(f"Config file {config_file} not found. Using defaults.")
        else:
            print("Using default configuration")
    return defaults


def build_grid(grid_cfg):
    len_x_mm = grid_cfg["len_x_mm"]
    len_y_mm = grid_cfg["len_y_mm"]
    distance_mm = grid_cfg["distance_mm"]

    num_points_x = int(len_x_mm / distance_mm)
    num_points_y = int(len_y_mm / distance_mm)
    distance_m = distance_mm / 1000.0
    x_coords = np.arange(0, num_points_x * distance_m, distance_m)
    y_coords = np.arange(0, num_points_y * distance_m, distance_m)

    X, Y = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([X.ravel(), Y.ravel()])
    return points, num_points_x, num_points_y


def greedy_farthest(points, free_idx, N, rng, top_k=6):
    if N <= 0 or free_idx.size == 0:
        return np.array([], dtype=int)

    free_points = points[free_idx]
    chosen = []

    first = rng.integers(len(free_idx))
    chosen.append(free_idx[first])

    dist = np.full(len(free_idx), np.inf)
    last_point = free_points[first]
    dist = np.minimum(dist, np.linalg.norm(free_points - last_point, axis=1))

    for _ in range(1, min(N, len(free_idx))):
        # pick randomly among top_k farthest points to avoid grid artifacts
        candidates = np.argsort(dist)[-min(top_k, len(dist)):]
        idx_local = rng.choice(candidates)
        chosen.append(free_idx[idx_local])
        new_point = free_points[idx_local]
        dist = np.minimum(dist, np.linalg.norm(free_points - new_point, axis=1))

    return np.array(chosen, dtype=int)

def local_swap_improve(points, base_idx, free_idx, rng, steps=200):
    """Local improvement: swap points to maximize min distance within the set."""
    if base_idx.size < 2:
        return base_idx
    current = base_idx.copy()
    free_set = set(free_idx.tolist())
    free_set -= set(current.tolist())
    free_list = np.array(list(free_set), dtype=int)
    if free_list.size == 0:
        return current

    dist_mat = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)

    def score(idxs):
        d = dist_mat[np.ix_(idxs, idxs)].copy()
        np.fill_diagonal(d, np.inf)
        nn = np.min(d, axis=1)
        return float(nn.min() + 0.2 * nn.mean())

    best_score = score(current)
    for _ in range(steps):
        out_pos = rng.integers(len(current))
        out_point = current[out_pos]
        in_point = rng.choice(free_list)
        trial = current.copy()
        trial[out_pos] = in_point
        s = score(trial)
        if s > best_score:
            best_score = s
            current = trial
            free_list = np.array([p for p in free_list if p != in_point] + [out_point], dtype=int)
    return current


def triangulation_edges(points, tri):
    edge_counts = {}
    for tri_idx in tri.triangles:
        a, b, c = tri_idx
        edges = [(a, b), (b, c), (c, a)]
        for u, v in edges:
            e = tuple(sorted((u, v)))
            edge_counts[e] = edge_counts.get(e, 0) + 1
    return edge_counts


def optimize_rotations(neighbors, rot_min_deg, rot_max_deg, rng,
                       iterations=60, candidates_per_point=80):
    rot_min = np.deg2rad(rot_min_deg)
    rot_max = np.deg2rad(rot_max_deg)
    n = len(neighbors)
    if n == 0:
        return np.array([])
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
                diffs = np.abs(a - nbr_angles) % (2 * np.pi)
                diffs = np.minimum(diffs, 2 * np.pi - diffs)
                score = diffs.min()
                if score > best_score:
                    best_score = score
                    best_angle = a
            angles[i] = best_angle
    return angles


def main():
    parser = argparse.ArgumentParser(description="Nested layer sampling and visualization")
    parser.add_argument("-c", "--config", help="Path to JSON config", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    rng = np.random.default_rng(42)
    top_layer_optimizer_cfg = config.get(
        "top_layer_optimizer",
        {
            "max_iterations": 20,
            "point_variance_tolerance": 0.0,
            "improvement_threshold": 0.001,
            "patience": 5
        }
    )

    points, num_x, num_y = build_grid(config["grid"])
    print(f"Grid size: {num_x} x {num_y} = {len(points)} points")
    free_idx = np.arange(len(points))

    layers_cfg = config["layers"]
    layers = []

    # 1) Sample top-level layers for regular distribution first
    layer_sizes = []
    for layer_cfg in layers_cfg:
        n = sum(int(sub.get("N", 0)) for sub in layer_cfg.get("sublayers", []))
        layer_sizes.append(n)

    # Size-aware order avoids starving smaller layers.
    layer_order = sorted(
        range(len(layer_sizes)),
        key=lambda i: (layer_sizes[i], float(rng.random()))
    )
    top_layer_indices = [np.array([], dtype=int) for _ in layer_sizes]
    for li in layer_order:
        n = layer_sizes[li]
        sel = greedy_farthest(points, free_idx, n, rng, top_k=8)
        sel = local_swap_improve(points, sel, free_idx, rng, steps=300)
        if sel.size > 0:
            free_idx = np.setdiff1d(free_idx, sel, assume_unique=False)
        top_layer_indices[li] = sel

    # Inter-layer balancing: optimize top-level layer distribution before sublayer split.
    if len(top_layer_indices) > 1:
        print("Optimizing top-layer distribution...")
        top_layer_indices = optimize_point_distribution(
            points, top_layer_indices, **top_layer_optimizer_cfg
        )
        used = np.concatenate(top_layer_indices) if top_layer_indices else np.array([], dtype=int)
        free_idx = np.setdiff1d(np.arange(len(points)), used, assume_unique=False)

    # 2) Split each top-level layer into sublayers, optimized within that layer
    for layer_cfg, layer_sel in zip(layers_cfg, top_layer_indices):
        layer = {
            "name": layer_cfg.get("name", "Layer"),
            "color": layer_cfg.get("color", "black"),
            "draw_triang": layer_cfg.get("draw_triang", True),
            "rot_limits": layer_cfg.get("rot_limits"),
            "sublayers": [],
            "indices": layer_sel
        }

        local_free = layer_sel.copy()
        for sub_cfg in layer_cfg.get("sublayers", []):
            n = int(sub_cfg.get("N", 0))
            sel = greedy_farthest(points, local_free, n, rng, top_k=8)
            sel = local_swap_improve(points, sel, local_free, rng, steps=250)
            if sel.size > 0:
                local_free = np.setdiff1d(local_free, sel, assume_unique=False)
            layer["sublayers"].append({
                "name": sub_cfg.get("name", "Sub"),
                "N": n,
                "color": sub_cfg.get("color", layer["color"]),
                "marker": sub_cfg.get("marker", "o"),
                "draw_triang": sub_cfg.get("draw_triang", True),
                "indices": sel,
                "rot_limits": sub_cfg.get("rot_limits")
            })

        layers.append(layer)

    # Build combined layer indices (already set), and ensure sublayers cover the layer
    for layer in layers:
        all_idx = []
        for sub in layer["sublayers"]:
            if sub["indices"].size > 0:
                all_idx.append(sub["indices"])
        if all_idx:
            layer["indices"] = np.concatenate(all_idx)

    # Validate rotation configuration and surface missing limits explicitly.
    missing_rotation_limits = []
    for layer in layers:
        for sub in layer["sublayers"]:
            if sub.get("rot_limits") is None and layer.get("rot_limits") is None:
                missing_rotation_limits.append(f"{layer['name']}/{sub['name']}")
    if missing_rotation_limits:
        print(
            "⚠ Rotations disabled (missing rot_limits) for: " +
            ", ".join(missing_rotation_limits)
        )

    # Compute rotations per sublayer (inherit from parent if not set)
    for layer in layers:
        for sub in layer["sublayers"]:
            rot_limits = sub.get("rot_limits")
            if rot_limits is None:
                rot_limits = layer.get("rot_limits")
            if rot_limits is None:
                sub["rot_rad"] = None
                continue
            idxs = sub["indices"]
            if idxs.size == 0:
                sub["rot_rad"] = None
                continue
            pts = points[idxs]
            if idxs.size >= 3:
                tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
                neighbors = build_neighbors_from_tri(tri, len(idxs))
            else:
                neighbors = [set() for _ in range(len(idxs))]
            rot_min, rot_max = rot_limits
            sub["rot_rad"] = optimize_rotations(
                neighbors, rot_min, rot_max, rng,
                iterations=60, candidates_per_point=80
            )

    # --- Interactive Visualization (toggle main layers) ---
    show_layer = {i: True for i in range(len(layers))}
    draw_layer_triang = {i: layers[i].get("draw_triang", True) for i in range(len(layers))}

    def getch_nonblocking(timeout=0.1):
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

    fig, ax = plt.subplots(figsize=(9, 7))

    def render():
        ax.clear()

        legend_handles = []
        legend_labels = []

        if free_idx.size > 0:
            h = ax.scatter(points[free_idx, 0], points[free_idx, 1],
                           color="lightgrey", s=4, label="Free grid points")
            legend_handles.append(h)
            legend_labels.append("Free grid points")

        for li, layer in enumerate(layers):
            # Main layer legend entry (always visible in legend)
            main_label = f"{li+1}. {layer['name']} (layer)"
            main_handle, = ax.plot([], [], color=layer["color"], linewidth=2.0)
            legend_handles.append(main_handle)
            legend_labels.append(main_label)

            if not show_layer.get(li, True):
                continue
            for sub in layer["sublayers"]:
                idxs = sub["indices"]
                if idxs.size == 0:
                    continue
                pts = points[idxs]
                if sub["marker"] == "x":
                    h = ax.scatter(pts[:, 0], pts[:, 1],
                                   color=sub["color"], s=18, marker="x",
                                   linewidths=0.5)
                else:
                    h = ax.scatter(pts[:, 0], pts[:, 1],
                                   color=sub["color"], s=18, marker=sub["marker"])
                legend_handles.append(h)
                legend_labels.append(f"{sub['name']} (N={len(idxs)})")

                # Sublayer triangulation disabled (draw only top-layer triangulation)

            # Top-level layer triangulation
            if draw_layer_triang.get(li, True):
                idxs = layer["indices"]
                if idxs.size >= 3:
                    pts = points[idxs]
                    tri = mtri.Triangulation(pts[:, 0], pts[:, 1])
                    edges = triangulation_edges(pts, tri)
                    for (u, v), count in edges.items():
                        if count == 2:
                            p1, p2 = pts[u], pts[v]
                            ax.plot([p1[0], p2[0]], [p1[1], p2[1]],
                                    color=layer["color"], linewidth=1.8, alpha=0.6)
                    # (main layer already in legend)

            # Rotation markers for sublayers (if rot_limits provided)
            for sub in layer["sublayers"]:
                if sub.get("rot_rad") is None:
                    continue
                idxs = sub["indices"]
                if idxs.size == 0:
                    continue
                pts = points[idxs]
                seg_len = config["grid"]["distance_mm"] / 1000.0 * 0.5
                u = np.cos(sub["rot_rad"]) * seg_len
                v = np.sin(sub["rot_rad"]) * seg_len
                ax.quiver(
                    pts[:, 0] - u, pts[:, 1] - v,
                    2 * u, 2 * v,
                    angles="xy", scale_units="xy", scale=1,
                    width=0.002, color=sub["color"], alpha=0.8
                )

        ax.set_aspect("equal")
        ax.set_title("Nested Layers: Sublayers + Layer Triangulation")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.legend(legend_handles, legend_labels,
                  loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
        fig.tight_layout(rect=[0, 0, 0.80, 1])
        fig.canvas.draw_idle()

    print("\n" + "=" * 60)
    print("CONTROLS:")
    print("  Numbers (1-N): Toggle main layer visibility")
    print("  t + number:    Toggle main layer triangulation (e.g., t1)")
    print("  x:             Quit")
    print("=" * 60)

    render()
    plt.show(block=False)

    pending_toggle = None
    while plt.fignum_exists(fig.number):
        key = getch_nonblocking(0.1)
        if not key:
            plt.pause(0.01)
            continue
        key = key.lower()
        if key == "x":
            plt.close(fig)
            break
        if key == "t":
            pending_toggle = "triang"
            continue
        if key.isdigit():
            idx = int(key) - 1
            if 0 <= idx < len(layers):
                if pending_toggle == "triang":
                    draw_layer_triang[idx] = not draw_layer_triang.get(idx, True)
                else:
                    show_layer[idx] = not show_layer.get(idx, True)
                pending_toggle = None
                render()
            continue


if __name__ == "__main__":
    main()
