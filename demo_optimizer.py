#!/usr/bin/env python3
"""
Demo: Integration of point distribution optimizer with existing grid analysis.

This script:
1. Loads configuration and generates initial point distribution
2. Runs point distribution optimization
3. Computes rotations based on optimized distribution
4. Visualizes results
"""

import numpy as np
import matplotlib.pyplot as plt
import json
import argparse
from pathlib import Path

# Import from existing modules
from sample_grid import load_config, GRID_POINTS, sample_additional, optimize_rotations
from point_distribution_optimizer import optimize_point_distribution
from utils import build_neighbors_from_tri, compute_layer_metrics

import matplotlib.tri as mtri


def run_optimization_pipeline(config_file=None):
    """Run full optimization pipeline: sampling → distribution optimization → rotation optimization."""
    
    # Load configuration
    config = load_config(config_file)
    layers_config = config["layers"]
    
    # Extract properties
    layer_names = [layer["name"] for layer in layers_config]
    N_list = [layer["N"] for layer in layers_config]
    draw_triang = [layer["draw_triang"] for layer in layers_config]
    rot_limits = [layer["rot_limits"] for layer in layers_config]
    layer_colors = [layer["color"] for layer in layers_config]
    
    print("\n" + "="*60)
    print("STEP 1: INITIAL POINT SAMPLING")
    print("="*60)
    
    # Sample layers (initial distribution)
    rng = np.random.default_rng(42)
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
    print(f"  Layer sizes: {[len(l) for l in selected_layers]}")
    
    # Print initial metrics
    print("\nInitial layer metrics:")
    for i, layer in enumerate(selected_layers):
        metrics = compute_layer_metrics(layer, GRID_POINTS)
        if metrics["n"] > 0:
            print(f"  Layer {i+1} ({layer_names[i]}): "
                  f"N={metrics['n']}, Regularity={metrics['regularity_score']:.4f}, "
                  f"Avg Dist={metrics['avg_dist']:.4f}")
    
    print("\n" + "="*60)
    print("STEP 2: POINT DISTRIBUTION OPTIMIZATION")
    print("="*60)
    
    # Optimize point distribution
    optimized_layers = optimize_point_distribution(
        GRID_POINTS, 
        selected_layers,
        max_iterations=20,
        point_variance_tolerance=0.10,
        improvement_threshold=0.001,
        patience=5
    )
    
    # Print optimized metrics
    print("\nOptimized layer metrics:")
    for i, layer in enumerate(optimized_layers):
        metrics = compute_layer_metrics(layer, GRID_POINTS)
        if metrics["n"] > 0:
            print(f"  Layer {i+1} ({layer_names[i]}): "
                  f"N={metrics['n']}, Regularity={metrics['regularity_score']:.4f}, "
                  f"Avg Dist={metrics['avg_dist']:.4f}")
    
    print("\n" + "="*60)
    print("STEP 3: ROTATION OPTIMIZATION")
    print("="*60)
    
    # Compute rotations for optimized distribution
    selected_all_opt = np.concatenate(optimized_layers)
    rot_rad = np.full(len(selected_all_opt), np.nan)
    
    try:
        import matplotlib.tri as mtri
    except ImportError:
        mtri = None
    
    for layer_i, layer in enumerate(optimized_layers):
        if layer.size == 0:
            continue
        
        if layer_i < len(rot_limits) and rot_limits[layer_i] is not None:
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
        
        global_indices = np.where(np.isin(selected_all_opt, layer))[0]
        rot_rad[global_indices] = layer_angles
    
    print("✓ Rotations computed for optimized distribution")
    
    # Compute visualization vectors
    seg_len = 0.1
    u = np.cos(rot_rad) * seg_len
    v = np.sin(rot_rad) * seg_len
    
    # Visualization
    print("\n" + "="*60)
    print("INTERACTIVE VISUALIZATION")
    print("="*60)
    
    # Prepare data for interactive visualization
    show_symbols = {i: True for i in range(len(optimized_layers))}
    draw_triang_interactive = draw_triang.copy()
    
    # Keyboard mapping
    symbol_keys = {
        'q': 0, 'w': 1, 'e': 2, 'r': 3, 't': 4,
        'y': 5, 'u': 6, 'i': 7, 'o': 8, 'p': 9
    }
    key_labels = {v: k for k, v in symbol_keys.items()}
    
    # Compute u, v fields for arrows
    u_map = np.full(len(GRID_POINTS), np.nan)
    v_map = np.full(len(GRID_POINTS), np.nan)
    u_map[selected_all_opt] = u
    v_map[selected_all_opt] = v
    
    # Find unused free points for visualization
    all_used = set(selected_all_opt)
    free_idx_final = np.array([i for i in range(len(GRID_POINTS)) if i not in all_used])
    
    # Interactive loop (same as sample_grid.py)
    while True:
        print("\n" + "="*60)
        print("CONTROLS:")
        print("  Numbers (1-N): Toggle Delaunay triangulation per layer")
        print("  Letters (q-p): Toggle colored symbols (show/hide as gray cross)")
        print("  'quit':        Exit interactive mode")
        print("="*60)
        print("OPTIMIZED LAYERS:")
        for i in range(len(optimized_layers)):
            status = "ON " if (i < len(draw_triang_interactive) and draw_triang_interactive[i]) else "OFF"
            name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
            symbol_key = key_labels.get(i, '?')
            symbol_status = "✓" if show_symbols.get(i, True) else "✗"
            print(f"  {i+1}. {name}: Tri={status} {symbol_key}={symbol_status}")
        print("="*60)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        
        # Plot free points
        if len(free_idx_final) > 0:
            plt.scatter(GRID_POINTS[free_idx_final, 0], GRID_POINTS[free_idx_final, 1],
                        color="lightgrey", s=2, label="Free grid points", alpha=0.5)
        
        # Plot layers
        for i, layer in enumerate(optimized_layers):
            if layer.size == 0:
                continue
            layer_points = GRID_POINTS[layer]
            
            if show_symbols.get(i, True):
                # Draw colored scatter points
                color = layer_colors[i % len(layer_colors)]
                layer_name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
                plt.scatter(layer_points[:, 0], layer_points[:, 1],
                            color=color, s=12, label=f"{layer_name} (N={len(layer)})")
            else:
                # Draw small gray crosses instead
                plt.scatter(layer_points[:, 0], layer_points[:, 1],
                            marker='x', color='gray', s=30, linewidths=0.4, alpha=0.6, 
                            label=f"Layer {i+1} (hidden)")
        
        # Plot rotations
        for i, layer in enumerate(optimized_layers):
            if layer.size == 0:
                continue
            # Skip rotation arrows if limits are None (fixed rotation)
            if i < len(rot_limits) and rot_limits[i] is None:
                continue
            # Skip rotation arrows if layer symbols are hidden
            if not show_symbols.get(i, True):
                continue
            
            color = layer_colors[i % len(layer_colors)]
            layer_name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
            layer_points = GRID_POINTS[layer]
            scale_factor = 0.2 if i < len(rot_limits) and rot_limits[i] is None else 0.6
            plt.quiver(
                layer_points[:, 0] - scale_factor * u_map[layer],
                layer_points[:, 1] - scale_factor * v_map[layer],
                2*scale_factor*u_map[layer], 2*scale_factor*v_map[layer],
                angles="xy", scale_units="xy", scale=1,
                width=0.003, color=color, alpha=0.85,
                label=f"{layer_name} Rotation"
            )
        
        # Plot triangulations
        if mtri is not None:
            for i, layer in enumerate(optimized_layers):
                if layer.size < 3:
                    continue
                if i < len(draw_triang_interactive) and not draw_triang_interactive[i]:
                    continue
                
                color = layer_colors[i % len(layer_colors)]
                layer_name = layer_names[i] if i < len(layer_names) else f"Layer {i+1}"
                layer_points = GRID_POINTS[layer]
                
                try:
                    layer_tri = mtri.Triangulation(layer_points[:, 0], layer_points[:, 1])
                    
                    for tri_idx in layer_tri.triangles:
                        triangle = layer_points[tri_idx]
                        triangle_closed = np.vstack([triangle, triangle[0]])
                        plt.plot(triangle_closed[:, 0], triangle_closed[:, 1], 
                                color=color, linewidth=0.15, alpha=0.35)
                    
                    plt.plot([], [], color=color, linewidth=0.9, label=f"{layer_name} Delaunay")
                except:
                    pass
        
        plt.gca().set_aspect("equal")
        plt.title("Optimized Point Distribution - Interactive Visualization")
        plt.xlabel("X (m)")
        plt.ylabel("Y (m)")
        plt.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0, fontsize=9)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        plt.show(block=False)
        
        # Get user input
        user_input = input(f"\nEnter layer number to toggle triangulation (1-{len(optimized_layers)}),\n"
                          f"letter (q-p) to toggle symbol visibility, or 'quit' to exit: ").strip().lower()
        
        if user_input in ['quit', 'exit', 'x']:
            print("✓ Exiting interactive visualization.")
            plt.close('all')
            break
        
        # Check if input is a symbol key (q-p for layers 0-9)
        if user_input in symbol_keys:
            layer_idx = symbol_keys[user_input]
            if layer_idx < len(optimized_layers):
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
            if 0 <= layer_idx < len(optimized_layers):
                draw_triang_interactive[layer_idx] = not draw_triang_interactive[layer_idx]
                print(f"✓ Toggled triangulation for layer {user_input}")
                plt.close('all')
            else:
                print(f"✗ Invalid layer number. Please enter 1-{len(optimized_layers)}")
        except ValueError:
            print("✗ Invalid input. Please enter a number (1-N), letter (q-p), or 'quit'.")
    
    return optimized_layers


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Optimize point distribution across layers"
    )
    parser.add_argument("-c", "--config", help="Path to JSON configuration file", default=None)
    args = parser.parse_args()
    
    run_optimization_pipeline(args.config)
