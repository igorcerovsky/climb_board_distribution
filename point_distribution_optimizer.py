"""Point distribution optimizer for multi-layer grid analysis.

Iteratively improves point distribution across layers by:
1. Analyzing per-layer regularity (via Delaunay triangulation metrics)
2. Identifying irregular layers
3. Proposing point swaps between layers
4. Updating distribution and re-triangulating
5. Stopping when no improvement or iteration limit reached
"""

import numpy as np
import matplotlib.tri as mtri
from utils import (
    build_neighbors_from_tri,
    compute_layer_metrics,
    find_point_to_swap
)


class PointDistributionOptimizer:
    """Optimizes point distribution across layers for uniform density and regularity."""
    
    def __init__(self, grid_points, initial_layers, max_iterations=50, 
                 point_variance_tolerance=0.10, improvement_threshold=0.001,
                 patience=5):
        """
        Initialize optimizer.
        
        Args:
            grid_points: (N, 2) array of all grid points
            initial_layers: list of arrays, each array contains indices of points in that layer
            max_iterations: maximum optimization iterations
            point_variance_tolerance: allow layer sizes to vary by this fraction (±10% = 0.10)
            improvement_threshold: minimum relative improvement to continue
            patience: iterations without improvement before stopping
        """
        self.grid_points = grid_points
        self.layers = [np.array(layer, dtype=int).copy() for layer in initial_layers]
        self.max_iterations = max_iterations
        self.point_variance_tolerance = point_variance_tolerance
        self.improvement_threshold = improvement_threshold
        self.patience = patience
        
        self.num_layers = len(self.layers)
        self.total_points = sum(len(layer) for layer in self.layers)
        self.initial_point_counts = [len(layer) for layer in self.layers]
        # Min size: can lose up to tolerance fraction, max size: can gain up to tolerance fraction
        self.min_point_counts = [
            (max(1, int(count * (1 - point_variance_tolerance))) if count > 0 else 0)
            for count in self.initial_point_counts
        ]
        self.max_point_counts = [
            (max(1, int(count * (1 + point_variance_tolerance))) if count > 0 else 0)
            for count in self.initial_point_counts
        ]
        
        self.iteration_history = []
    
    def optimize(self):
        """Run the optimization loop."""
        print("\n" + "="*60)
        print("POINT DISTRIBUTION OPTIMIZER")
        print("="*60)
        print(f"Initial configuration: {[len(l) for l in self.layers]} points per layer")
        print(f"Allowed range: {self.min_point_counts} to {self.max_point_counts}")
        print(f"Tolerance: ±{self.point_variance_tolerance*100:.1f}%")
        print()
        
        no_improvement_count = 0
        best_overall_score = float('inf')
        best_layers = [np.array(layer, dtype=int).copy() for layer in self.layers]
        
        for iteration in range(self.max_iterations):
            # Compute metrics for all layers
            metrics_list = []
            # Expected spacing per layer based on area/points (size-aware)
            all_points = self.grid_points
            x_min, x_max = all_points[:, 0].min(), all_points[:, 0].max()
            y_min, y_max = all_points[:, 1].min(), all_points[:, 1].max()
            area = (x_max - x_min) * (y_max - y_min)

            for layer_idx, layer in enumerate(self.layers):
                n = len(layer)
                expected_spacing = np.sqrt(area / max(n, 1)) if n > 0 else None
                metrics = compute_layer_metrics(layer, self.grid_points, expected_spacing=expected_spacing)
                metrics_list.append(metrics)
            
            # Global regularity score (mean of per-layer regularity)
            regularity_scores = [m["regularity_score"] for m in metrics_list 
                                if not np.isnan(m["regularity_score"])]
            global_regularity = np.mean(regularity_scores) if regularity_scores else 0
            
            # Density uniformity (coefficient of variation in densities)
            densities = [m["density"] for m in metrics_list if m["density"] > 0]
            density_cv = np.std(densities) / np.mean(densities) if densities else 0
            
            overall_score = global_regularity + 0.5 * density_cv
            
            self.iteration_history.append({
                "iteration": iteration,
                "global_regularity": global_regularity,
                "density_cv": density_cv,
                "overall_score": overall_score,
                "layer_sizes": [len(l) for l in self.layers],
                "metrics": metrics_list
            })
            
            # Print progress
            print(f"Iteration {iteration:3d}: Global Regularity={global_regularity:.4f}, "
                  f"Density CV={density_cv:.4f}, Score={overall_score:.4f}, "
                  f"Sizes={[len(l) for l in self.layers]}")
            
            # Check improvement
            if overall_score < best_overall_score - self.improvement_threshold:
                best_overall_score = overall_score
                best_layers = [np.array(layer, dtype=int).copy() for layer in self.layers]
                no_improvement_count = 0
            else:
                no_improvement_count += 1
            
            # Stop if no improvement
            if no_improvement_count >= self.patience:
                print(f"\nNo improvement for {self.patience} iterations. Stopping.")
                break
            
            # Identify worst layer(s)
            worst_layer_idx = np.argmax([m["regularity_score"] if not np.isnan(m["regularity_score"]) 
                                        else 0 for m in metrics_list])
            
            # Try to improve by swapping points
            swapped = self._attempt_optimization_step(worst_layer_idx, metrics_list)
            
            if not swapped:
                print(f"  → No beneficial swaps found.")
                no_improvement_count += 2  # Encourage stopping if no swaps possible
            else:
                print(f"  → Swapped point from layer {worst_layer_idx}")
        
        # Restore the best state observed during optimization.
        self.layers = [np.array(layer, dtype=int).copy() for layer in best_layers]

        print("\n" + "="*60)
        print("OPTIMIZATION COMPLETE")
        print("="*60)
        print(f"Final configuration: {[len(l) for l in self.layers]} points per layer")
        if best_overall_score < float('inf'):
            print(f"Final overall score: {best_overall_score:.4f}")
        elif self.iteration_history:
            print(f"Final overall score: {self.iteration_history[-1]['overall_score']:.4f}")
        print()
        
        return self.layers
    
    def _attempt_optimization_step(self, worst_layer_idx, metrics_list):
        """Attempt to improve distribution by swapping a point."""
        worst_metrics = metrics_list[worst_layer_idx]
        worst_regularity = worst_metrics["regularity_score"]
        
        # For variable-size mode, avoid draining the source layer too close to min.
        # In fixed-size mode (min==max), allow swap logic below to run.
        fixed_size_worst = (
            self.min_point_counts[worst_layer_idx] ==
            self.max_point_counts[worst_layer_idx]
        )
        if (not fixed_size_worst) and (
            len(self.layers[worst_layer_idx]) <= self.min_point_counts[worst_layer_idx] + 1
        ):
            return False
        
        # Find candidate layers to swap points FROM/TO
        # Only consider layers with SIGNIFICANTLY better regularity (at least 10% improvement)
        better_layers = []
        for i, metrics in enumerate(metrics_list):
            if i == worst_layer_idx:
                continue
            if metrics["regularity_score"] < worst_regularity * 0.9 and len(self.layers[i]) > 0:
                better_layers.append(i)
        
        if not better_layers:
            return False
        
        # Try swapping with best candidate
        target_layer_idx = min(better_layers, 
                              key=lambda i: metrics_list[i]["regularity_score"])
        
        # Find point to move
        point_to_move = find_point_to_swap(
            worst_layer_idx, target_layer_idx, 
            None, self.grid_points, self.layers
        )

        if point_to_move is None:
            return False

        # If sizes are fixed (min == max), do a swap to keep counts stable
        if self.min_point_counts[worst_layer_idx] == self.max_point_counts[worst_layer_idx] and \
           self.min_point_counts[target_layer_idx] == self.max_point_counts[target_layer_idx]:
            point_to_return = find_point_to_swap(
                target_layer_idx, worst_layer_idx,
                None, self.grid_points, self.layers
            )
            if point_to_return is None:
                return False
            # Perform swap
            self.layers[worst_layer_idx] = np.setdiff1d(
                self.layers[worst_layer_idx], [point_to_move], assume_unique=False
            )
            self.layers[worst_layer_idx] = np.concatenate([
                self.layers[worst_layer_idx], [point_to_return]
            ])
            self.layers[target_layer_idx] = np.setdiff1d(
                self.layers[target_layer_idx], [point_to_return], assume_unique=False
            )
            self.layers[target_layer_idx] = np.concatenate([
                self.layers[target_layer_idx], [point_to_move]
            ])
            return True

        # Check size constraints - enforce strict min/max bounds
        if len(self.layers[worst_layer_idx]) <= self.min_point_counts[worst_layer_idx]:
            return False
        if len(self.layers[target_layer_idx]) >= self.max_point_counts[target_layer_idx]:
            return False

        # Perform move
        self.layers[worst_layer_idx] = np.setdiff1d(
            self.layers[worst_layer_idx], [point_to_move], assume_unique=False
        )
        self.layers[target_layer_idx] = np.concatenate([
            self.layers[target_layer_idx], [point_to_move]
        ])
        
        return True
    
    def get_optimized_layers(self):
        """Return optimized layer point distributions."""
        return self.layers
    
    def print_summary(self):
        """Print summary of optimization results."""
        if not self.iteration_history:
            return
        
        initial = self.iteration_history[0]
        final = self.iteration_history[-1]
        
        print("\nOPTIMIZATION SUMMARY")
        print("-" * 60)
        print(f"Iterations completed: {len(self.iteration_history)}")
        print(f"\nInitial state:")
        print(f"  Layer sizes: {initial['layer_sizes']}")
        print(f"  Global regularity: {initial['global_regularity']:.4f}")
        print(f"  Overall score: {initial['overall_score']:.4f}")
        print(f"\nFinal state:")
        print(f"  Layer sizes: {final['layer_sizes']}")
        print(f"  Global regularity: {final['global_regularity']:.4f}")
        print(f"  Overall score: {final['overall_score']:.4f}")
        print(f"\nImprovement:")
        improvement = initial['overall_score'] - final['overall_score']
        improvement_pct = (improvement / initial['overall_score'] * 100) if initial['overall_score'] > 0 else 0
        print(f"  Score improved by {improvement:.4f} ({improvement_pct:.1f}%)")
        print()


def optimize_point_distribution(grid_points, initial_layers, **kwargs):
    """
    Convenience function to optimize point distribution.
    
    Args:
        grid_points: (N, 2) array of all grid points
        initial_layers: list of point index arrays
        **kwargs: passed to PointDistributionOptimizer
    
    Returns:
        optimized_layers: list of point index arrays
    """
    optimizer = PointDistributionOptimizer(grid_points, initial_layers, **kwargs)
    optimized = optimizer.optimize()
    optimizer.print_summary()
    return optimized
