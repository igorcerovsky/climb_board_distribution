# Point Distribution Optimizer

A system for iteratively optimizing point distribution across multiple layers in a grid, targeting uniform density and regular spacing.

## Files

### `utils.py`
Shared utility functions extracted from the main grid analysis:
- `angular_diff()` - Angular difference computation
- `build_neighbors_from_tri()` - Build neighbor graph from Delaunay triangulation
- `euclidean_mst()` - Euclidean minimum spanning tree
- `compute_layer_metrics()` - Analyze per-layer distribution metrics
  - Count of points
  - Average/min/max neighbor distances
  - Density estimate
  - **Regularity score** (coefficient of variation of neighbor distances, lower is better)
- `compute_convex_hull_area()` - Convex hull area for density calculation
- `find_point_to_swap()` - Identify candidate point for moving between layers

### `point_distribution_optimizer.py`
Main optimization engine:

**`PointDistributionOptimizer` class:**
- Iteratively analyzes per-layer metrics via Delaunay triangulation
- Identifies irregular layers (high regularity score)
- Proposes point swaps to improve global uniformity
- Re-triangulates after each redistribution
- Stops on convergence or max iterations

**Key parameters:**
- `max_iterations` - Maximum optimization iterations (default: 50)
- `point_variance_tolerance` - Allow layer sizes to vary by ±10% (configurable)
- `improvement_threshold` - Minimum improvement to continue (default: 0.001)
- `patience` - Stop if no improvement for N iterations (default: 5)

**Metrics:**
- **Global Regularity** - Mean of per-layer regularity scores (lower = more uniform spacing)
- **Density CV** - Coefficient of variation in per-layer densities (lower = more uniform)
- **Overall Score** - Combined metric for optimization

### `demo_optimizer.py`
Example integration pipeline:
1. Load configuration and sample initial layers
2. Run point distribution optimization
3. Compute rotation angles for optimized points
4. Visualize before/after comparison

## Usage

### Basic optimization
```python
from point_distribution_optimizer import optimize_point_distribution

# Assuming you have grid_points and initial_layers
optimized_layers = optimize_point_distribution(
    grid_points=GRID_POINTS,
    initial_layers=selected_layers,
    max_iterations=30,
    point_variance_tolerance=0.10
)
```

### Complete pipeline with visualization
```bash
python3 demo_optimizer.py

# Or with custom config
python3 demo_optimizer.py -c custom_config.json
```

## How It Works

### Optimization Algorithm

1. **For each iteration:**
   - Compute Delaunay triangulation for each layer
   - Calculate neighbor distance statistics → regularity score
   - Calculate per-layer density
   - Compute global metrics: average regularity + density variance

2. **Decision logic:**
   - Identify worst-performing layer (highest regularity score = irregular spacing)
   - Find candidate layers with better regularity
   - Select a point from worst layer most likely to improve distribution:
     - Maximize distance to its own layer centroid (outlier point)
     - Minimize distance to target layer centroid (good fit)
   - Check size constraints (±10% variance allowed)
   - If valid, move the point

3. **Stopping criteria:**
   - No improvement for `patience` iterations
   - Maximum iterations reached
   - Number of completed iterations

### Metrics Explained

**Regularity Score** = Standard Deviation / Mean of neighbor distances
- Lower = more uniform spacing within layer
- Ranges from 0 (perfect uniformity) to high values (chaotic)
- Used to identify layers needing rebalancing

**Density CV** = Std Dev / Mean of per-layer densities
- Lower = more uniform layer densities
- Ensures points aren't concentrated in few layers

**Overall Score** = Regularity + 0.5×Density CV
- Lower = better global distribution

## Integration with `sample_grid.py`

The optimizer works with the existing grid analysis framework:
- Reuses `sample_additional()` for initial sampling
- Compatible with layer configuration format
- Can be run as preprocessing before rotation optimization
- Results can be fed into existing visualization pipeline

## Example Output

```
============================================================
POINT DISTRIBUTION OPTIMIZER
============================================================
Initial configuration: [32, 32, 44] points per layer
Target points: [35, 35, 48]
Tolerance: ±10.0%

Iteration   0: Global Regularity=0.4523, Density CV=0.2341, Score=0.5691, Sizes=[32, 32, 44]
  → Swapped point from layer 0
Iteration   1: Global Regularity=0.3847, Density CV=0.2156, Score=0.4975, Sizes=[31, 33, 44]
  → Swapped point from layer 1
...
Iteration   8: Global Regularity=0.2341, Density CV=0.0987, Score=0.2835, Sizes=[33, 32, 43]
  → No beneficial swaps found.

No improvement for 5 iterations. Stopping.

OPTIMIZATION SUMMARY
────────────────────────────────────────────────────────────
Iterations completed: 9

Initial state:
  Layer sizes: [32, 32, 44]
  Global regularity: 0.4523
  Overall score: 0.5691

Final state:
  Layer sizes: [33, 32, 43]
  Global regularity: 0.2341
  Overall score: 0.2835

Improvement:
  Score improved by 0.2856 (50.2%)
```

## Future Enhancements

- Batch point swaps (multiple moves per iteration)
- Layer grouping/merging heuristics
- Interactive visualization of optimization progress
- Export optimization history to JSON
- Parallel evaluation of candidate swaps
