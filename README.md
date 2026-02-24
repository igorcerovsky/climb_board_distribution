# Climb Board Distribution

This repository contains a Python script for distributing points on a grid using Mitchell's Best-Candidate Sampling, layered point sets, rotation assignment optimized over Delaunay neighbors, and visualization of the resulting graph.

## Quick Start

1. Install dependencies:
```bash
python3 -m pip install numpy matplotlib
```

2. Run the script:
```bash
python3 sample_grid.py
```

## What You Can Configure

- `N_list` to control how many points are added per iteration
- `layer_colors` to color each point layer and its Delaunay triangulation
- `rot_min_deg` / `rot_max_deg` to set rotation range

## Files

- `sample_grid.py` main script

## Notes

- Remaining free grid points are shown in grey.
- Delaunay triangulation is drawn per layer in the same color as that layer.
- Rotation markers match the color of their corresponding points.
