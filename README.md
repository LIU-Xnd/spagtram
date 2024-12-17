# SpaGTraM - Spatial Gradient Trajectory Mapper

A tool for automatically finding spatial gradient trajectories of certain feature values.

It uses a multiscale-sliding-window linear-regression strategy, with many user-customizable hyperparameters.

(Still under construction ...)

## Data formats

Feature values, e.g., expression values of a gene, `Y` (a numpy array). Each dimension corresponds to a sample.

Spatial coordinates of each sample, `df_coords` (a pandas DataFrame with columns `['x', 'y']`).

## Requirements

```
# python == 3.10.15
matplotlib == 3.5.3
numpy == 1.26.4
pandas == 1.5.3
sklearn == 1.5.1
statsmodels == 0.14.2
```

## Demo

[demonstration.ipynb](./demonstration.ipynb)
