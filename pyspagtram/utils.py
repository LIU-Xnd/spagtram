#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from typing import List, Tuple, Union
import matplotlib.pyplot as plt


# In[17]:


def plot_raw(
    Y: Union[pd.Series, np.ndarray],
    df_coord: pd.DataFrame,
    s=10,
    alpha=0.8,
):
    """
    Create an axes with given data plotted.
    
    Args:
        Y (pd.Series | np.ndarray): feature values of spots.
        df_coord (pd.DataFrame): coordinates of spots, spot-by-coordinate, with two columns 'x' and 'y'.
    
    Return:
        Axes: x and y coordinates plotted with Y as color values.
    """
    try:
        x = df_coord['x']
        y = df_coord['y']
    except KeyError:
        assert df_coord.shape[1] == 2
        x = df_coord[:, 0]
        y = df_coord[:, 1]

    ax = plt.scatter(
        x=x,
        y=y,
        c=Y,
        s=s,
        alpha=alpha,
        cmap='jet',
    )
    return ax


# In[18]:


def simulate(
    coordinates_centers: Union[List[np.ndarray], None] = None,
    n_spots_per_cluster: int = 128,
    means_centers: Union[List[float], None] = None,
    scales_centers: Union[List[float], None] = None,
    var_noise: Union[float, None] = None,
    var_name: str = "feature",
    spot_names: Union[List[str], None] = None,
    non_negative: bool = False,
) -> Tuple[pd.Series, pd.DataFrame]:
    """
    Simulates a series of feature values and a DataFrame of spot coordinates.

    Args:
        coordinates_centers (list[np.ndarray] | None): Cluster centers for coordinates.
        means_centers (list[float] | None): Mean feature values for each cluster.
        scales_centers (list[float] | None): Feature distribution scales for each cluster.
        var_noise (float | None): Variance of noise to add to feature values.
        var_name (str): Name of the feature variable.
        spot_names (list[str] | None): Spot names.
        non_negative (bool): Whether to set all negative feature values to 0. Defaults to False.

    Returns:
        tuple[pd.Series, pd.DataFrame]: Feature values and spot coordinates DataFrame.
    """
    # Defaults for clusters
    n_clusters = len(coordinates_centers) if coordinates_centers else np.random.randint(3, 7)
    total_spots = n_clusters * n_spots_per_cluster

    # Generate random cluster centers if not provided
    if coordinates_centers is None:
        coordinates_centers = [np.random.uniform(0, 20, size=2) for _ in range(n_clusters)]
    
    if means_centers is None:
        means_centers = [np.random.uniform(1, 20) for _ in range(n_clusters)]
    
    if scales_centers is None:
        scales_centers = [np.random.uniform(1, 5) for _ in range(n_clusters)]
    
    if var_noise is None:
        var_noise = np.random.uniform(0.5, 1)

    # Generate spot coordinates and feature values
    coordinates = []
    feature_values = []

    for i, center in enumerate(coordinates_centers):
        cluster_coords = np.random.normal(loc=center, scale=scales_centers[i], size=(n_spots_per_cluster, 2))
        cluster_features = means_centers[i] * np.exp(-((cluster_coords-center)**2).sum(1)/(2*scales_centers[i]**2))
        noise = np.random.normal(0, var_noise, size=n_spots_per_cluster)
        cluster_features += noise

        coordinates.append(cluster_coords)
        feature_values.extend(cluster_features)

    # Combine all clusters
    coordinates = np.vstack(coordinates)
    feature_values = np.array(feature_values)

    # Create spot names if not provided
    if spot_names is None:
        spot_names = [f"spot{i}" for i in range(1, total_spots+1)]

    # Create DataFrame and Series
    df_coord = pd.DataFrame(coordinates, columns=["x", "y"])
    df_coord.index = spot_names
    Y = pd.Series(feature_values, index=spot_names, name=var_name)
    if non_negative:
        Y[Y<0] = 0

    return Y, df_coord


# In[19]:


# Example usage
if __name__ == '__main__':    
    Y, df_coord = simulate(non_negative=True)
    print(Y.head())  # Feature values
    print(df_coord.head())  # Coordinates
    plot_raw(Y, df_coord)
    plt.colorbar()
    plt.show()


# In[ ]:





# In[ ]:




