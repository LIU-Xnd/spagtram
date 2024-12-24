#!/usr/bin/env python
# coding: utf-8

# In[50]:


import pandas as pd
import numpy as np
from typing import List, Tuple, Union, NamedTuple, Literal
import matplotlib.pyplot as plt
from statsmodels.nonparametric.smoothers_lowess import lowess

from .spagtram import GradientType, TrajectoryType, GradientTrajectoryMapper, simulate


# In[20]:


# LOGIC - 2024.11.27
# 1. Given a trajectory, reads the gradients therein. For each gradient,
# 2. uses the coord_start, size_window and offset properties to
#  a) determine the window region;
#  b) project all points in the window (YAndCoords) onto the line specified by coord_start, direction_vector, and length
#   and get the projected position (x_proj) and sorted (by x_proj) feature intensities (y_proj).
# 3. Records all projections (i.e., ProjectionType(x_proj, y_proj, length_total, ix_sorted))
# 4. Plot along the trajectory (sums them up). ProjectionType(x_proj_concat, y_proj_concat, length_total, ix_sorted_concat).


# In[21]:


# Type definition
class ProjectionType(NamedTuple):
    x_proj: Union[np.ndarray]
    y_proj: Union[np.ndarray]
    length_total: float
    indices: Union[np.ndarray]
    def __add__(self, other):
        if not isinstance(other, ProjectionType):
            return NotImplemented
        return ProjectionType(
            x_proj=np.concatenate([self.x_proj, other.x_proj + self.length_total]),
            y_proj=np.concatenate([self.y_proj, other.y_proj]),
            length_total=self.length_total + other.length_total,
            indices=np.concatenate([self.indices, other.indices])
        )
    def __repr__(self):
        return f"Projection(length: {self.length_total}, n_points: {len(self.x_proj)})"

# Example usage:
if __name__ == '__main__':
    v1 = ProjectionType(np.array([0.,1.,2.]),np.array([0.,1.,2.]),2.,np.array([0,1,2]))
    v2 = ProjectionType(np.array([0.,1.,2.]),np.array([0.,1.,2.]),2.,np.array([3,4,5]))
    v3 = v1 + v2  # Uses __add__
    print(f'v1: {v1}, v2: {v2}')
    print('v3:', v3)


# In[35]:


def _project_forOneGradient(
    df_coords: pd.DataFrame,
    Y: np.ndarray,
    gradient: GradientType,
) -> ProjectionType:
    # Determine the window region
    halfsize = gradient.size_window // 2
    xmin_window = gradient.coord_start[0] + (gradient.offset[0] - 1) * halfsize
    xmax_window = gradient.coord_start[0] + (gradient.offset[0] + 1) * halfsize
    ymin_window = gradient.coord_start[1] + (gradient.offset[1] - 1) * halfsize
    ymax_window = gradient.coord_start[1] + (gradient.offset[1] + 1) * halfsize
    # Filter data points within the window
    mask = (df_coords['x'] >= xmin_window) & (df_coords['x'] <= xmax_window) & \
           (df_coords['y'] >= ymin_window) & (df_coords['y'] <= ymax_window)
    ix_within = np.where(np.array(mask))[0]
    points_in_window = df_coords[mask]
    values_in_window = np.array(Y)[ix_within]
    R = points_in_window - gradient.coord_start # relative coords of each points
    x_proj = np.array(
        R.dot((gradient.direction_vector/np.linalg.norm(gradient.direction_vector)) * gradient.direction)
    )
    ix_sorted = np.argsort(x_proj)
    ix_within_sorted = ix_within[ix_sorted]
    return ProjectionType(
        x_proj=x_proj[ix_sorted],
        y_proj=values_in_window[ix_sorted],
        length_total=gradient.length,
        indices=ix_within_sorted
    )
    


# In[75]:


class TrajectoryProjector:
    def __init__(
        self,
        gradient_trajectory_mapper: GradientTrajectoryMapper
    ):
        self.gradient_trajectory_mapper = gradient_trajectory_mapper
        self.projections: List[ProjectionType] = [] # proj1, proj2, ... (for traj1, traj2, ...)
        return

    def __repr__(self) -> str:
        projs_valid = [True for proj in self.projections if proj.length_total > 1e-8]
        return f"""==== TrajectoryProjector ====
- bound gradient trajectory mapper:
{self.gradient_trajectory_mapper}
- projections:
    + {len(self.projections)} projected
    + {len(projs_valid)}/{len(self.projections)} of them are valid (with length>0)
==== ==== ==== ==== ==== ===="""

    def fit(self) -> None:
        self.projections = []
        for i_traj, traj in enumerate(self.gradient_trajectory_mapper.trajectories):
            i_data = self.gradient_trajectory_mapper.dataset_used['order_fit'][i_traj]
            list_projs = []
            for i_grad, grad in enumerate(traj.list_gradient):
                list_projs.append(
                    _project_forOneGradient(
                        df_coords=self.gradient_trajectory_mapper.dataset_used['YAndCoords'][i_data][1],
                        Y=self.gradient_trajectory_mapper.dataset_used['YAndCoords'][i_data][0],
                        gradient=grad
                    )
                )
            if len(list_projs) == 0:
                self.projections.append(ProjectionType(
                    x_proj=np.array([]),
                    y_proj=np.array([]),
                    length_total=0,
                    indices=np.array([])
                ))
                continue
            res_i = list_projs[0]
            for proj_ in list_projs[1:]:
                res_i = res_i + proj_
            self.projections.append(res_i)
        return
    def transform(self) -> List[ProjectionType]:
        return self.projections.copy()
    def fit_transform(self) -> List[ProjectionType]:
        self.fit()
        return self.transform()

    def plot(
        self,
        index: int | None = None,
        scaling_positions: bool = True,
        plot_loess: Literal['raw-only', 'loess-only', 'both'] = 'both',
        frac_loess: float = 0.3,
        alpha: float = 0.6,
    ):
        """
        Plots the projection of the `index`-th trajectory's.
         `index=-1` for plotting the last projection.
         `index=None` for plotting all.
         `scaling_positions=True` for scaling all position ranges to [0,1].
        Only valid projections are plotted.
        """
        assert plot_loess in ['raw-only', 'loess-only', 'both']
        if self.projections == []:
            raise ValueError("No projection to plot. Run fit_transform first.")
        if index is None:
            for i_proj, proj in enumerate(self.projections):
                if proj.length_total < 1e-8:
                    continue
                if scaling_positions:
                    xs = np.linspace(0,1,num=len(proj.y_proj))
                else:
                    xs = proj.x_proj
                if plot_loess == 'raw-only':
                    plt.plot(xs, proj.y_proj, color=plt.cm.tab10.colors[i_proj%10], label=f"Proj {i_proj}", alpha=alpha)
                else:
                    smoothed = lowess(proj.y_proj, xs, frac=frac_loess)
                    plt.plot(smoothed[:,0], smoothed[:,1], '--', color=plt.cm.tab10.colors[i_proj%10], label=f"Proj {i_proj} loess", alpha=alpha)
                    if plot_loess == 'both':
                        plt.plot(xs, proj.y_proj, color=plt.cm.tab10.colors[i_proj%10], label=f"Proj {i_proj} raw", alpha=alpha)
        else:
            assert type(index) is int
            if scaling_positions:
                xs = np.linspace(0,1,num=len(self.projections[index].y_proj))
            else:
                xs = self.projections[index].x_proj
            if plot_loess == 'raw-only':
                lt.plot(xs, self.projections[index].y_proj, label=f"Proj {index}")
            else:
                smoothed = lowess(self.projections[index].y_proj, xs, frac=frac_loess)
                plt.plot(smoothed[:,0], smoothed[:,1], '--', label=f"Proj {index} loess")
                if plot_loess == 'both':
                    plt.plot(xs, self.projections[index].y_proj, label=f"Proj {index} raw")
        plt.legend(loc='upper left', bbox_to_anchor=(1.05, 1), borderaxespad=0.)
        return plt.gca()
        


# In[78]:


if __name__ == '__main__':
    plt.figure(figsize=(10,10))
    Y, df_coords = simulate(n_spots_per_cluster=200)
    traj_mapper = GradientTrajectoryMapper(n_extraWindows=4)
    traj_mapper.fit_transform(Y, df_coords, 'highest')
    [traj_mapper.fit_transform(Y, df_coords) for _ in range(4)]
    print(traj_mapper)
    traj_mapper.plot()
    plt.show()
    traj_projector = TrajectoryProjector(gradient_trajectory_mapper=traj_mapper)
    print(traj_projector)
    traj_projector.fit()
    traj_projector.plot(plot_loess='loess-only')
    plt.show()
    print(traj_projector)


# In[ ]:




