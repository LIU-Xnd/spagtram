#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from typing import List, Tuple, Union, NamedTuple, Literal
import matplotlib.pyplot as plt

from .utils import simulate


# In[2]:


# Type definition
class GradientType(NamedTuple):
    """Gradient type.
    Note: the starting point (and the ending point) could be at the highest(/lowest) end of
     the gradient vector, or at the lowest(/highest) end thereof. It depends on the
     `direction` property (-1 for at the highest(/lowest) end and 1 for at the lowest(/highest) end).
    
    Properties:
        coord_start (np.ndarray): coordinate [x, y] of the starting point
        direction_vector (np.ndarray): direction of regressed gradient (coeff) vector.
         Always indicates the increasing direction.
        length (float): length of the gradient vector
        coord_end (np.ndarray): coordinate [x, y] of the ending point. It must be
         on the window boundary.
        size_window (float): length of the edge of the sliding window (square shaped)
        R_squared (float): R squared of regression
        offset (Tuple[int, int]): the offset of the sub-window from the starting point.
         For example, (1,0) means right-center offset.
        direction (int): 1 for increasing, -1 for decreasing, and 0 for undefined.
    """
    coord_start: np.ndarray
    direction_vector: np.ndarray
    length: float
    coord_end: np.ndarray
    size_window: float
    R_squared: Union[float, None]
    offset: Tuple[int]
    @property
    def direction(self) -> int:
        if self.length < 1e-8:
            return 0
        delta = self.coord_end - self.coord_start
        if np.abs(delta[0]) > 1e-8:
            res = (delta[0] * self.direction_vector[0]) > 0
        else:
            assert np.abs(delta[1]) > 1e-8
            res = (delta[1] * self.direction_vector[1]) > 0
        if res:
            res = 1
        else:
            res = -1
        return res
    def __repr__(self):
        return f"""Gradient(R_squared: {self.R_squared}, direction: {self.direction_vector})"""

class TrajectoryType(NamedTuple):
    list_gradient: list[GradientType]
    direction: int
    coord_start: np.ndarray
    coord_end: np.ndarray
    @property
    def list_RSquared(self) -> List[Union[float, None]]:
        list_RSquared_ = []
        for grad in self.list_gradient:
            list_RSquared_.append(grad.R_squared)
        return list_RSquared_
    @property
    def n_gradients(self) -> int:
        return len(self.list_gradient)
    @property
    def RSquared_average(self) -> Union[float, None]:
        if None in (list_RSquared_:=self.list_RSquared):
            return None
        else:
            return np.mean(list_RSquared_)
    def __repr__(self):
        return f"""Trajectory(RSquared_average: {self.RSquared_average}, direction: {self.direction}, n_gradients: {self.n_gradients})"""


# In[3]:


def linear_regression_2d(
    X: Union[np.ndarray, pd.DataFrame],
    y: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Perform 2d linear regression and return R_squared and direction vector.
    """
    if type(X) is pd.DataFrame:
        X = X.values
    model = LinearRegression()
    model.fit(X, y)
    coeff = model.coef_
    intercept = model.intercept_
    R_squared = model.score(X, y)
    # a1*x1 + a2*x2 + b = y
    # ==> dy/dx = [a1, a2]
    # ==> direction = [a1, a2] / norm([a1, a2])
    if (norm_:=np.linalg.norm(coeff))>0:
        direction = coeff / norm_
    else:
        direction = np.zeros((2,))
    return (R_squared, direction)


# In[4]:


def find_coord_end(window_bounds: np.ndarray, coord_start: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """
    Find the coord_end given window bounds, coord_start, and direction.

    Parameters:
        window_bounds (np.ndarray): A 2x2 array defining the [min, max] bounds of the window 
                                   for each dimension (rows: dimensions, columns: [min, max]).
        coord_start (np.ndarray): A 1D array representing the starting coordinate.
        direction (np.ndarray): A 1D array representing the direction vector.

    Returns:
        coord_end (np.ndarray): The coordinate where the direction intersects the window boundary.
    """
    # Ensure direction is a unit vector (positive gradient)
    direction = direction / np.linalg.norm(direction)

    # Initialize coord_end as coord_start
    coord_end = np.copy(coord_start)

    # Calculate t-values for intersection with each boundary
    t_values = []
    for dim in range(coord_start.size):
        if direction[dim] != 0:  # Avoid division by zero
            t_min = (window_bounds[dim, 0] - coord_start[dim]) / direction[dim]
            t_max = (window_bounds[dim, 1] - coord_start[dim]) / direction[dim]
            t_values.extend([t_min, t_max])

    # Filter positive t-values (forward direction) and find the smallest
    t_values = [t for t in t_values if t > 0]
    if t_values:
        t_min_positive = min(t_values)
        coord_end = coord_start + t_min_positive * direction
    else: # Try negative direction
        direction = -direction
        # Calculate t-values for intersection with each boundary
        t_values = []
        for dim in range(coord_start.size):
            if direction[dim] != 0:  # Avoid division by zero
                t_min = (window_bounds[dim, 0] - coord_start[dim]) / direction[dim]
                t_max = (window_bounds[dim, 1] - coord_start[dim]) / direction[dim]
                t_values.extend([t_min, t_max])
    
        # Filter positive t-values (forward direction) and find the smallest
        t_values = [t for t in t_values if t > 0]
        if t_values:
            t_min_positive = min(t_values)
            coord_end = coord_start + t_min_positive * direction
    if (window_bounds[0,0]-1e-8 <= coord_end[0] <= window_bounds[0,1]+1e-8) and (window_bounds[1,0]-1e-8 <= coord_end[1] <= window_bounds[1,1]+1e-8):
        return coord_end
    else:
        return coord_start


# In[5]:


def _make_odd(int_: int) -> int:
    if int_ % 2 == 0:
        int_ = int_ + 1
    return int_


# In[6]:


# def _array_in(a: np.ndarray, list_of_arrays: List[np.ndarray]) -> bool:
#     for compared in list_of_arrays:
#         if np.all(a == compared):
#             return True
#     return False

def _tupleOfTwoArrays_in(toa: Tuple[np.ndarray], list_of_toas: List[Tuple[np.ndarray]]) -> int:
    for i, compared in enumerate(list_of_toas):
        if np.all(np.array(toa[0]) == np.array(compared[0])) and np.all(np.array(toa[1]) == np.array(compared[1])):
            return i # index
    return -1 # not found


# In[19]:


class GradientTrajectoryMapper:
    """Note: `size_window` will be set to a fixed value once a fit_transform is run.
     It can be modified by explicitly reclaiming it either to another value or 'infer'.
    
    The df_coord is dataframe with cols ['x','y'] which are expected to be integers.
     """
    def __init__(
        self,
        threshold_RSquared: float = 0.05,
        threshold_RSquared_switchWindowSize: float = 0.45,
        size_window: Union[float, Literal['infer']] = 'infer',
        sizeRatio_windowSmaller: float = 0.75,
        sizeRatio_windowLarger: float = 1.25, # larger first
        n_extraWindows_larger: int = 3,
        n_extraWindows_smaller: int = 3,
        max_iter: int = 30,
        n_points_enough_for_regression: int = 8,
    ):
        self.threshold_RSquared = threshold_RSquared
        self.threshold_RSquared_switchWindowSize = threshold_RSquared_switchWindowSize
        self.size_window = size_window
        self.sizeRatio_windowSmaller = sizeRatio_windowSmaller
        self.sizeRatio_windowLarger = sizeRatio_windowLarger
        self.n_extraWindows_larger = n_extraWindows_larger
        self.n_extraWindows_smaller = n_extraWindows_smaller
        # Sizes of windows in a list (needs initializing)
        self.sizes_windows = None
        self.max_iter = max_iter
        self.n_points_enough_for_regression = n_points_enough_for_regression
        
        self.trajectories = [] # traj1, traj2, ...
        # A trajectory is a (named) list of GradientType with annonated
        #  direction (1 for increase and -1 for decrease) and coord_start and coord_end,
        #  as well as some other statistics
        self.dataset_used = {
            'YAndCoords': [], # [(Y, df_coords), ...]
            'order_fit': [], # [0, 0, 1, 2, 2, 3, 4, ...]
        }
        return

    def __repr__(self) -> str:
        return f"""==== GradientTrajectoryMapper ====
- threshold_RSquared: {self.threshold_RSquared}
- threshold_RSquared_switchWIndowSize: {self.threshold_RSquared_switchWindowSize}
- size_window: {self.size_window}
    + larger: x{self.sizeRatio_windowLarger}
    + smaller: x{self.sizeRatio_windowSmaller}
    + n_extraWindows: {self.n_extraWindows_larger} larger, {self.n_extraWindows_smaller} smaller
- max_iter: {self.max_iter}
- n_points_enough_for_regression: {self.n_points_enough_for_regression}
- trajectories: {len(self.trajectories)} trajectories saved in total
- dataset_used:
    + {len(self.dataset_used['YAndCoords'])} sets of data used
    + {len(self.dataset_used['order_fit'])} fits performed
==== ==== ==== ==== ==== ===="""

    def _find_starting_gradient(self,
        Y: Union[np.ndarray, pd.Series],
        df_coord: pd.DataFrame,
        coord_start: np.ndarray,
    ) -> Union[GradientType, None]:
        # Use eight types of windows: from top-left to bottom-right except center
        # Each with candidate sizes: normal, larger, smaller
        # Process 2d linear regression
        # Find the optimal one (optimal R_squared)
        # Record the gradient as GradientType
        RSquared_optimal = 0.
        gradient_optimal = None
        offsets = [ # measured by x and y
            (-1, -1), (0, -1), (1, -1), # top-left, top-center, top-right
            (-1, 0), (1, 0), # mid-left, mid-right
            (-1, 1), (0, 1), (1, 1), # bot-left, bot-center, bot-right
        ]
        
        for size in self.sizes_windows:
            for offset in offsets:
                # Determine window bounds (all inclusive)
                halfsize = size // 2
                xmin_window = coord_start[0] + (offset[0] - 1) * halfsize
                xmax_window = coord_start[0] + (offset[0] + 1) * halfsize
                ymin_window = coord_start[1] + (offset[1] - 1) * halfsize
                ymax_window = coord_start[1] + (offset[1] + 1) * halfsize

                # Filter data points within the window
                mask = (df_coord['x'] >= xmin_window) & (df_coord['x'] <= xmax_window) & \
                       (df_coord['y'] >= ymin_window) & (df_coord['y'] <= ymax_window)
                points_in_window = df_coord[mask]
                values_in_window = Y[mask]
                
                # Skip if there aren't enough points for regression
                if len(points_in_window) < self.n_points_enough_for_regression:
                    continue

                # Perform 2d regression
                R_squared, direction = linear_regression_2d(
                    X=points_in_window,
                    y=values_in_window,
                )
                if R_squared < self.threshold_RSquared:
                    continue
                if R_squared > RSquared_optimal:
                    RSquared_optimal = R_squared
                    coord_end = find_coord_end(
                        window_bounds=np.array([
                            [xmin_window, xmax_window],
                            [ymin_window, ymax_window],
                        ]),
                        coord_start=coord_start,
                        direction=direction,
                    )
                    length = np.linalg.norm(coord_end-coord_start)
                    if length < 1e-8:
                        continue
                    gradient_optimal = GradientType(
                        coord_start=coord_start,
                        direction_vector=direction,
                        length=length,
                        coord_end=coord_end,
                        size_window=size,
                        R_squared=RSquared_optimal,
                        offset=offset,
                    )
            if RSquared_optimal > self.threshold_RSquared_switchWindowSize:
                # Do not continue switching windowSize if R_squared is OK
                break
        return gradient_optimal

    def fit_transform(
        self,
        Y: Union[np.ndarray, pd.Series],
        df_coord: pd.DataFrame,
        coord_start: Union[np.ndarray, Literal['random', 'highest', 'lowest']] = 'random',
    ) -> TrajectoryType:
        """
        Infers an optimal gradient trajectory from a starting coordinate.
        Each time this method is called, a new trajectory will be added to .results.
        
        Parameters:
            Y (np.ndarray | pd.Series): Feature values.
            df_coord (pd.DataFrame): Spot coordinates with columns ['x', 'y'].
            coord_start (np.ndarray | Literal['random', 'highest', 'lowest']): Starting coordinate. Randomly chosen if 'random'; the coordinate where
             Y reaches its highest if 'highest'; and that of lowest if 'lowest'.
        
        Returns:
            np.ndarray | None: The inferred trajectory this time. None for no results.
        """
        # Find coord_start
        if coord_start == 'random':
            coord_start = df_coord.sample(1).values.flatten()
        elif coord_start == 'highest':
            coord_start = df_coord.iloc[np.argmax(Y),:].values
        elif coord_start == 'lowest':
            coord_start = df_coord.iloc[np.argmin(Y),:].values
        else:
            assert coord_start.shape == (2,)
        
        # Assign size_window and so on
        xmin_sampleSpace, ymin_sampleSpace = df_coord.min(0)
        xmax_sampleSpace, ymax_sampleSpace = df_coord.max(0)
        length_x_sampleSpace = xmax_sampleSpace - xmin_sampleSpace
        length_y_sampleSpace = ymax_sampleSpace - ymin_sampleSpace
        if self.size_window == 'infer':
            # Pick one nineth of the sample space length
            self.size_window = int(np.round((1/9) * max(length_x_sampleSpace, length_y_sampleSpace)))
        self.size_window = _make_odd(self.size_window) # make it odd

        # Sizes of windows in a list
        sizes_windows: list = [self.size_window]
        for i_extra in range(1, self.n_extraWindows_larger+1):
            sizes_windows.append(_make_odd(int(self.size_window * self.sizeRatio_windowLarger ** i_extra)))
        for i_extra in range(1, self.n_extraWindows_smaller+1):
            sizes_windows.append(_make_odd(int(self.size_window * self.sizeRatio_windowSmaller ** i_extra)))
        self.sizes_windows = sizes_windows

        # Save dataset_used
        ix_dataset = _tupleOfTwoArrays_in((Y, df_coord), self.dataset_used['YAndCoords'])
        if ix_dataset == -1:
            self.dataset_used['YAndCoords'].append((Y, df_coord))
            self.dataset_used['order_fit'].append(len(self.dataset_used['YAndCoords'])-1)
        else:
            self.dataset_used['order_fit'].append(ix_dataset)
        
        # Gradient trajectory logic
        trajectory: List[GradientType]= [] # Trajectory
        current_point = coord_start
        
        # Find the starting gradient
        gradient_start = self._find_starting_gradient(
            Y=Y,
            df_coord=df_coord,
            coord_start=coord_start,
        )
        if gradient_start is None:
            trajectory = TrajectoryType(
                list_gradient=[],
                direction=0,
                coord_start=coord_start,
                coord_end=coord_start,
            )
            self.trajectories.append(trajectory)
            return trajectory
        trajectory.append(gradient_start)
        # Find consecutive gradients to make a trajectory
        for _ in range(self.max_iter):
            gradient_next = self._compute_next_gradient(trajectory[-1], df_coord, Y)
            if gradient_next is None:
                break
            trajectory.append(gradient_next)
        if trajectory:
            trajectory = TrajectoryType(
                list_gradient=trajectory,
                direction=trajectory[-1].direction,
                coord_start=trajectory[0].coord_start,
                coord_end=trajectory[-1].coord_end,
            )
        else:
            trajectory = TrajectoryType(
                list_gradient=[],
                direction=0,
                coord_start=coord_start,
                coord_end=coord_start,
            )
        self.trajectories.append(trajectory)
        return trajectory

    def _compute_next_gradient(self,
            last_gradient: GradientType,
            df_coord: pd.DataFrame,
            Y: pd.Series | np.ndarray,
    ):
        """
        Computes the next gradient given last gradient.
        """
        # Must keep direction consistent
        RSquared_optimal = 0.
        gradient_optimal = None
        offsets = [ # measured by x and y
            (-1, -1), (0, -1), (1, -1), # top-left, top-center, top-right
            (-1, 0), (1, 0), # mid-left, mid-right
            (-1, 1), (0, 1), (1, 1), # bot-left, bot-center, bot-right
        ]
        coord_start = last_gradient.coord_end
        for size in self.sizes_windows:
            for offset in offsets:
                # Avoid U-turn
                offset_uTurn_direct = (-last_gradient.offset[0], -last_gradient.offset[1])
                if offset == offset_uTurn_direct:
                    continue
            
                # Determine window bounds (all inclusive)
                halfsize = size // 2
                xmin_window = coord_start[0] + (offset[0] - 1) * halfsize
                xmax_window = coord_start[0] + (offset[0] + 1) * halfsize
                ymin_window = coord_start[1] + (offset[1] - 1) * halfsize
                ymax_window = coord_start[1] + (offset[1] + 1) * halfsize

                # Filter data points within the window
                mask = (df_coord['x'] >= xmin_window) & (df_coord['x'] <= xmax_window) & \
                       (df_coord['y'] >= ymin_window) & (df_coord['y'] <= ymax_window)
                points_in_window = df_coord[mask]
                values_in_window = Y[mask]
                
                # Skip if there aren't enough points for regression
                if len(points_in_window) < self.n_points_enough_for_regression:
                    continue

                # Perform 2d regression
                R_squared, direction = linear_regression_2d(
                    X=points_in_window,
                    y=values_in_window,
                )
                if R_squared < self.threshold_RSquared:
                    continue
                if R_squared > RSquared_optimal:
                    coord_end = find_coord_end(
                        window_bounds=np.array([
                            [xmin_window, xmax_window],
                            [ymin_window, ymax_window],
                        ]),
                        coord_start=coord_start,
                        direction=direction,
                    )
                    length = np.linalg.norm(coord_end-coord_start)
                    if length < 1e-8:
                        continue
                    gradient_ = GradientType(
                        coord_start=coord_start,
                        direction_vector=direction,
                        length=length,
                        coord_end=coord_end,
                        size_window=size,
                        R_squared=R_squared,
                        offset=offset,
                    )
                    if last_gradient.direction != gradient_.direction:
                        continue
                    gradient_optimal = gradient_
                    RSquared_optimal = R_squared
            if RSquared_optimal > self.threshold_RSquared_switchWindowSize:
                # Do not continue switching windowSize if R_squared is OK
                break
        return gradient_optimal

    def plot(self,
             index: int = -1):
        """
        Plots the inferred trajectories over the `index`-th data's spatial coordinates (see
         `.dataset_used['order_fit']` for the index of the dataset used in the certain fit).
         `index=-1` for plotting the last used data's trajectories.
        """
        if self.trajectories == []:
            raise ValueError("No trajectory to plot. Run fit_transform first.")
        
        assert type(index) is int
        if index==-1:
            index = self.dataset_used['order_fit'][-1]
        ix_fits_usedSameData = (np.array(self.dataset_used['order_fit'])==index)
        ix_fits_usedSameData = np.where(ix_fits_usedSameData)[0]
        Y, df_coord = self.dataset_used['YAndCoords'][index]
        plt.scatter(df_coord['x'], df_coord['y'], c=Y, s=5, alpha=0.7, cmap='jet', label='Spots')
        for i in ix_fits_usedSameData:
            traj: TrajectoryType = self.trajectories[i]
            coords = []
            for grad in traj.list_gradient:
                coords.append(grad.coord_start)
            coords.append(traj.coord_end)
            coords = np.array(coords)
            ax = plt.plot(coords[:,0], coords[:,1], label=f'Traj {i}: ave_$R^2$={traj.RSquared_average:.4f}')
        plt.legend(loc='upper left', bbox_to_anchor=(1.3, 1), borderaxespad=0.)
        plt.colorbar(label='Feature Intensity')
        return ax


# In[21]:


# Example
if __name__ == '__main__':
    Y, df_coord = simulate(n_spots_per_cluster=256)
    print(Y.head())
    print(df_coord.head())
    plt.figure(figsize=(10,10))
    trajmapper = GradientTrajectoryMapper(threshold_RSquared_switchWindowSize=0.6, n_points_enough_for_regression=10)
    trajmapper.fit_transform(Y=Y, df_coord=df_coord, coord_start='highest')
    trajmapper.fit_transform(Y=Y, df_coord=df_coord, coord_start='lowest')
    for _ in range(3):
        trajmapper.fit_transform(Y=Y, df_coord=df_coord, coord_start='random')
    trajmapper.plot()
    print(trajmapper)
    print(trajmapper.trajectories)


# In[ ]:




