import ctypes
import numpy as np
import pyastar.astar
from typing import Optional, Tuple, Union, List


# Define array types
ndmat_f_type = np.ctypeslib.ndpointer(
    dtype=np.float32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=1, flags="C_CONTIGUOUS")
ndmat_i2_type = np.ctypeslib.ndpointer(
    dtype=np.int32, ndim=2, flags="C_CONTIGUOUS")

# Define input/output types
pyastar.astar.restype = ndmat_i2_type  # Nx2 (i, j) coordinates or None
pyastar.astar.argtypes = [
    ndmat_f_type,   # weights
    ctypes.c_int,   # height
    ctypes.c_int,   # width
    ctypes.c_int,   # start index in flattened grid
    ctypes.c_int,   # goal index in flattened grid
    ctypes.c_bool,  # allow diagonal
]


def astar_path(
        weights: np.ndarray,
        start: Tuple[int, int],
        goals: List[Tuple[int, int]],
        allow_diagonal: bool = False) -> Union[np.ndarray, None]:
    # For the heuristic to be valid, each move must cost at least 1.
    if weights.min(axis=None) < 1.:
        raise ValueError("Minimum cost to move must be 1, but got %f" % (
            weights.min(axis=None)))
    # Ensure start is within bounds.
    if (start[0] < 0 or start[0] >= weights.shape[0] or
            start[1] < 0 or start[1] >= weights.shape[1]):
        raise ValueError(f"Start of {start} lies outside grid.")
    # Ensure goals are within bounds.
    for goal in goals:
        if (goal[0] < 0 or goal[0] >= weights.shape[0] or
                goal[1] < 0 or goal[1] >= weights.shape[1]):
            raise ValueError(f"Goal of {goal} lies outside grid.")

    height, width = weights.shape
    # Build indexes on flattened weights
    start_idx = np.ravel_multi_index(start, (height, width))
    goals = [[i[0] for i in goals], [i[1] for i in goals]]
    goals_idxs = np.ravel_multi_index(goals, (height, width))

    path = pyastar.astar.astar(
        weights.flatten(), height, width, start_idx, goals_idxs.astype(np.int32), goals_idxs.shape[0], allow_diagonal,
    )
    return path
