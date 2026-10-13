"""Small display-only helper for the book's coordinates; no learning or sampling."""

import matplotlib.pyplot as plt
import numpy as np

from modular_rl.environment import DLFS4_GRID_SHAPE, DLFS4_STATES


def plot_grid(values, policy=None, *, title="", ax=None, fmt=".2f"):
    """Display compact state values and maximum-probability policy arrows."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4.5, 3.5))
    grid = np.full(DLFS4_GRID_SHAPE, np.nan)
    for state, coordinate in enumerate(DLFS4_STATES):
        grid[coordinate] = values[state]
    cmap = plt.get_cmap("RdYlGn").copy()
    cmap.set_bad("#555555")
    image = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap)
    for state, (row, col) in enumerate(DLFS4_STATES):
        label = format(values[state], fmt)
        if policy is not None and (row, col) != (0, 3):
            label += "\n" + "".join(arrow for action, arrow in enumerate(("↑", "↓", "←", "→"))
                                   if policy[state, action] == policy[state].max())
        ax.text(col, row, label, ha="center", va="center", fontsize=10)
    for coordinate, label in (((2, 0), "START"), ((0, 3), "GOAL +1"), ((1, 3), "R=-1")):
        row, col = coordinate
        ax.text(col - .43, row - .30, label, fontsize=8)
    rows, cols = DLFS4_GRID_SHAPE
    ax.set(title=title, xlabel="column", ylabel="row", xticks=range(cols), yticks=range(rows))
    ax.set_xticks(np.arange(cols + 1) - .5, minor=True)
    ax.set_yticks(np.arange(rows + 1) - .5, minor=True)
    ax.grid(which="minor", color="white", linewidth=.7)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.figure.colorbar(image, ax=ax, shrink=.75)
    return ax
