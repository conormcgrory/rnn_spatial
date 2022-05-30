"""Functions for visualizing results of model analysis."""

import numpy as np
import matplotlib.pyplot as plt

from spatialnet.trajectory import Boundary


## Trajectory plotting

def plot_position(boundary: Boundary, pos: np.ndarray, ax=None):

    if ax is None:
        ax = plt.gca()

    # Plot boundary
    boundary.plot(ax)

    # Add origin point to beginning of position sequence
    pos_x = np.concatenate(([0.0], pos[:, 0]))
    pos_y = np.concatenate(([0.0], pos[:, 1]))

    # Plot position values
    ax.plot(pos_x, pos_y)

    ax.set_aspect('equal')

def plot_position_estimate(boundary: Boundary, pos_true: np.ndarray, pos_est: np.ndarray, ax=None):

    if ax is None:
        ax = plt.gca()

    # Plot boundary
    boundary.plot(ax)

    # Add origin point to beginning of position sequences
    x_true = np.concatenate(([0.0], pos_true[:, 0]))
    y_true = np.concatenate(([0.0], pos_true[:, 1]))
    x_est = np.concatenate(([0.0], pos_est[:, 0]))
    y_est = np.concatenate(([0.0], pos_est[:, 1]))

    # Plot position sequences
    ax.plot(x_true, y_true, color='black', label='true')
    ax.plot(x_est, y_est, color='red', label='est')

    ax.set_aspect('equal')
    ax.legend()


## Activity plotting

def plot_activity(boundary: Boundary, pos: np.ndarray, hvals: np.ndarray, 
    threshold=0.2, n_trials=None, ax=None):

    if ax is None:
        ax = plt.gca()

    if n_trials is None:
        n_trials = pos.shape[0]

    # Plot boundary
    boundary.plot(ax)

    for t in range(n_trials):

        x = pos[t, :, 0]
        y = pos[t, :, 1]

        spk_idx = hvals[t, :] > threshold
        x_spk = x[spk_idx]
        y_spk = y[spk_idx]

        ax.plot(x, y, color='grey', alpha=0.2)
        ax.plot(x_spk, y_spk, 'k.')

    ax.set_aspect('equal')


## Ratemap plotting code adapted from Ganguli et al. 2019: (https://github.com/ganguli-lab/grid-pattern-formation)

def concat_images(images, image_width, spacer_size):
    """Concat image horizontally with spacer."""

    spacer = np.ones([image_width, spacer_size, 4], dtype=np.uint8) * 255
    images_with_spacers = []

    image_size = len(images)

    for i in range(image_size):
        images_with_spacers.append(images[i])
        if i != image_size - 1:
            # Add spacer
            images_with_spacers.append(spacer)
    ret = np.hstack(images_with_spacers)
    return ret

def concat_images_in_rows(images, row_size, image_width, spacer_size=4):
    """Concat images in rows"""

    column_size = len(images) // row_size
    spacer_h = np.ones([spacer_size, image_width*column_size + (column_size-1)*spacer_size, 4],
                       dtype=np.uint8) * 255

    row_images_with_spacers = []

    for row in range(row_size):
        row_images = images[column_size*row:column_size*row+column_size]
        row_concated_images = concat_images(row_images, image_width, spacer_size)
        row_images_with_spacers.append(row_concated_images)

        if row != row_size-1:
            row_images_with_spacers.append(spacer_h)

    ret = np.vstack(row_images_with_spacers)
    return ret

def rgb(im, cmap='jet'):

    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid='ignore')
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    im = cmap(im)
    im = np.uint8(im * 255)
    return im

def plot_ratemaps(activations: np.ndarray, n_plots: int, cmap='jet', smooth=True, width=16):

    images = [rgb(im, cmap, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig