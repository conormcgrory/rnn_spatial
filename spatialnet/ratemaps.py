"""Module for computing and visualizing tuning plots of hidden RNN units.

This code was adapted from Ganguli et al., 2019 
(https://github.com/ganguli-lab/grid-pattern-formation)

"""

import numpy as np

import torch
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d

from spatialnet.model import PathRNN
from spatialnet.trajectory import TrajectoryGenerator


def compute_ratemaps(model:PathRNN, vel: np.ndarray, pos: np.ndarray, res=20):
    """Compute ratemaps for given model."""

    # Run model on test batch and save hidden unit values
    _, h  = model.run_np(vel)

    # Combine all trials
    pos_all = np.reshape(pos, (-1, 2))
    h_all = np.reshape(h, (-1, model.n_units))

    # Compute activation estimates
    activations = binned_statistic_2d(
        pos_all[:, 0], pos_all[:, 1], h_all.T, statistic='mean', bins=res)[0]

    return activations, h


# TODO: Simplify this by only using one batch
def _compute_ratemaps_orig(model:PathRNN, tgen:TrajectoryGenerator, batch_size=200, n_batches=10, res=20):
    """Compute ratemaps for given model."""

    # Number of hidden units
    h_size = model.n_units

    # Number of total points in each batch
    n_pts = batch_size * tgen.n_steps

    # Hidden unit values
    h = np.zeros([n_batches, n_pts, h_size])

    # Position values
    pos = np.zeros([n_batches, n_pts, 2])

    for index in range(n_batches):

        # Sample test batch
        vel_batch, pos_batch = tgen.smp_batch(batch_size)
        
        # Run model on test batch and save hidden unit values
        _, h_batch = model.run_np(vel_batch)

        # Combine all trials from batch
        pos_batch = np.reshape(pos_batch, [-1, 2])
        h_batch = h_batch.reshape(-1, h_size)
        
        # Save activity and position values
        h[index] = h_batch
        pos[index] = pos_batch
               
    pos = pos.reshape([-1, 2])
    h = h.reshape([-1, h_size])

    activations = scipy.stats.binned_statistic_2d(pos[:, 0], pos[:, 1], h.T, bins=res)[0]

    return activations, pos, h


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


def convert_to_colormap(im, cmap):

    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def rgb(im, cmap='jet', smooth=True):

    cmap = plt.cm.get_cmap(cmap)
    np.seterr(invalid='ignore')
    im = (im - np.min(im)) / (np.max(im) - np.min(im))
    im = cmap(im)
    im = np.uint8(im * 255)
    return im


def plot_ratemaps(activations, n_plots, cmap='jet', smooth=True, width=16):

    images = [rgb(im, cmap, smooth) for im in activations[:n_plots]]
    rm_fig = concat_images_in_rows(images, n_plots//width, activations.shape[-1])
    return rm_fig


# Saving this because it computes histogram directly, instead of via SciPy. This
# might be necessary to compare against to make sure we're doing things right in
# our version. Eventually want to delete this if we're not using it.
def _compute_ratemaps_old(model:PathRNN, tgen:TrajectoryGenerator, batch_size=200, n_batches=10, res=20, idxs=None):

    # Check boundary shape and set boundary dimensions
    if tgen.boundary_shape != 'square':
        raise ValueError('Only square boundaries supported')
    b_width = tgen.boundary_height
    b_height = tgen.boundary_height

    # Number of hidden units
    h_size = model.n_units

    # Set indices
    if not np.any(idxs):
        idxs = np.arange(h_size)
    idxs = idxs[:h_size]

    # Number of total points in each batch
    n_pts = batch_size * tgen.n_steps

    # Hidden unit values
    h = np.zeros([n_batches, n_pts, h_size])

    # Position values
    pos = np.zeros([n_batches, n_pts, 2])

    # Hidden unit activations
    activations = np.zeros([h_size, res, res]) 

    # Number of nonzero values
    counts  = np.zeros([res, res])

    for index in range(n_batches):

        # Sample test batch
        vel_batch, pos_batch = tgen.smp_batch(batch_size)
        
        # Run model on test batch and save hidden unit values
        vel_batch_t = torch.Tensor(vel_batch)
        _, h_batch_t = model(vel_batch_t)
        h_batch = h_batch_t.detach().numpy()

        # Combine all trials from batch
        pos_batch = np.reshape(pos_batch, [-1, 2])
        h_batch = h_batch[:,:,idxs].reshape(-1, h_size)
        
        # Save activity and position values
        h[index] = h_batch
        pos[index] = pos_batch

        x_batch = (pos_batch[:,0] + b_width / 2) / b_width * res
        y_batch = (pos_batch[:,1] + b_height / 2) / b_height * res

        for i in range(n_pts):
            x = x_batch[i]
            y = y_batch[i]
            if x >=0 and x < res and y >=0 and y < res:
                counts[int(x), int(y)] += 1
                activations[:, int(x), int(y)] += h_batch[i, :]

    for x in range(res):
        for y in range(res):
            if counts[x, y] > 0:
                activations[:, x, y] /= counts[x, y]
                
    h = h.reshape([-1, h_size])
    pos = pos.reshape([-1, 2])

    return activations, pos, h