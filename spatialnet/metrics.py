"""Functions for measuring model performance and tuning."""

import torch
import numpy as np
from scipy.stats import binned_statistic_2d

from spatialnet.model import PathRNN
from spatialnet.trajectory import TrajectoryGenerator


def compute_mse(model: PathRNN, tgen: TrajectoryGenerator, test_batch_size: int):
    """Compute mean squared error of model on trajectory batch."""

    # Sample batch from trajectory generator
    vel_np, pos_np = tgen.smp_batch(test_batch_size)
    vel = torch.Tensor(vel_np)
    pos = torch.Tensor(pos_np)

    # Predict position for batch
    pos_est, _ = model(vel)

    # TODO: Find a way to do this without creating object every time
    # Compute MSE
    mse_loss = torch.nn.MSELoss()
    return mse_loss(pos, pos_est).item()


def compute_ratemaps(model: PathRNN, vel: np.ndarray, pos: np.ndarray, res=20):
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