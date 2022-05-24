"""Class for managing parameters of full training runs."""

import subprocess

import torch

from model import PathRNN
from trajectory import TrajectoryGenerator


def get_git_commit():
    """Return short version of current Git commit hash."""

    short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return str(short_hash, "utf-8").strip()


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