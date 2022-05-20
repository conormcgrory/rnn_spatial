"""Script for training model."""

import os
from datetime import datetime

import torch

from trajectory import TrajectoryGenerator
from model import PathRNN
from trainer import Trainer
from utils import get_default_params, print_params, get_git_commit
from fileio import save_params, save_checkpoint, save_runinfo


# TODO: Make both of these parameters arguments!
# Directory where model is saved 
OUT_DIR = 'models/20220519_test'

# Set parameters
params = get_default_params()
params['trajectory']['rng_seed'] = 1993
params['trainer']['lambda_h'] = 4.0
params['n_epochs'] = 3
params['n_batches'] = 100


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


def main():

    print('all params:\n')
    print_params(params)
    
    print('initializing...')

    # Training setup for network
    tgen_train = TrajectoryGenerator(**params['trajectory'])
    model = PathRNN(**params['model'])
    trainer = Trainer(tgen_train, model, **params['trainer'])

    # Trajectory generator for testing
    tgen_test_params = dict(params['trajectory'], rng_seed=1989)
    tgen_test = TrajectoryGenerator(**tgen_test_params)

    # Create output directory and save parameters
    os.mkdir(OUT_DIR)
    save_params(OUT_DIR, params)

    print('done')

    mse_vals = []
    ts_start = datetime.now().isoformat()
    print('training network...')
    for epoch in range(params['n_epochs']):

        # Train network
        trainer.train(params['n_batches'])

        # Compute MSE on batch of test data
        mse = compute_mse(model, tgen_test, params['test_batch_size'])
        mse_vals.append(mse)

        # Save checkpoint
        save_checkpoint(OUT_DIR, trainer.model, epoch)

        print(f'epoch: {epoch:02d}')
        print(f'    mse: {mse}')

    ts_end = datetime.now().isoformat()
    print('done.')

    print('saving run info...')
    git_commit = get_git_commit()
    save_runinfo(OUT_DIR, git_commit, ts_start, ts_end, mse_vals)
    print('done.')


if __name__ == '__main__':
    main()
