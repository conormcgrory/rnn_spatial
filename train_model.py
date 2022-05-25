"""Script for training model."""

import os
from datetime import datetime

from trajectory import TrajectoryGenerator
from model import PathRNN
from trainer import Trainer
from parameters import parse_args, print_params
from fileio import save_params, save_checkpoint, save_runinfo
from utils import get_git_commit, compute_mse


def main():

    output_dir, params = parse_args()

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
    os.mkdir(output_dir)
    save_params(output_dir, params)

    print('done')

    mse_vals = []
    ts_start = datetime.now().isoformat()
    print('training network...')
    for epoch in range(1, params['num_epochs'] + 1):

        # Train network
        trainer.train(params['num_batches'])

        # Compute MSE on batch of test data
        mse = compute_mse(model, tgen_test, params['test_batch_size'])
        mse_vals.append(mse)

        # Save checkpoint
        save_checkpoint(output_dir, trainer.model, epoch)

        print(f'epoch: {epoch:02d}')
        print(f'    mse: {mse}')

    ts_end = datetime.now().isoformat()
    print('done.')

    print('saving run info...')
    runinfo = {
        'commit': get_git_commit(),
        'mse_vals': mse_vals,
        'time_started': ts_start,
        'time_finished': ts_end,
        'loss_mse': trainer.loss_mse,
        'loss_w': trainer.loss_w,
        'loss_h': trainer.loss_h,
        'loss_total': trainer.loss_total,
    }
    save_runinfo(output_dir, runinfo)
    print('done.')


if __name__ == '__main__':
    main()
