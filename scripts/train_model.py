"""Script for training model."""

import os
import argparse
from datetime import datetime

from spatialnet import parameters
from spatialnet.trajectory import TrajectoryGenerator
from spatialnet.model import PathRNN
from spatialnet.trainer import Trainer
from spatialnet.metrics import compute_mse
from spatialnet.fileio import save_params, save_checkpoint, save_runinfo
from spatialnet.utils import get_git_commit


def parse_args() -> tuple[str, dict]:
    """Parse output directory and parameters from command-line args."""

    # Parser object
    parser = argparse.ArgumentParser(
        description='Train RNN on path integration task.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    required_group = parser.add_argument_group('required')
    param_group = parser.add_argument_group('parameters')

    # Directory to store results in
    required_group.add_argument(
        '-o', '--out-dir', type=str, required=True,
        help='Directory to store results in'
    )

    # Global parameters
    param_group.add_argument(
        '--num-epochs', type=int, 
        default=parameters.NUM_EPOCHS, 
        help='Number of epochs to train for'
    )
    param_group.add_argument(
        '--num-batches', type=int, 
        default=parameters.NUM_BATCHES,
        help='Number of batches per epoch'
    )
    param_group.add_argument(
        '--test-batch-size', type=int, 
        default=parameters.TEST_BATCH_SIZE, 
        help='Size of batch used to compute test error'
    )

    # Trajectory parameters
    param_group.add_argument(
        '--traj-rng-seed', type=int, 
        default=parameters.TRAJ_RNG_SEED,
        help='Random seed for trajectory generation'
    )
    param_group.add_argument(
        '--traj-n-steps', type=int, 
        default=parameters.TRAJ_N_STEPS,
        help='Number of steps per trajectory'
    )
    param_group.add_argument(
        '--traj-boundary-shape', type=str, choices=['square'],
        default=parameters.TRAJ_BOUNDARY_SHAPE,
        help='Shape of boundary for trajectories'
    )
    param_group.add_argument(
        '--traj-boundary-height', type=float,
        default=parameters.TRAJ_BOUNDARY_HEIGHT,
        help='Height of boundary for trajectories'
    )
    param_group.add_argument(
        '--traj-time-step', type=float,
        default=parameters.TRAJ_TIME_STEP, 
        help='Time step for trajectories'
    )
    param_group.add_argument(
        '--traj-std-norm', type=float,
        default=parameters.TRAJ_STD_NORM,
        help='Standard deviation of trajectory angle after 1 sec'
    )
    param_group.add_argument(
        '--traj-mean-speed', type=float,
        default=parameters.TRAJ_MEAN_SPEED,
        help='Mean speed of trajectory'
    )
    param_group.add_argument(
        '--traj-coordinates', type=str, choices=['cartesian', 'polar'],
        default=parameters.TRAJ_COORDINATES,
        help='Coordinate system used as input to network')

    # Model parameters
    param_group.add_argument(
        '--model-n-units', type=int,
        default=parameters.MODEL_N_UNITS,
        help='Number of hidden units in RNN'
    )
    param_group.add_argument(
        '--model-rnn-bias', action=argparse.BooleanOptionalAction,
        default=parameters.MODEL_RNN_BIAS,
        help='RNN has bias term for recurrent units'
    )
    param_group.add_argument(
        '--model-output-bias', action=argparse.BooleanOptionalAction,
        default=parameters.MODEL_OUTPUT_BIAS,
        help='RNN has bias term for output units'
    )

    # Training parameters
    param_group.add_argument(
        '--train-batch-size', type=int, 
        default=parameters.TRAIN_BATCH_SIZE,
        help='Batch size used for training model'
    )
    param_group.add_argument(
        '--train-lambda-w', type=float,
        default=parameters.TRAIN_LAMBDA_W,
        help='Coefficient of L2 penalty on network weights'
    )
    param_group.add_argument(
        '--train-lambda-h', type=float,
        default=parameters.TRAIN_LAMBDA_H,
        help='Coefficient of L2 penalty on network activity'
    )
    param_group.add_argument(
        '--train-learning-rate', type=float,
        default=parameters.TRAIN_LEARNING_RATE,
        help='Learning rate used for training.'
    )

    # Parse arguments from command-line
    args = parser.parse_args()

    # Create parameter dict from arguments
    params = {
        'num_epochs': args.num_epochs,
        'num_batches': args.num_batches,
        'test_batch_size': args.test_batch_size,
        'trajectory': {
            'rng_seed': args.traj_rng_seed,
            'n_steps': args.traj_n_steps,
            'boundary_shape': args.traj_boundary_shape,
            'boundary_height': args.traj_boundary_height,
            'time_step': args.traj_time_step,
            'std_norm': args.traj_std_norm,
            'mean_speed': args.traj_mean_speed,
            'coordinates': args.traj_coordinates,
        },
        'model': {
            'n_units': args.model_n_units,
            'rnn_bias': args.model_rnn_bias,
            'output_bias': args.model_output_bias,
        },
        'trainer': {
            'batch_size': args.train_batch_size,
            'lambda_w': args.train_lambda_w,
            'lambda_h': args.train_lambda_h,
            'learning_rate': args.train_learning_rate,
        },
    }

    return args.out_dir, params


def main():

    # Parse and validate cli arguments
    output_dir, params = parse_args()
    if os.path.exists(output_dir):
        print(f'error: directory "{output_dir}" already exists.')
        return
    if not parameters.validate_params(params):
        print('error: parameters invalid')
        return

    print('all params:\n')
    parameters.print_params(params)

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
