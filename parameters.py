"""Class for managing parameters of full training runs."""

import argparse

import yaml


# Default parameter values
NUM_EPOCHS = 8
NUM_BATCHES = 1000
TEST_BATCH_SIZE = 500
TRAJ_RNG_SEED = 999
TRAJ_N_STEPS = 450
TRAJ_BOUNDARY_SHAPE = 'square'
TRAJ_BOUNDARY_HEIGHT = 2.0
TRAJ_TIME_STEP = 0.1
TRAJ_STD_NORM = 0.5
TRAJ_MEAN_SPEED = 0.2
TRAJ_COORDINATES = 'cartesian'
MODEL_N_UNITS = 100
MODEL_RNN_BIAS = True
MODEL_OUTPUT_BIAS = False
TRAIN_BATCH_SIZE = 500
TRAIN_LAMBDA_W = 0.5
TRAIN_LAMBDA_H = 0.1
TRAIN_LEARNING_RATE = 1e-4


def get_default_params() -> dict:
    """Default parameter values for RNN training."""

    return {
        'num_epochs': NUM_EPOCHS,
        'num_batches': NUM_BATCHES,
        'test_batch_size': TEST_BATCH_SIZE,
        'trajectory': {
            'rng_seed': TRAJ_RNG_SEED,
            'n_steps': TRAJ_N_STEPS,
            'boundary_shape': TRAJ_BOUNDARY_SHAPE,
            'boundary_height': TRAJ_BOUNDARY_HEIGHT,
            'time_step': TRAJ_TIME_STEP,
            'std_norm': TRAJ_STD_NORM,
            'mean_speed': TRAJ_MEAN_SPEED,
            'coordinates': TRAJ_COORDINATES,
        },
        'model': {
            'n_units': MODEL_N_UNITS,
            'rnn_bias': MODEL_RNN_BIAS,
            'output_bias': MODEL_OUTPUT_BIAS,
        },
        'trainer': {
            'batch_size': TRAIN_BATCH_SIZE,
            'lambda_w': TRAIN_LAMBDA_W,
            'lambda_h': TRAIN_LAMBDA_H,
            'learning_rate': TRAIN_LEARNING_RATE,
        },
    }


def print_params(params: dict):
    """Print parameter values."""

    print(yaml.dump(params, default_flow_style=False))


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
        default=NUM_EPOCHS, 
        help='Number of epochs to train for'
    )
    param_group.add_argument(
        '--num-batches', type=int, 
        default=NUM_BATCHES,
        help='Number of batches per epoch'
    )
    param_group.add_argument(
        '--test_batch_size', type=int, 
        default=TEST_BATCH_SIZE, 
        help='Size of batch used to compute test error'
    )

    # Trajectory parameters
    param_group.add_argument(
        '--traj-rng-seed', type=int, 
        default=TRAJ_RNG_SEED,
        help='Random seed for trajectory generation'
    )
    param_group.add_argument(
        '--traj-n-steps', type=int, 
        default=TRAJ_N_STEPS,
        help='Number of steps per trajectory'
    )
    param_group.add_argument(
        '--traj-boundary-shape', type=str, choices=['square'],
        default=TRAJ_BOUNDARY_SHAPE,
        help='Shape of boundary for trajectories'
    )
    param_group.add_argument(
        '--traj-boundary-height', type=float,
        default=TRAJ_BOUNDARY_HEIGHT,
        help='Height of boundary for trajectories'
    )
    param_group.add_argument(
        '--traj-time-step', type=float,
        default=TRAJ_TIME_STEP, 
        help='Time step for trajectories'
    )
    param_group.add_argument(
        '--traj-std-norm', type=float,
        default=TRAJ_STD_NORM,
        help='Standard deviation of trajectory angle after 1 sec'
    )
    param_group.add_argument(
        '--traj-mean-speed', type=float,
        default=TRAJ_MEAN_SPEED,
        help='Mean speed of trajectory'
    )
    param_group.add_argument(
        '--traj-coordinates', type=str, choices=['cartesian', 'polar'],
        default=TRAJ_COORDINATES,
        help='Coordinate system used as input to network')

    # Model parameters
    param_group.add_argument(
        '--model-n-units', type=int,
        default=MODEL_N_UNITS,
        help='Number of hidden units in RNN'
    )
    param_group.add_argument(
        '--model-rnn-bias', action=argparse.BooleanOptionalAction,
        default=MODEL_RNN_BIAS,
        help='RNN has bias term for recurrent units'
    )
    param_group.add_argument(
        '--model-output-bias', action=argparse.BooleanOptionalAction,
        default=MODEL_OUTPUT_BIAS,
        help='RNN has bias term for output units'
    )

    # Training parameters
    param_group.add_argument(
        '--train-batch-size', type=int, 
        default=TRAIN_BATCH_SIZE,
        help='Batch size used for training model'
    )
    param_group.add_argument(
        '--train-lambda-w', type=float,
        default=TRAIN_LAMBDA_W,
        help='Coefficient of L2 penalty on network weights'
    )
    param_group.add_argument(
        '--train-lambda-h', type=float,
        default=TRAIN_LAMBDA_H,
        help='Coefficient of L2 penalty on network activity'
    )
    param_group.add_argument(
        '--train-learning-rate', type=float,
        default=TRAIN_LEARNING_RATE,
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