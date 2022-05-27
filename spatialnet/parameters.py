"""Class for managing parameters of full training runs."""

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

def validate_params(params: dict) -> bool:
    """Return True if params are valid, False if not."""

    dparams = get_default_params()

    if params.keys() != dparams.keys():
        return False
    elif params['trajectory'].keys() != dparams['trajectory'].keys():
        return False
    elif params['model'].keys() != dparams['model'].keys():
        return False
    elif params['trainer'].keys() != dparams['trainer'].keys():
        return False
    else:
        return True