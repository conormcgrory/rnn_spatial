"""Class for managing parameters of full training runs."""

import subprocess
import yaml


def get_default_params():
    """Default parameter values for RNN training."""

    return {
        'trajectory': {
            'rng_seed': 999,
            'n_steps': 450,
            'boundary_shape': 'square',
            'boundary_height': 2.0,
            'time_step': 0.1,
            'std_norm': 0.5,
            'mean_speed': 0.2,
            'coordinates': 'cartesian',
        },
        'model': {
            'n_units': 100,
            'rnn_bias': True,
            'output_bias': False,
        },
        'trainer': {
            'batch_size': 500,
            'lambda_w': 0.5,
            'lambda_h': 0.1,
            'learning_rate': 1e-4,
        },
        'n_epochs': 3,
        'n_batches': 100,
        'test_batch_size': 500
    }

def print_params(params: dict):
    """Print parameter values."""

    print(yaml.dump(params, default_flow_style=False))

def get_git_commit():
    """Return short version of current Git commit hash."""

    short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return str(short_hash, "utf-8").strip()
