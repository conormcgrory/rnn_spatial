"""Script for sampling trajectories and saving to file"""

import os

import numpy as np
import matplotlib.pyplot as plt
import json

import parameters
import trajectory


# Output directory
DATA_DIR = 'data/test_02'

# Trajectory parameters
params = parameters.get_default_params()['trajectory']

# Number of trials to run simulation for
N_TRIALS = 2000


def save_results(params, vel, pos, dirpath):

    # Create directory
    os.mkdir(dirpath)

    # Filepaths for params, velocity and position arraysj
    params_fpath = os.path.join(dirpath, 'params.json')
    vel_fpath = os.path.join(dirpath, 'vel.npy')
    pos_fpath = os.path.join(dirpath, 'pos.npy')

    # Save parameters to JSON file
    with open(params_fpath, 'w') as f:
        json.dump(params, f, indent=4)

    # Save velocity and position arrays to .npy files
    np.save(vel_fpath, vel)
    np.save(pos_fpath, pos)


def main():

    print('params:')
    params.print_params(params)
    print('')

    print('sampling trajectories...')
    tgen = trajectory.TrajectoryGenerator(**params)
    vel, pos = tgen.smp_batch(N_TRIALS)
    print('done.\n')

    print('saving results...')
    save_results(params, vel, pos, DATA_DIR)
    print('done.\n')

if __name__ == '__main__':
    main()