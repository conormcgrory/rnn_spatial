"""Script for sampling trajectories and saving to file"""

import os
import dataclasses

import numpy as np
import matplotlib.pyplot as plt
import json

import trajectory


# Output directory
DATA_DIR = 'data/test_01'

# Trajectory parameters
params = trajectory.TrajectoryParams(rng_seed=888)

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
        pdict = dataclasses.asdict(params)
        json.dump(pdict, f, indent=4)

    # Save velocity and position arrays to .npy files
    np.save(vel_fpath, vel)
    np.save(pos_fpath, pos)


def main():

    print('params:')
    print(dataclasses.asdict(params))
    print('')

    print('sampling trajectories...')
    tgen = trajectory.TrajectoryGenerator(params)
    vel, pos = tgen.smp_batch(N_TRIALS)
    print('done.\n')

    print('saving results...')
    save_results(params, vel, pos, DATA_DIR)
    print('done.\n')

if __name__ == '__main__':
    main()