"""Script for sampling trajectories"""

import numpy as np
import matplotlib.pyplot as plt

import motion


# Output filepath
DATA_FPATH = 'data/sim_2022_03_29.npz'

# Simulation parameters
N_STEPS = 450
N_TRIALS = 950000
BOUNDARY_TYPE = 'square'
BOUNDARY_HEIGHT = 2.0
TIME_STEP = 1.0
STD_NORM = 0.33

# Need to change to match Chris's code!
SPEED_VALS = [0.0, 0.1, 0.2, 0.3, 0.4]
P_SPEED = [0.8, 0.05, 0.05, 0.05, 0.05]

RNG_SEED = 999


def print_params():

    print('filepaths:')
    print(f'{DATA_FPATH=}')
    print('')

    print('simulation params:')
    print(f'{N_STEPS=}')
    print(f'{N_TRIALS=}')
    print(f'{BOUNDARY_TYPE=}')
    print(f'{BOUNDARY_HEIGHT=}')
    print(f'{TIME_STEP=}')
    print(f'{STD_NORM=}')
    print(f'{SPEED_VALS=}')
    print(f'{P_SPEED=}')
    print(f'{RNG_SEED=}')
    print('')

def main():

    print('all params:\n')
    print_params()

    print('running simulation...')
    sim = motion.MotionSimulation(
        n_steps=N_STEPS,
        n_trials=N_TRIALS,
        boundary_type=BOUNDARY_TYPE,
        boundary_height=BOUNDARY_HEIGHT,
        time_step=TIME_STEP,
        std_norm=STD_NORM,
        speed_vals=SPEED_VALS,
        p_speed=P_SPEED,
        rng_seed=RNG_SEED
    )
    sim.run()
    print('done.')

    print('saving results...')
    motion.save_simulation(sim, DATA_FPATH)
    print('done.\n')

if __name__ == '__main__':
    main()