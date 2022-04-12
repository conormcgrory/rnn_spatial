"""Script for running trajectory simulation and saving results."""

import numpy as np
import matplotlib.pyplot as plt

import motion


# Output directory
DATA_DIR = 'data/sim_2022_04_12'

# Simulation parameters
N_STEPS = 450
BOUNDARY_TYPE = 'square'
BOUNDARY_HEIGHT = 2.0
TIME_STEP = 1.0
STD_NORM = 0.33
MAX_SPEED = 0.2
P_MOVE = 0.1
RNG_SEED = 999

# Number of trials to run simulation for
N_TRIALS = 2000


def print_params():

    print('output directory:')
    print(f'{DATA_DIR=}')
    print('')

    print('simulation params:')
    print(f'{N_STEPS=}')
    print(f'{BOUNDARY_TYPE=}')
    print(f'{BOUNDARY_HEIGHT=}')
    print(f'{TIME_STEP=}')
    print(f'{STD_NORM=}')
    print(f'{MAX_SPEED=}')
    print(f'{P_MOVE=}')
    print(f'{RNG_SEED=}')
    print('')

    print('num. trials:')
    print(f'{N_TRIALS=}')

def main():

    print('all params:\n')
    print_params()

    print('creating simulation...')
    sim = motion.MotionSimulationCartesian(
        n_steps=N_STEPS,
        boundary_type=BOUNDARY_TYPE,
        boundary_height=BOUNDARY_HEIGHT,
        time_step=TIME_STEP,
        std_norm=STD_NORM,
        max_speed=MAX_SPEED,
        p_move=P_MOVE,
        rng_seed=RNG_SEED
    )
    print('done.')

    print('running simulation...')
    vel, pos = sim.smp_batch(N_TRIALS)
    print('done.')

    print('saving results...')
    motion.save_simulation_results(sim.get_params(), vel, pos, DATA_DIR)
    print('done.\n')

if __name__ == '__main__':
    main()