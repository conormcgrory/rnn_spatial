"""Script for training path integration RNN"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

import motion
from path_rnn import PathRNN


# Path where model is saved
MODEL_FPATH = 'models/test_2022_04_19_03.pt'

# Simulation parameters
N_STEPS = 450
BOUNDARY_TYPE = 'square'
BOUNDARY_HEIGHT = 2.0
TIME_STEP = 1.0
STD_NORM = 0.33
MEAN_SPEED = 0.2
RNG_SEED = 999

# RNN parameters
NUM_UNITS = 100

# Training parameters
N_BATCHES = 1900
BATCH_SIZE = 500
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4


def print_params():

    print('filepaths:')
    print(f'{MODEL_FPATH=}')
    print('')

    print('simulation params:')
    print(f'{N_STEPS=}')
    print(f'{BOUNDARY_TYPE=}')
    print(f'{BOUNDARY_HEIGHT=}')
    print(f'{TIME_STEP=}')
    print(f'{STD_NORM=}')
    print(f'{MEAN_SPEED=}')
    print(f'{RNG_SEED=}')
    print('')

    print('rnn params:')
    print(f'{NUM_UNITS=}')
    print('')

    print('training params:')
    print(f'{N_BATCHES=}')
    print(f'{BATCH_SIZE=}')
    print(f'{LEARNING_RATE=}')
    print(f'{WEIGHT_DECAY=}')
    print('')

def main():

    print('all params:\n')
    print_params()

    print('initializing...')

    # Create simulation
    sim = motion.MotionSimulation(
        n_steps=N_STEPS,
        boundary_type=BOUNDARY_TYPE,
        boundary_height=BOUNDARY_HEIGHT,
        time_step=TIME_STEP,
        std_norm=STD_NORM,
        max_speed=MEAN_SPEED,
        rng_seed=RNG_SEED
    )

    # Create model
    model = PathRNN(n_units=NUM_UNITS)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    print('done.')

    print('training network...')
    for i in range(1, N_BATCHES + 1):

        # Sample next batch
        vel_np, pos_np = sim.smp_batch(BATCH_SIZE)
        vel = torch.Tensor(vel_np)
        pos = torch.Tensor(pos_np)

        # Clear gradients from previous batch
        optimizer.zero_grad()

        # Compute loss
        pos_est, _ = model(vel)
        loss = criterion(pos_est, pos)

        # Compute gradient via backprop
        loss.backward()

        # Gradient clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

        # Update model parameters
        optimizer.step()
    
        if i % 100 == 0:
            print('Batch: {}/{}.............'.format(i, N_BATCHES), end=' ')
            print("Loss: {:.4f}".format(loss.item()))

    print('done.')

    print('saving model...')
    torch.save(model.state_dict(), MODEL_FPATH)
    print('done.')

if __name__ == '__main__':
    main()