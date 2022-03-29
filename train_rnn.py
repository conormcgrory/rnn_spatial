"""Script for training network on data"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader

import motion
from path_rnn import PathRNN

# Path where data is loaded from
DATA_FPATH = 'data/sim_2022_03_28.npz'

# Path where model is saved
MODEL_FPATH = 'models/test_2022_03_28.pt'

# RNN parameters
NUM_UNITS = 32

# Training parameters
NUM_EPOCHS = 1000
BATCH_SIZE = 500
LEARNING_RATE = 0.001


def print_params():

    print('filepaths:')
    print(f'{DATA_FPATH=}')
    print(f'{MODEL_FPATH=}')
    print('')

    print('rnn params:')
    print(f'{NUM_UNITS=}')
    print('')

    print('training params:')
    print(f'{NUM_EPOCHS=}')
    print(f'{BATCH_SIZE=}')
    print(f'{LEARNING_RATE=}')
    print('')

def main():

    print('all params:\n')
    print_params()

    print('loading simulation data...')
    sim = motion.load_simulation(DATA_FPATH)
    print('done.')

    print('initializing...')

    # Create DataLoader for batch generation
    dset = sim.to_dataset()
    train_dataloader = DataLoader(dset, batch_size=BATCH_SIZE, shuffle=True)

    # Instantiate model
    model = PathRNN(n_units=NUM_UNITS)

    # Define loss function and optimizer
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print('done.')

    print('training network...')
    for epoch in range(1, NUM_EPOCHS + 1):

        for vel_batch, pos_batch in train_dataloader:

            # Clear gradients from previous epoch
            optimizer.zero_grad()

            # Compute loss
            pos_est, u_vals = model(vel_batch)
            loss = criterion(pos_est, pos_batch)

            # Compute gradient via backprop
            loss.backward()

            # Update model parameters
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)
            optimizer.step()
    
        if epoch % 100 == 0:
            print('Epoch: {}/{}.............'.format(epoch, NUM_EPOCHS), end=' ')
            print("Loss: {:.4f}".format(loss.item()))
    print('done.')

    print('saving model...')
    torch.save(model.state_dict(), MODEL_FPATH)
    print('done.')


if __name__ == '__main__':
    main()