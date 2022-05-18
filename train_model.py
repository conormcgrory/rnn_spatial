"""Script for training model."""

import os
import subprocess
import json 
import dataclasses
from datetime import datetime

import torch

from trajectory import TrajectoryGenerator
from model import PathRNN
from trainer import Trainer
from run import get_default_params, RunParameters


# Directory where model is saved
OUT_DIR = 'models/20220518_test'

# Set parameters
params = get_default_params()
params.traj.rng_seed = 1993
params.trainer.lambda_h = 4.0

# TODO: Add these to main params
N_EPOCHS = 3
N_BATCHES = 100
TEST_BATCH_SIZE = 500


def compute_mse(model, tgen):

    # Sample batch from trajectory generator
    vel_np, pos_np = tgen.smp_batch(TEST_BATCH_SIZE)
    vel = torch.Tensor(vel_np)
    pos = torch.Tensor(pos_np)

    # Predict position for batch
    pos_est, _ = model(vel)

    # TODO: Find a way to do this without creating object every time
    # Compute MSE
    mse_loss = torch.nn.MSELoss()
    return mse_loss(pos, pos_est).item()

def params_to_dict(params):
    return {
        'traj': dataclasses.asdict(params.traj),
        'model': dataclasses.asdict(params.model),
        'trainer': dataclasses.asdict(params.trainer),
        'n_epochs': N_EPOCHS,
        'n_batches': N_BATCHES,
        'test_batch_size': TEST_BATCH_SIZE,
    }

def save_params(params: RunParameters, fpath):
    with open(fpath, 'w') as f:
        pdict = params_to_dict(params)
        json.dump(pdict, f, indent=4)

def get_git_short_hash():
    short_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])
    return str(short_hash, "utf-8").strip()

def save_runinfo(ts_start, ts_end, mse_vals, fpath):
    info_dict = {
        'commit': get_git_short_hash(),
        'mse_vals': mse_vals,
        'time_started': ts_start,
        'time_finished': ts_end
    }
    with open(fpath, 'w') as f:
       json.dump(info_dict, f, indent=4)


def main():

    print('all params:\n')
    params.print()
    
    print('initializing...')

    # Training setup for network
    tgen_train = TrajectoryGenerator(params.traj)
    model = PathRNN(params.model)
    trainer = Trainer(params.trainer, tgen_train, model)

    # Trajectory generator for testing
    tgen_test_params = dataclasses.replace(params.traj, rng_seed=1989)
    tgen_test = TrajectoryGenerator(tgen_test_params)

    # Create output directory and save parameters
    os.mkdir(OUT_DIR)
    params_fpath = os.path.join(OUT_DIR, 'params.json')
    save_params(params, params_fpath)

    print('done')

    # Array for storing MSE values
    mse_vals = []

    ts_start = datetime.now().isoformat()
    print('training network...')
    for epoch in range(N_EPOCHS):

        # Train network
        trainer.train(N_BATCHES)

        # Compute MSE
        mse = compute_mse(model, tgen_test)
        mse_vals.append(mse)

        # Save checkpoint
        model_fpath = os.path.join(OUT_DIR, f'model_{epoch:02d}.pt')
        torch.save(trainer.model.state_dict(), model_fpath)

        print(f'epoch: {epoch:02d}')
        print(f'    mse: {mse}')
        print(f'    saved to: {model_fpath}')

    ts_end = datetime.now().isoformat()
    print('done.')

    print('saving run info...')
    runinfo_fpath = os.path.join(OUT_DIR, 'runinfo.json')
    save_runinfo(ts_start, ts_end, mse_vals, runinfo_fpath)
    print('done.')

if __name__ == '__main__':
    main()
