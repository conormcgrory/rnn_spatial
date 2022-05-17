"""Script for training model."""

from trajectory import TrajectoryGenerator
from model import PathRNN
from trainer import Trainer
from run import get_default_params, save_run

# Path where model is saved
RUN_FPATH = 'models/20220517_test'

# Set parameters
params = get_default_params()
params.traj.rng_seed = 1993
params.trainer.n_batches = 400 #8000
params.trainer.lambda_h = 4.0


def main():

    print('all params:\n')
    params.print()
    
    print('initializing...')
    traj_gen = TrajectoryGenerator(params.traj)
    model = PathRNN(params.model)
    trainer = Trainer(params.trainer, traj_gen, model)
    print('done')

    print('training network...')
    trainer.train()
    print('done.')

    print('saving params and model...')
    save_run(params, trainer, RUN_FPATH)
    print('done.')


if __name__ == '__main__':
    main()
