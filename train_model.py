"""Script for training model."""

from trajectory import TrajectoryGenerator
from model import PathRNN
from trainer import Trainer

from parameters import get_default_params

# Path where model is saved
MODEL_FPATH = 'models/test_2022_04_21.pt'

# Set parameters
params = get_default_params()
#params.trainer.batch_size = 100
#params.trainer.n_batches = 3
#params.traj.rng_seed = 998


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

    print('saving model...')
    trainer.save_model(MODEL_FPATH)
    print('done.')


if __name__ == '__main__':
    main()