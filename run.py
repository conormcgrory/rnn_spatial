"""Class for managing parameters of full training runs."""

import os
import dataclasses

import torch
import json

from trajectory import TrajectoryParams
from model import ModelHyperParams, PathRNN
from trainer import TrainerParams


@dataclasses.dataclass
class RunParameters:

    # Trajectory generation parameters
    traj: TrajectoryParams = TrajectoryParams()

    # Model parameters
    model: ModelHyperParams = ModelHyperParams()

    # Training parameters
    trainer: TrainerParams = TrainerParams()

    def print(self):

        print('trajectory:')
        print(dataclasses.asdict(self.traj))
        print('')

        print('model:')
        print(dataclasses.asdict(self.model))
        print('')

        print('trainer:')
        print(dataclasses.asdict(self.trainer))
        print('')


def params_to_dict(params):
    return {
        'traj': dataclasses.asdict(params.traj),
        'model': dataclasses.asdict(params.model),
        'trainer': dataclasses.asdict(params.trainer)
    }


def params_from_dict(pdict):
    return RunParameters(
        traj=TrajectoryParams(**pdict['traj']),
        model=ModelHyperParams(**pdict['model']),
        trainer=TrainerParams(**pdict['trainer'])
    )


def get_default_params():
    return RunParameters()


def save_run(params:RunParameters, model: PathRNN, dirpath: str):

    # Create directory for saving results
    os.mkdir(dirpath)

    # Filenames for parameters and model
    params_fpath = os.path.join(dirpath, 'params.json')
    model_fpath = os.path.join(dirpath, 'model.pt')

    # Save parameters to JSON file
    with open(params_fpath, 'w') as f:
        pdict = params_to_dict(params)
        json.dump(pdict, f, indent=4)

    # Save state dict of model
    torch.save(model.state_dict(), model_fpath)


def load_run(dirpath):

    # Filenames for parameters and model
    params_fpath = os.path.join(dirpath, 'params.json')
    model_fpath = os.path.join(dirpath, 'model.pt')

    # Load parameters from JSON file
    with open(params_fpath, 'r') as f:
        pdict = json.load(f)
        params = params_from_dict(pdict)

    # Create model object and load state dict
    model = PathRNN(params.model)
    model.load_state_dict(torch.load(model_fpath))
    model.eval()

    return params, model