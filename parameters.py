"""Class for managing parameters of trajectory generation, model, and training."""

import dataclasses

from trajectory import TrajectoryParams
from model import ModelHyperParams
from trainer import TrainerParams


@dataclasses.dataclass
class Parameters:

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


def get_default_params():
    return Parameters()