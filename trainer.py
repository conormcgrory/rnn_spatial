"""Class for training RNN models"""

import dataclasses

import numpy as np
import torch


from trajectory import TrajectoryGenerator
from model import PathRNN


@dataclasses.dataclass
class TrainerParams:

    n_batches: int = 1900
    batch_size: int = 500
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4


class Trainer:

    def __init__(self, params: TrainerParams, traj_gen: TrajectoryGenerator, model: PathRNN):

        # Training parameters
        self.params = params

        # Trajectory generator
        self.traj_gen = traj_gen

        # Model
        self.model = model

        # Loss function
        self.criterion = torch.nn.MSELoss()

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.params.learning_rate, 
            weight_decay=self.params.weight_decay
        )

    def train(self):

        for i in range(1, self.params.n_batches + 1):

            # Sample next batch
            vel_np, pos_np = self.traj_gen.smp_batch(self.params.batch_size)
            vel = torch.Tensor(vel_np)
            pos = torch.Tensor(pos_np)

            # Clear gradients from previous batch
            self.optimizer.zero_grad()

            # Compute loss
            pos_est, _ = self.model(vel)
            loss = self.criterion(pos_est, pos)

            # Compute gradient via backprop
            loss.backward()

            # Gradient clipping
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0, norm_type=2)

            # Update model parameters
            self.optimizer.step()
    
            if i % 100 == 0:
                print('Batch: {}/{}.............'.format(i, self.params.n_batches), end=' ')
                print("Loss: {:.4f}".format(loss.item()))

    def save_model(self, fpath):
        torch.save(self.model.state_dict(), fpath)