"""Class for training RNN models"""

import dataclasses

import numpy as np
import torch


from trajectory import TrajectoryGenerator
from model import PathRNN


class TrainingLoss(torch.nn.Module):
    
    lambda_w: float
    lambda_h: float

    def __init__(self, lambda_w, lambda_h):
        super(TrainingLoss, self).__init__()
        self.lambda_w = lambda_w
        self.lambda_h = lambda_h

    def forward(self, pos, h, pos_est, w_ih, w_out):

        # Mean squared error component of loss
        loss_mse = torch.mean(torch.square(pos_est - pos))

        # Weight regularization component of loss
        L2_ih = torch.mean(torch.square(w_ih))
        L2_out = torch.mean(torch.square(w_out))
        loss_w = L2_ih + L2_out

        # Metabolic component of loss
        loss_h = torch.mean(torch.square(h))

        # Total loss 
        return loss_mse + self.lambda_w * loss_w + self.lambda_h * loss_h


@dataclasses.dataclass
class TrainerParams:

    # Number of batches to run training for
    n_batches: int = 1900

    # Number of trials per batch
    batch_size: int = 500

    # Coefficient of L2 regularization term in loss function
    lambda_w: float = 0.5

    # Coefficient of metabolic term in loss function
    lambda_h: float = 0.1

    # Learning rate used for optimization
    learning_rate: float = 1e-4


class Trainer:

    def __init__(self, params: TrainerParams, traj_gen: TrajectoryGenerator, model: PathRNN):

        # Training parameters
        self.params = params

        # Trajectory generator
        self.traj_gen = traj_gen

        # Model
        self.model = model

        # Loss
        self.criterion = TrainingLoss(lambda_w=params.lambda_w, lambda_h=params.lambda_h)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.params.learning_rate,
        )

    def train(self):

        for i in range(1, self.params.n_batches + 1):

            # Sample next batch
            vel_np, pos_np = self.traj_gen.smp_batch(self.params.batch_size)
            vel = torch.Tensor(vel_np)
            pos = torch.Tensor(pos_np)

            # Clear gradients from previous batch
            self.optimizer.zero_grad()

            # Predict position for batch
            pos_est, h = self.model(vel)

            # Compute loss using predictions, activations, and weights
            w_ih = self.model.rnn.weight_ih_l0.data
            w_out = self.model.output.weight.data
            loss = self.criterion(pos, h, pos_est, w_ih, w_out)
 
            # Compute gradient via backprop
            loss.backward()

            # Update model parameters
            self.optimizer.step()
    
            if i % 100 == 0:
                print('Batch: {}/{}.............'.format(i, self.params.n_batches), end=' ')
                print("Loss: {:.4f}".format(loss.item()))

    def save_model(self, fpath):
        torch.save(self.model.state_dict(), fpath)
