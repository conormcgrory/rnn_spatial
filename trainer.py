"""Class for training RNN models"""

import dataclasses
import copy

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
        self.loss_criterion = TrainingLoss(lambda_w=params.lambda_w, lambda_h=params.lambda_h)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.params.learning_rate,
        )

        # Counter for number of training steps
        self.n_steps = 0

    def step(self):
        """Present one batch to model and update parameters"""

        # Sample batch from trajectory generator
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
        loss = self.loss_criterion(pos, h, pos_est, w_ih, w_out)
 
        # Compute gradient with respect to loss via backprop
        loss.backward()

        # Update model parameters
        self.optimizer.step()

        # Increment step counter
        self.n_steps = self.n_steps + 1

    def train(self, n_steps):
        """Train model for number of steps."""

        for i in range(n_steps):
            self.step()


    # TODO: Move this out of class
    def model_state(self):
        """Return a copy of the model's current state dict"""

        return copy.deepcopy(self.model.state_dict())
