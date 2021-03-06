"""Class for training RNN models"""

import copy

import torch

from spatialnet.trajectory import TrajectoryGenerator
from spatialnet.model import PathRNN


class TrainingLoss(torch.nn.Module):
    
    lambda_w: float
    lambda_h: float

    def __init__(self, lambda_w, lambda_h):
        super(TrainingLoss, self).__init__()
        self.lambda_w = lambda_w
        self.lambda_h = lambda_h

        # TODO: Remove this eventually
        self._loss_mse = 0
        self._loss_w = 0
        self._loss_h = 0
        self._loss_total = 0

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
        loss_total = loss_mse + self.lambda_w * loss_w + self.lambda_h * loss_h

        # Save loss values
        self._loss_mse = loss_mse.item()
        self._loss_w = loss_w.item()
        self._loss_h = loss_h.item()
        self._loss_total = loss_total.item()

        return loss_total


class Trainer:

    def __init__(self, traj_gen: TrajectoryGenerator, model: PathRNN, 
        batch_size=500, lambda_w=0.5, lambda_h=0.1, learning_rate=0.1):

        # Training parameters
        self.batch_size = batch_size
        self.lambda_w = lambda_w
        self.lambda_h = lambda_h
        self.learning_rate = learning_rate

        # Trajectory generator
        self.traj_gen = traj_gen

        # Model
        self.model = model

        # Counter for number of training steps
        self.n_steps = 0

        # Loss
        self._loss_criterion = TrainingLoss(lambda_w=lambda_w, lambda_h=lambda_h)

        # Optimizer
        self._optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
        )

        # Loss values
        self.loss_mse = []
        self.loss_w = []
        self.loss_h = []
        self.loss_total = []


    def step(self):
        """Present one batch to model and update parameters"""

        # Sample batch from trajectory generator
        vel_np, pos_np = self.traj_gen.smp_batch(self.batch_size)
        vel = torch.Tensor(vel_np)
        pos = torch.Tensor(pos_np)

        # Clear gradients from previous batch
        self._optimizer.zero_grad()

        # Predict position for batch
        pos_est, h = self.model(vel)

        # Compute loss using predictions, activations, and weights
        w_ih = self.model.rnn.weight_ih_l0.data
        w_out = self.model.output.weight.data
        loss = self._loss_criterion(pos, h, pos_est, w_ih, w_out)

        # Add loss values to running lists
        self.loss_mse.append(self._loss_criterion._loss_mse)
        self.loss_w.append(self._loss_criterion._loss_w)
        self.loss_h.append(self._loss_criterion._loss_h)
        self.loss_total.append(self._loss_criterion._loss_total)
 
        # Compute gradient with respect to loss via backprop
        loss.backward()

        # Update model parameters
        self._optimizer.step()

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
