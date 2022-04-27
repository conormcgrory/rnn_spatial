"""RNN for path integration"""

from inspect import Parameter

import numpy as np
import torch
import dataclasses


@dataclasses.dataclass
class ModelHyperParams:

    n_units: int = 100
    rnn_bias: bool = True
    output_bias: bool = False

class PathRNN(torch.nn.Module):
    
    def __init__(self, hparams: ModelHyperParams):

        super(PathRNN, self).__init__()

        # Save hyperparameters
        self.hparams = hparams

        # RNN Layer
        self.rnn = torch.nn.RNN(
            input_size=2, 
            hidden_size=hparams.n_units, 
            num_layers=1, 
            nonlinearity='tanh', 
            batch_first=True, 
            bias=hparams.rnn_bias
        )

        # Output layer
        self.output = torch.nn.Linear(hparams.n_units, 2, bias=hparams.output_bias)

        # Initialize RNN weights
        torch.nn.init.zeros_(self.rnn.bias_ih_l0.data)
        torch.nn.init.zeros_(self.rnn.bias_hh_l0.data)
        torch.nn.init.normal_(self.rnn.weight_ih_l0.data, mean=0.0, std=np.sqrt(1/2))
        torch.nn.init.orthogonal_(self.rnn.weight_hh_l0.data)

        # Initialize output weights
        torch.nn.init.zeros_(self.output.weight.data)

    def init_hidden(self, batch_size):

        return torch.zeros(1, batch_size, self.hparams.n_units)
    
    def forward(self, vel):

        # Initializing hidden state for first input
        batch_size = vel.size(0)
        u_init = self.init_hidden(batch_size)
        
        # Run RNN on velocity sequences to get hidden unit values
        u_vals, _ = self.rnn(vel, u_init)
        
        # Apply output weights to get estimated position
        pos_est = self.output(u_vals)
        
        return pos_est, u_vals