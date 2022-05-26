"""RNN for path integration"""

import numpy as np
import torch


class PathRNN(torch.nn.Module):
    
    def __init__(self, n_units=100, rnn_bias=True, output_bias=False):

        super(PathRNN, self).__init__()

        # Save hyperparameters
        self.n_units = n_units
        self.rnn_bias = rnn_bias
        self.output_bias = output_bias

        # RNN Layer
        self.rnn = torch.nn.RNN(
            input_size=2, 
            hidden_size=n_units, 
            num_layers=1, 
            nonlinearity='tanh', 
            batch_first=True, 
            bias=rnn_bias
        )

        # Output layer
        self.output = torch.nn.Linear(n_units, 2, bias=output_bias)

        # Initialize RNN weights
        torch.nn.init.zeros_(self.rnn.bias_ih_l0.data)
        torch.nn.init.zeros_(self.rnn.bias_hh_l0.data)
        torch.nn.init.normal_(self.rnn.weight_ih_l0.data, mean=0.0, std=np.sqrt(1/2))
        torch.nn.init.orthogonal_(self.rnn.weight_hh_l0.data)

        # Initialize output weights
        torch.nn.init.zeros_(self.output.weight.data)

        # CUDA
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)


    def init_hidden(self, batch_size):

        return torch.zeros(1, batch_size, self.n_units).to(self.device)
    
    def forward(self, vel):

        # Initializing hidden state for first input
        batch_size = vel.size(0)
        u_init = self.init_hidden(batch_size)
        
        # Run RNN on velocity sequences to get hidden unit values
        u_vals, _ = self.rnn(vel, u_init)
        
        # Apply output weights to get estimated position
        pos_est = self.output(u_vals)
        
        return pos_est, u_vals
    
    def run_np(self, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Convenience method for running model on batch stored in NumPy array"""

        # Convert velocity array to Tensor in order to run model
        vel_t = torch.Tensor(vel)

        # Predict estimated position
        pos_est_t, u_vals_t = self(vel_t)

        # Convert estimated position back to Numpy array
        pos_est = pos_est_t.detach().numpy()
        u_vals = u_vals_t.detach().numpy()

        return pos_est, u_vals
