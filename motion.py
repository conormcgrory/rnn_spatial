"""Motion and environment simulation"""

import abc

import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset


class Boundary(abc.ABC):

    @abc.abstractmethod
    def contains(x, y):
        pass

    @abc.abstractmethod
    def plot(ax):
        pass


class SquareBoundary:

    def __init__(self, height):

        self.height = height
        self.x_min = -0.5 * height
        self.x_max = 0.5 * height 
        self.y_min = -0.5 * height 
        self.y_max = 0.5 * height 

    def contains(self, x, y):

        return ((x > self.x_min) 
                and (x < self.x_max) 
                and (y > self.y_min) 
                and (y < self.y_max))

    def plot(self, ax=None):

        if ax is None:
            ax = plt.gca()

        x_vals = np.array([
            self.x_min, self.x_max, self.x_max, self.x_min, self.x_min])
        y_vals = np.array([
            self.y_min, self.y_min, self.y_max, self.y_max, self.y_min])

        ax.plot(x_vals, y_vals, '--')


class MotionDataset(Dataset):

    def __init__(self, vel, pos):

        self.vel = torch.Tensor(vel)
        self.pos = torch.Tensor(pos)
        self.num_trials = vel.shape[0]

    def __getitem__(self, index):
        return self.vel[index], self.pos[index]
 
    def __len__(self):
        return self.num_trials


class MotionSimulation:

    def __init__(self, n_steps, n_trials,
        boundary_type='square', boundary_height=2.0, 
        time_step=0.1, std_norm=0.5, 
        speed_vals=[0.0, 0.1, 0.2, 0.3, 0.4], p_speed=[0.2, 0.2, 0.2, 0.2, 0.2], 
        rng_seed=999):

        # Save number of steps and trials
        self.n_steps = n_steps
        self.n_trials = n_trials

        # Boundary for spatial environment
        self.boundary_type = boundary_type
        self.boundary_height = boundary_height
        if boundary_type == 'square':
            self.boundary = SquareBoundary(boundary_height)
        else:
            raise ValueError(f'Boundary "{boundary_type}" not supported.')
            
        # Compute Brownian motion stddev from time step and normalized stddev
        self.time_step = time_step
        self.std_norm = std_norm
        self.std_brownian = np.sqrt(time_step) * std_norm
        
        # Speed distribution of animal
        self.speed_vals = speed_vals
        self.p_speed = p_speed

        # Random seed
        self.rng_seed = rng_seed

        # Arrays for velocity and position are initialized to None
        self.vel = None
        self.pos = None

    def _init_rng(self):
        self.rng = np.random.default_rng(self.rng_seed)
        
    def _smp_speed(self):
        return self.rng.choice(self.speed_vals, p=self.p_speed)

    def _smp_init_direction(self):
        return self.rng.random() * 2 * np.pi

    def _smp_direction_step(self):
        return self.std_brownian * self.rng.standard_normal()

    def _smp_direction_collision(self):
        return self.rng.random() * 2 * np.pi

    def _get_xstep(self, speed, theta):
        return speed * self.time_step * np.cos(theta)

    def _get_ystep(self, speed, theta):
        return speed * self.time_step * np.sin(theta)

    def _smp_trial(self):

        # Position (cartesian coordinates)
        pos_x = np.full(self.n_steps, np.nan)
        pos_y = np.full(self.n_steps, np.nan)

        # Velocity (polar coordinates)
        speed = np.full(self.n_steps, np.nan)
        theta = np.full(self.n_steps, np.nan)

        # Initialize velocity
        speed[0] = self._smp_speed()
        theta[0] = self._smp_init_direction()

        # Initalize position
        pos_x[0] = self._get_xstep(speed[0], theta[0])
        pos_y[0] = self._get_ystep(speed[0], theta[0])

        # Check boundary condition
        if not self.boundary.contains(pos_x[0], pos_y[0]):
            raise ValueError('First step is outside boundary')

        for t in range(1, self.n_steps):

            # Update velocity
            speed[t] = self._smp_speed()
            theta[t] = theta[t - 1] + self._smp_direction_step()

            # Update position
            pos_x[t] = pos_x[t - 1] + self._get_xstep(speed[t], theta[t])
            pos_y[t] = pos_y[t - 1] + self._get_ystep(speed[t], theta[t])
 
            # If animal collides with wall, sample angle from uniform distribution
            while not self.boundary.contains(pos_x[t], pos_y[t]):

                # Resample direciton
                theta[t] = self._smp_direction_collision()

                # Update position
                pos_x[t] = pos_x[t - 1] + self._get_xstep(speed[t], theta[t])
                pos_y[t] = pos_y[t - 1] + self._get_ystep(speed[t], theta[t])

        # Normalize direction so that it lies in [0, 2pi]
        theta_norm = np.mod(theta, 2 * np.pi)

        vel = np.stack((speed, theta_norm), axis=-1)
        pos = np.stack((pos_x, pos_y), axis=-1)

        return vel, pos

    def run(self):

        self._init_rng()
        self.pos = np.full((self.n_trials, self.n_steps, 2), np.nan)
        self.vel = np.full((self.n_trials, self.n_steps, 2), np.nan)

        for k in range(self.n_trials):
            self.vel[k], self.pos[k] = self._smp_trial()

    def to_dataset(self):
        return MotionDataset(self.vel, self.pos)

    def plot_position(self, trial, ax=None):

        if ax is None:
            ax = plt.gca()

        # Make sure x- and y-scales are the same
        ax.set_aspect('equal')

        # Plot boundary
        self.boundary.plot(ax)

        # Add origin point to beginning of position sequence
        pos_x = np.concatenate(([0.0], self.pos[trial, :, 0]))
        pos_y = np.concatenate(([0.0], self.pos[trial, :, 1]))

        # Plot position values
        ax.plot(pos_x, pos_y)


def plot_position_estimate(boundary, pos_true, pos_est, ax=None):

    if ax is None:
        ax = plt.gca()

    # Plot boundary
    boundary.plot(ax)

    # Add origin point to beginning of position sequences
    x_true = np.concatenate(([0.0], pos_true[:, 0]))
    y_true = np.concatenate(([0.0], pos_true[:, 1]))
    x_est = np.concatenate(([0.0], pos_est[:, 0]))
    y_est = np.concatenate(([0.0], pos_est[:, 1]))

    # Plot position sequences
    ax.plot(x_true, y_true, color='black', label='true')
    ax.plot(x_est, y_est, color='red', label='est')

    ax.set_aspect('equal')
    ax.legend()

def save_simulation(sim, fpath):
    """Save MotionSimulation object to file."""

    # Don't save simulation if it hasn't been run
    if sim.vel is None or sim.pos is None:
        raise ValueError('Simulation has not been run.')

    # Save all parameters and data as NumPy arrays
    np.savez(fpath, 
        n_steps=sim.n_steps,
        n_trials=sim.n_trials,
        boundary_type=sim.boundary_type,
        boundary_height=sim.boundary.height,
        time_step=sim.time_step,
        speed_vals=sim.speed_vals,
        p_speed=sim.p_speed,
        rng_seed=sim.rng_seed,
        pos=sim.pos, 
        vel=sim.vel,
        allow_pickle=False
    )

def load_simulation(fpath):
    """Load MotionSimulation object from file."""

    with np.load(fpath) as data:

        # Create MotionSimulation object from parameters (stored as NumPy arrays)
        sim = MotionSimulation(
            n_steps=data['n_steps'].item(),
            n_trials=data['n_trials'].item(),
            boundary_type=data['boundary_type'].item(),
            boundary_height=data['boundary_height'].item(),
            time_step=data['time_step'].item(),
            speed_vals=data['speed_vals'],
            p_speed=data['p_speed'],
            rng_seed=data['rng_seed'].item()
        )

        # Set velocity and position arrays of object
        sim.vel = data['vel']
        sim.pos = data['pos']

        return sim