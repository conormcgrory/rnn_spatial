import abc
import os
import json
import dataclasses

import numpy as np
import matplotlib.pyplot as plt


# Names of files storing parameters, velocity, and position data in output directory
SIM_PARAMS_FNAME = 'params.json'
SIM_VEL_FNAME = 'vel.npy'
SIM_POS_FNAME = 'pos.npy'


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

def get_boundary(shape: str, height: float) -> Boundary:

    if shape == 'square':
        return SquareBoundary(height)
    else:
        raise ValueError(f'Shape "{shape}" not supported.')


@dataclasses.dataclass
class TrajectoryParams:

    rng_seed: int = 999
    n_steps: int = 450
    boundary_shape: str = 'square'
    boundary_height: float = 2.0
    time_step: float = 0.1
    std_norm: float = 0.5
    mean_speed: float = 0.2
    coordinates = 'cartesian'

class TrajectoryGenerator:

    def __init__(self, params: TrajectoryParams):

        # Save parameters
        self.params = params

        # Copy parameters to reduce clutter
        self.n_steps = params.n_steps
        self.time_step = params.time_step
        self.coordinates = params.coordinates

        # Boundary for spatial environment
        self.boundary = get_boundary(params.boundary_shape, params.boundary_height)
           
        # Compute Brownian motion stddev from time step and normalized stddev
        self.std_brownian = np.sqrt(params.time_step) * params.std_norm

        # Mean speed of animal is used to compute scale of Rayleigh distribution
        self.scl_speed = params.mean_speed * np.sqrt(2 / np.pi)

        # Initialize random generator using seed
        self.rng = np.random.default_rng(params.rng_seed)

    def _smp_speed(self):
        return self.rng.rayleigh(self.scl_speed)

    def _smp_init_direction(self):
        return self.rng.random() * 2 * np.pi

    def _smp_direction_step(self):
        return self.std_brownian * self.rng.standard_normal()

    def _smp_direction_collision(self):
        return self.rng.random() * 2 * np.pi

    def _smp_trial(self):

        # Position (cartesian coordinates)
        pos_x = np.full(self.n_steps, np.nan)
        pos_y = np.full(self.n_steps, np.nan)

        # Velocity (polar coordinates)
        speed = np.full(self.n_steps, np.nan)
        theta = np.full(self.n_steps, np.nan)

        # Velocity (cartesian coordinates)
        vel_x = np.full(self.n_steps, np.nan)
        vel_y = np.full(self.n_steps, np.nan)

        # Initialize velocity
        speed[0] = self._smp_speed()
        theta[0] = self._smp_init_direction()
        vel_x[0] = speed[0] * np.cos(theta[0])
        vel_y[0] = speed[0] * np.sin(theta[0])

        # Initalize position
        pos_x[0] = self.time_step * vel_x[0]
        pos_y[0] = self.time_step * vel_y[0]

        # Check boundary condition
        if not self.boundary.contains(pos_x[0], pos_y[0]):
            raise ValueError('First step is outside boundary')

        for t in range(1, self.params.n_steps):

            # Update velocity
            speed[t] = self._smp_speed()
            theta[t] = theta[t - 1] + self._smp_direction_step()
            vel_x[t] = speed[t] * np.cos(theta[t])
            vel_y[t] = speed[t] * np.sin(theta[t])

            # Update position
            pos_x[t] = pos_x[t - 1] + self.time_step * vel_x[t]
            pos_y[t] = pos_y[t - 1] + self.time_step * vel_y[t]
 
            # If animal collides with wall, sample angle from uniform distribution
            while not self.boundary.contains(pos_x[t], pos_y[t]):

                # Resample direction
                theta[t] = self._smp_direction_collision()
                vel_x[t] = speed[t] * np.cos(theta[t])
                vel_y[t] = speed[t] * np.sin(theta[t])

                # Update position
                pos_x[t] = pos_x[t - 1] + self.time_step * vel_x[t]
                pos_y[t] = pos_y[t - 1] + self.time_step * vel_y[t]

        if self.coordinates == 'cartesian':
            vel_cart = np.stack((vel_x, vel_y), axis=-1)
            pos = np.stack((pos_x, pos_y), axis=-1)
            return vel_cart, pos

        elif self.coordinates == 'polar':
            theta_norm = np.mod(theta, 2 * np.pi)
            vel_polar = np.stack((speed, theta_norm), axis=-1)
            pos = np.stack((pos_x, pos_y), axis=-1)
            return vel_polar, pos

        else:
            raise ValueError(f'Coordinates {self.coordinates} not supported.')

    def smp_batch(self, n_trials):

        vel = np.full((n_trials, self.n_steps, 2), np.nan)
        pos = np.full((n_trials, self.n_steps, 2), np.nan)

        for k in range(n_trials):
            vel[k], pos[k] = self._smp_trial()

        return vel, pos


def plot_position(boundary, pos, ax=None):

    if ax is None:
        ax = plt.gca()

    # Plot boundary
    boundary.plot(ax)

    # Add origin point to beginning of position sequence
    pos_x = np.concatenate(([0.0], pos[:, 0]))
    pos_y = np.concatenate(([0.0], pos[:, 1]))

    # Plot position values
    ax.plot(pos_x, pos_y)

    ax.set_aspect('equal')


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