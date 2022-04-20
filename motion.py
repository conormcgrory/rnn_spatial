"""Motion and environment simulation"""

import abc
import os
import json

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


class MotionSimulation:

    def __init__(self, n_steps,
        boundary_type='square', boundary_height=2.0, 
        time_step=0.1, std_norm=0.5, 
        mean_speed=0.2,
        rng_seed=999):

        # Number of time steps per trial
        self.n_steps = n_steps

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
        
        # Mean speed of animal is used to compute scale of Rayleigh distribution
        self.mean_speed = mean_speed
        self.scl_speed = mean_speed * np.sqrt(2 / np.pi)

        # Initialize random generator using seed
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)
       
    def _smp_speed(self):
        return self.rng.rayleigh(self.scl_speed)

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

    def smp_batch(self, n_trials):

        vel = np.full((n_trials, self.n_steps, 2), np.nan)
        pos = np.full((n_trials, self.n_steps, 2), np.nan)

        for k in range(n_trials):
            vel[k], pos[k] = self._smp_trial()

        return vel, pos

    def get_params(self):
        return dict(
            n_steps=self.n_steps,
            boundary_type=self.boundary_type,
            boundary_height=self.boundary.height,
            time_step=self.time_step,
            std_norm=self.std_norm,
            mean_speed=self.mean_speed,
            rng_seed=self.rng_seed
        )


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

class MotionSimulationCartesian:
    """Class for generating simulated trajectories with velocity in Cartesian coordinates."""

    def __init__(self, n_steps,
        boundary_type='square', boundary_height=2.0, 
        time_step=0.1, std_norm=0.5, 
        mean_speed=0.2,
        rng_seed=999):

        # Number of time steps per trial
        self.n_steps = n_steps

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

        # Mean speed of animal is used to compute scale of Rayleigh distribution
        self.mean_speed = mean_speed
        self.scl_speed = mean_speed * np.sqrt(2 / np.pi)

        # Initialize random generator using seed
        self.rng_seed = rng_seed
        self.rng = np.random.default_rng(rng_seed)

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

        for t in range(1, self.n_steps):

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

        vel = np.stack((vel_x, vel_y), axis=-1)
        pos = np.stack((pos_x, pos_y), axis=-1)

        return vel, pos

    def smp_batch(self, n_trials):

        vel = np.full((n_trials, self.n_steps, 2), np.nan)
        pos = np.full((n_trials, self.n_steps, 2), np.nan)

        for k in range(n_trials):
            vel[k], pos[k] = self._smp_trial()

        return vel, pos

    def get_params(self):
        return dict(
            n_steps=self.n_steps,
            boundary_type=self.boundary_type,
            boundary_height=self.boundary.height,
            time_step=self.time_step,
            std_norm=self.std_norm,
            mean_speed=self.mean_speed,
            rng_seed=self.rng_seed
        )


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

def save_simulation_results(params, vel, pos, dirpath):
    """Save MotionSimulation results to directory."""

    # Create directory for saving results
    os.mkdir(dirpath)

    # Filenames for simulation parameters, velocity, and position data
    params_fpath = os.path.join(dirpath, SIM_PARAMS_FNAME)
    vel_fpath = os.path.join(dirpath, SIM_VEL_FNAME)
    pos_fpath = os.path.join(dirpath, SIM_POS_FNAME)

    # Save parameters to JSON file
    with open(params_fpath, 'w') as f:
        json.dump(params, f, indent=4)

    # Save velocity and position arrays to .npy files
    np.save(vel_fpath, vel)
    np.save(pos_fpath, pos)

def load_simulation_results(dirpath):
    """Load MotionSimulation results from directory."""

    # Filenames for simulation parameters, velocity, and position data
    params_fpath = os.path.join(dirpath, SIM_PARAMS_FNAME)
    vel_fpath = os.path.join(dirpath, SIM_VEL_FNAME)
    pos_fpath = os.path.join(dirpath, SIM_POS_FNAME)

    # Load parameters from JSON file
    with open(params_fpath, 'r') as f:
        params = json.load(f)

    # Load velocity and position arrays from .npy files
    vel = np.load(vel_fpath)
    pos = np.load(pos_fpath)

    return params, vel, pos
