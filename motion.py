"""Motion and environment simulation"""

import abc

import numpy as np
import matplotlib.pyplot as plt


class Boundary(abc.ABC):

    @abc.abstractmethod
    def contains(x, y):
        pass

    @abc.abstractmethod
    def plot(ax):
        pass


class RectangleBoundary:

    def __init__(self, ctr_x, ctr_y, width, height):

        self.x_min = ctr_x - 0.5 * width
        self.x_max = ctr_x + 0.5 * width
        self.y_min = ctr_y - 0.5 * height 
        self.y_max = ctr_y + 0.5 * height

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


#class MotionSimulation:
#
#    def __init__(self):
#
#        self.boundary = RectangleBoundary(0.0, 0.0, 2.0, 2.0)
#
#    def sample_trial(n_pts):
        
                
