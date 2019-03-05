from src.data_generators import DataGenerator
import numpy as np
from collections.__init__ import namedtuple
import tensorflow as tf
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ExpSineSquared


class GPDataGen(DataGenerator):
    def __init__(self, params=None,
                 param_sampler=None,
                 kernel= RBF(),
                 mean_fn = None,
                 xs_sampler=None):
        self.params = params
        self.param_sampler = None
        self.xs_sampler = xs_sampler
        self.mean_fn = mean_fn
        self.kernel = kernel
        self.gp = GaussianProcessRegressor(self.kernel)

    def generate_data(self, xs, params=None):
        xs = xs.reshape((-1, 1))
        return self.gp.sample_y(xs.reshape((-1, 1))).flatten()

