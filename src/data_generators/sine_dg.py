from . import DataGenerator
import numpy as np
from collections.__init__ import namedtuple


class SineDataGen(DataGenerator):
    def __init__(self, params=None, param_sampler=None, xs_sampler=None):
        if params is None:
            params = {'amp':1, 'phase':0}
        self.params = params
        self.param_sampler = param_sampler
        self.xs_sampler = xs_sampler

    def generate_data(self, xs, params=None):
        if params is None:
            params = self.params
        phase = params['phase']
        amp = params['amp']
        return amp*np.sin(phase+xs)

