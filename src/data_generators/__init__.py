from abc import ABC, abstractmethod


class DataGenerator(ABC):
    def __init__(self, params=None, param_sampler=None, xs_sampler=None):

        self.params = params
        self.param_sampler = param_sampler
        self.xs_sampler = xs_sampler

    @abstractmethod
    def generate_data(self, params, xs):
        pass

    @abstractmethod
    def generate_data(self, params, xs):
        pass

    def param_sampler(self):
        return self.param_sampler()

    def xs_sampler(self):
        return self.xs_sampler()
