from collections import OrderedDict
import os
from os.path import join as pjoin

import numpy as np
import theano.tensor as T
from smartlearner.utils import sharedX
import smartlearner.initializers as initer
from smartlearner import Model

from .utils import load_dict_from_json_file, save_dict_to_json_file


class Perceptron(Model):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = sharedX(value=np.zeros((input_size, output_size)), name='W', borrow=True)
        self.b = sharedX(value=np.zeros(output_size), name='b', borrow=True)

    def initialize(self, weights_initializer=initer.UniformInitializer(random_seed=1234)):
        weights_initializer(self.W)

    @property
    def updates(self):
        return {}  # No updates.

    @property
    def parameters(self):
        return [self.W, self.b]

    def get_model_output(self, X):
        preactivation = T.dot(X, self.W) + self.b
        probs = T.nnet.softmax(preactivation)
        return probs

    def use(self, X):
        probs = self.get_model_output(X)
        return T.argmax(probs, axis=1, keepdims=True)

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        hyperparameters = {'input_size': self.input_size,
                           'output_size': self.output_size}
        save_dict_to_json_file(pjoin(path, "meta.json"), {"name": self.__class__.__name__})
        save_dict_to_json_file(pjoin(path, "hyperparams.json"), hyperparameters)

        params = {param.name if param.name is not None else param.auto_name: param.get_value()
                  for param in self.parameters}
        np.savez(pjoin(path, "params.npz"), **params)

    @classmethod
    def load(cls, path):
        meta = load_dict_from_json_file(pjoin(path, "meta.json"))
        assert meta['name'] == cls.__name__

        hyperparams = load_dict_from_json_file(pjoin(path, "hyperparams.json"))

        model = cls(**hyperparams)
        parameters = np.load(pjoin(path, "params.npz"))
        for param, saved_param in zip(model.parameters, parameters.values()):
            param.set_value(saved_param)

        return model
