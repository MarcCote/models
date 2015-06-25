import os
import numpy as np
from os.path import join as pjoin

import theano
import theano.tensor as T

from smartpy import Model
from utils import load_dict_from_json_file, save_dict_to_json_file
from utils import WeightsInitializer


dtype = theano.config.floatX


class Perceptron(Model):
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.W = theano.shared(value=np.zeros((input_size, output_size), dtype=dtype), name='W', borrow=True)
        self.b = theano.shared(value=np.zeros(output_size, dtype=dtype), name='b', borrow=True)

    def initialize(self, weights_initializer=None):
        if weights_initializer is None:
            weights_initializer = WeightsInitializer().uniform

        self.W.set_value(weights_initializer(self.W.get_value().shape))

    @property
    def parameters(self):
        return {'W': self.W, 'b': self.b}

    def fprop(self, X):
        preactivation = T.dot(X, self.W) + self.b
        probs = T.nnet.softmax(preactivation)
        return probs

    def use(self, X):
        probs = self.fprop(X)
        return T.argmax(probs, axis=1, keepdims=True)

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        hyperparameters = {'input_size': self.input_size,
                           'output_size': self.output_size}
        save_dict_to_json_file(pjoin(path, "meta.json"), {"name": self.__class__.__name__})
        save_dict_to_json_file(pjoin(path, "hyperparams.json"), hyperparameters)

        params = {param_name: param.get_value() for param_name, param in self.parameters.items()}
        np.savez(pjoin(path, "params.npz"), **params)

    @classmethod
    def load(cls, path):
        meta = load_dict_from_json_file(pjoin(path, "meta.json"))
        assert meta['name'] == cls.__name__

        hyperparams = load_dict_from_json_file(pjoin(path, "hyperparams.json"))

        model = cls(**hyperparams)
        parameters = np.load(pjoin(path, "params.npz"))
        for param_name, param in model.parameters.items():
            param.set_value(parameters[param_name])

        return model
