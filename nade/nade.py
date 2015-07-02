import theano
import theano.tensor as T

import os
from os.path import join as pjoin
import numpy as np

from smartpy import Model

from utils import load_dict_from_json_file, save_dict_to_json_file
from utils import WeightsInitializer
from utils import ACTIVATION_FUNCTIONS

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


class NADE(Model):
    def __init__(self,
                 input_size,
                 hidden_size,
                 hidden_activation="sigmoid",
                 tied_weights=False,
                 ordering_seed=1234,
                 *args, **kwargs):
        super(NADE, self).__init__(*args, **kwargs)

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_activation_name = hidden_activation
        self.hidden_activation = ACTIVATION_FUNCTIONS[self.hidden_activation_name]
        self.tied_weights = tied_weights
        self.ordering_seed = ordering_seed

        # Define layers weights and biases (a.k.a parameters)
        self.W = theano.shared(value=np.zeros((input_size, hidden_size), dtype=theano.config.floatX), name='W', borrow=True)
        self.bhid = theano.shared(value=np.zeros(hidden_size, dtype=theano.config.floatX), name='bhid', borrow=True)
        self.bvis = theano.shared(value=np.zeros(input_size, dtype=theano.config.floatX), name='bvis', borrow=True)

        self.V = self.W
        if not tied_weights:
            self.V = theano.shared(value=np.zeros((input_size, hidden_size), dtype=theano.config.floatX), name='V', borrow=True)

        # If needed change the input ordering.
        self.ordering = np.arange(self.input_size)
        if self.ordering_seed is not None:
            rng = np.random.RandomState(self.ordering_seed)
            rng.shuffle(self.ordering)

        self.ordering_reverse = np.argsort(self.ordering)

    @property
    def parameters(self):
        params = {}
        params[self.W.name] = self.W
        params[self.V.name] = self.V
        params[self.bhid.name] = self.bhid
        params[self.bvis.name] = self.bvis
        return params

    def save(self, path):
        if not os.path.isdir(path):
            os.makedirs(path)

        hyperparameters = {'input_size': self.input_size,
                           'hidden_size': self.hidden_size,
                           'hidden_activation_name': self.hidden_activation_name,
                           'tied_weights': self.tied_weights,
                           'ordering_seed': self.ordering_seed}

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

    def initialize(self, weights_initialization=None):
        if weights_initialization is None:
            weights_initialization = WeightsInitializer().uniform

        self.W.set_value(weights_initialization(self.W.get_value().shape))

        if not self.tied_weights:
            self.V.set_value(weights_initialization(self.V.get_value().shape))

    def fprop(self, input, return_output_preactivation=False):
        input = input[:, self.ordering]  # Does not matter if ordering is the default one, indexing is fast.
        input_times_W = input.T[:, :, None] * self.W[:, None, :]

        # This next commented line uses the SplitOp which isn't available on the GPU (Theano 0.7.0).
        # acc_input_times_W = T.concatenate([T.zeros_like(input_times_W[[0]]), T.cumsum(input_times_W, axis=0)[:-1]], axis=0)
        # Hack to stay on the GPU
        acc_input_times_W = T.cumsum(input_times_W, axis=0)
        acc_input_times_W = T.set_subtensor(acc_input_times_W[1:], acc_input_times_W[:-1])
        acc_input_times_W = T.set_subtensor(acc_input_times_W[0, :], 0.0)

        acc_input_times_W += self.bhid[None, None, :]
        h = self.hidden_activation(acc_input_times_W)

        pre_output = T.sum(h * self.V[:, None, :], axis=2) + self.bvis[:, None]
        output = T.nnet.sigmoid(pre_output)

        # Change back the ordering
        output = output.T[:, self.ordering_reverse]
        pre_output = pre_output.T[:, self.ordering_reverse]

        if return_output_preactivation:
            return output, pre_output

        return output

    def get_model_output(self, inputs):
        output, pre_output = self.fprop(inputs, return_output_preactivation=True)
        return pre_output

    # def get_nll(self, input, target):
    #     target = target[:, self.ordering]  # Does not matter if ordering is the default one, indexing is fast.
    #     output, pre_output = self.fprop(input, return_output_preactivation=True)
    #     nll = T.sum(T.nnet.softplus(-target.T * pre_output.T + (1 - target.T) * pre_output.T), axis=0)
    #     #nll = T.sum(T.nnet.softplus(-input.T * pre_output.T + (1 - input.T) * pre_output.T), axis=0)
    #     #self.sum_diff = (input-target).sum(dtype="float64")

    #     # The following does not give the same results, numerical precision error?
    #     #nll = T.sum(T.nnet.softplus(-target * pre_output + (1 - target) * pre_output), axis=1)
    #     #nll = T.sum(T.nnet.softplus(-input * pre_output + (1 - input) * pre_output), axis=1)
    #     return nll

    # def mean_nll_loss(self, input, target):
    #     nll = self.get_nll(input, target)
    #     return nll.mean()

    def build_sampling_function(self, seed=None):
        # Build sampling function
        rng = np.random.RandomState(seed)
        theano_rng = RandomStreams(rng.randint(2**30))
        bit = T.iscalar('bit')
        input = T.matrix('input')
        pre_acc = T.dot(input, self.W) + self.bhid
        h = self.hidden_activation(pre_acc)
        pre_output = T.sum(h * self.V[bit], axis=1) + self.bvis[bit]
        probs = T.nnet.sigmoid(pre_output)
        bits = theano_rng.binomial(p=probs, size=probs.shape, n=1, dtype=theano.config.floatX)
        sample_bit_plus = theano.function([input, bit], bits)

        def _sample(nb_samples):
            samples = np.zeros((nb_samples, self.input_size), dtype="float32")
            for bit in range(self.input_size):
                samples[:, bit] = sample_bit_plus(samples, bit)

            return samples
        return _sample


# def BinaryCrossEntropyNADE(Loss):
#     def __init__(self, dataset, nade):
#         super(BinaryCrossEntropyNADE, self).__init__(dataset)
#         self.nade = nade

#     def _loss_function(self, model_output):
#         target = target[:, self.ordering]  # Does not matter if ordering is the default one, indexing is fast.
#         output, pre_output = self.nade.fprop(input, return_output_preactivation=True)
#     #     nll = T.sum(T.nnet.softplus(-target.T * pre_output.T + (1 - target.T) * pre_output.T), axis=0)
#     #     #nll = T.sum(T.nnet.softplus(-input.T * pre_output.T + (1 - input.T) * pre_output.T), axis=0)
#     #     #self.sum_diff = (input-target).sum(dtype="float64")

#     #     # The following does not give the same results, numerical precision error?
#     #     #nll = T.sum(T.nnet.softplus(-target * pre_output + (1 - target) * pre_output), axis=1)
#     #     #nll = T.sum(T.nnet.softplus(-input * pre_output + (1 - input) * pre_output), axis=1)
#     #     return nll

#     # def mean_nll_loss(self, input, target):
#     #     nll = self.get_nll(input, target)
#     #     return nll.mean()