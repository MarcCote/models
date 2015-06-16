# -*- coding: utf-8 -*-

import theano.tensor as T

from nade import NADE
from utils import load_binarized_mnist
from utils import Timer
from smartpy import Trainer, tasks
from smartpy.optimizers import SGD
from smartpy.update_rules import ConstantLearningRate


def train_simple_perceptron():
    with Timer("Loading dataset"):
        trainset, validset, testset = load_binarized_mnist()

    with Timer("Creating model"):
        hidden_size = 50
        model = NADE(trainset.input_size, hidden_size)
        model.initialize()  # By default, uniform initialization.

    with Timer("Making loss symbolic graph"):
        def mean_nll(input, target):
            probs = model.fprop(input)
            nll = -T.log(probs)
            indices = T.cast(target[:, 0], dtype="int32")  # Targets are floats.
            selected_nll = nll[T.arange(target.shape[0]), indices]
            return T.mean(selected_nll)

    with Timer("Building optimizer"):
        optimizer = SGD(model, loss_fct=mean_nll, dataset=trainset, batch_size=100)
        optimizer.append_update_rule(ConstantLearningRate(0.0001))

    with Timer("Building trainer"):
        # Train for 10 epochs
        stopping_criterion = tasks.MaxEpochStopping(10)

        trainer = Trainer(optimizer, stopping_criterion=stopping_criterion)

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())

        # Print mean/stderror of classification errors.
        classif_error = tasks.ClassificationError(model.use, validset)
        trainer.append_task(tasks.Print("Validset - Classif error: {0:.1%} ± {1:.1%}", classif_error.mean, classif_error.stderror))

    with Timer("Training"):
        trainer.train()


if __name__ == "__main__":
    train_simple_perceptron()
