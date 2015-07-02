# -*- coding: utf-8 -*-

import theano.tensor as T

from nade import NADE
from utils import load_binarized_mnist
from utils import Timer
from smartpy import Trainer, tasks
from smartpy.optimizers import SGD
from smartpy.batch_scheduler import MiniBatchScheduler
from smartpy.losses import BinaryCrossEntropy
from smartpy.update_rules import ConstantLearningRate


def train_simple_nade():
    with Timer("Loading dataset"):
        trainset, validset, testset = load_binarized_mnist()
        # The target for distribution estimator is the input
        trainset._targets_shared = trainset.inputs
        validset._targets_shared = validset.inputs
        testset._targets_shared = testset.inputs

    with Timer("Creating model"):
        hidden_size = 50
        model = NADE(trainset.input_size, hidden_size)
        model.initialize()  # By default, uniform initialization.

    with Timer("Building optimizer"):
        optimizer = SGD(loss=BinaryCrossEntropy(model, trainset))
        optimizer.append_update_rule(ConstantLearningRate(0.0001))

    with Timer("Building trainer"):
        # Train for 10 epochs
        stopping_criterion = tasks.MaxEpochStopping(10)
        # Train using mini batches of 100 examples
        batch_scheduler = MiniBatchScheduler(trainset, 100)

        trainer = Trainer(optimizer, batch_scheduler, stopping_criterion=stopping_criterion)

        # Print time for one epoch
        trainer.append_task(tasks.PrintEpochDuration())
        trainer.append_task(tasks.PrintTrainingDuration())

        # Print mean/stderror of classification errors.
        #classif_error = tasks.ClassificationError(model.use, validset)
        #trainer.append_task(tasks.Print("Validset - NLL: {0:.1%} ± {1:.1%}", classif_error.mean, classif_error.stderror))

    with Timer("Training"):
        trainer.train()


if __name__ == "__main__":
    train_simple_nade()
