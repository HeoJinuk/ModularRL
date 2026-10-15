"""Temporarily evaluate bootstrap values without changing caller module modes."""

from contextlib import contextmanager


@contextmanager
def evaluating(model):
    """Disable Dropout/BatchNorm updates, then restore even mixed child modes."""
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training
