"""Neural TD residuals and unreduced-loss aggregation, without optimization."""

from collections.abc import Callable

from torch import Tensor, nn

from modular_rl.data.updates import TDLossOutput


class TDLossUpdateRule:
    """Compute signed TD errors and a scalar mean loss from supplied tensors.

    prediction/target are nonempty floating Tensors with identical shape,
    dtype and device, including scalar () or additional batch dimensions.
    Optional weights have the same contract. loss_fn is callable and returns
    a floating loss Tensor with the input shape; default: MSELoss(reduction="none").

    No input is detached or modified. Learner selects the bootstrap target,
    controls its graph, runs the optimizer and detaches errors for metrics.
    Weights are used as supplied, without correction or normalization.
    """

    def __init__(self, loss_fn: Callable[[Tensor, Tensor], Tensor] | None = None) -> None:
        if loss_fn is not None and not callable(loss_fn):
            raise TypeError("loss_fn must be callable or None")
        self.loss_fn = nn.MSELoss(reduction="none") if loss_fn is None else loss_fn

    def compute(
        self, prediction: Tensor, target: Tensor, *, weights: Tensor | None = None
    ) -> TDLossOutput:
        self._validate(prediction, target, weights)
        td_error = target - prediction
        per_sample_loss = self.loss_fn(prediction, target)
        self._validate_loss(per_sample_loss, prediction)
        if weights is not None:
            per_sample_loss = weights * per_sample_loss
        return TDLossOutput(td_error=td_error, loss=per_sample_loss.mean())

    @staticmethod
    def _validate(prediction: Tensor, target: Tensor, weights: Tensor | None) -> None:
        if not isinstance(prediction, Tensor) or not isinstance(target, Tensor):
            raise TypeError("prediction and target must be Tensors")
        if prediction.shape != target.shape:
            raise ValueError("prediction and target must have the same shape")
        if not prediction.is_floating_point() or not target.is_floating_point():
            raise TypeError("prediction and target must have floating dtype")
        if prediction.dtype != target.dtype:
            raise TypeError("prediction and target must have the same dtype")
        if prediction.device != target.device:
            raise ValueError("prediction and target must be on the same device")
        if prediction.numel() == 0:
            raise ValueError("prediction must be nonempty to calculate a mean loss")
        if weights is not None:
            if not isinstance(weights, Tensor):
                raise TypeError("weights must be a Tensor or None")
            if weights.shape != prediction.shape:
                raise ValueError("weights and prediction must have the same shape")
            if weights.dtype != prediction.dtype:
                raise TypeError("weights and prediction must have the same floating dtype")
            if weights.device != prediction.device:
                raise ValueError("weights and prediction must be on the same device")

    @staticmethod
    def _validate_loss(loss: Tensor, prediction: Tensor) -> None:
        if not isinstance(loss, Tensor) or not loss.is_floating_point():
            raise TypeError("loss_fn must return a floating loss Tensor")
        if loss.shape != prediction.shape:
            raise ValueError("loss_fn must return the input shape; use reduction='none'")
