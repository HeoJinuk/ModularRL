"""Model composition: InputAdapter -> Representation -> Estimator -> Selector."""

import torch
from torch import Tensor, nn

from modular_rl.component.action_selector.base import BaseActionSelector
from modular_rl.component.estimator.base import BaseEstimator
from modular_rl.component.representation.base import BaseRepresentation
from modular_rl.data import ActionOutput, EstimatorOutput
from modular_rl.model.input_adapter import InputAdapter, Observation


class Model(nn.Module):
    """One facade for both tabular and neural estimation.

    forward() keeps autograd enabled for learners.
    act() prevents autograd bookkeeping but does not silently switch train/eval
    mode; callers choose model.train() or model.eval() explicitly.
    """

    def __init__(
        self,
        representation: BaseRepresentation,
        estimator: BaseEstimator,
        action_selector: BaseActionSelector | None = None,
        input_adapter: InputAdapter | None = None,
    ) -> None:
        super().__init__()
        self.representation = representation
        self.estimator = estimator
        self.action_selector = action_selector
        self.input_adapter = input_adapter or InputAdapter()
        # Tracks .to(device), .float(), .double() even for parameterless tabular models.
        self.register_buffer("_input_reference", torch.empty(0), persistent=False)

    def encode(self, observation: Observation) -> Tensor:
        x = self.input_adapter(
            observation,
            device=self._input_reference.device,
            float_dtype=self._input_reference.dtype,
        )
        features = self.representation(x)
        if not isinstance(features, Tensor):
            raise TypeError("Representation.forward() must return torch.Tensor")
        return features

    def forward(self, observation: Observation) -> EstimatorOutput:
        estimate = self.estimator(self.encode(observation))
        if not isinstance(estimate, EstimatorOutput):
            raise TypeError("Estimator.forward() must return EstimatorOutput")
        return estimate

    @torch.no_grad()
    def act(
        self, observation: Observation, *, deterministic: bool = False
    ) -> ActionOutput:
        if self.action_selector is None:
            raise RuntimeError("Model.act() requires an ActionSelector")
        estimate = self.forward(observation)
        result = self.action_selector.select(estimate, deterministic=deterministic)
        if not isinstance(result, ActionOutput):
            raise TypeError("ActionSelector.select() must return ActionOutput")
        return result

    def evaluate_q(self, observation: Observation, actions: Observation) -> EstimatorOutput:
        """Optional action-conditioned Q(s,a) evaluation, e.g. continuous SAC.

        Estimators that implement `evaluate(features, actions)` support this.
        Simple DQN and tabular Q estimators need only `forward(features)`.
        """
        evaluate = getattr(self.estimator, "evaluate", None)
        if evaluate is None or not callable(evaluate):
            raise TypeError("Estimator has no action-conditioned evaluate()")
        features = self.encode(observation)
        action_tensor = self.input_adapter(
            actions, device=self._input_reference.device, float_dtype=self._input_reference.dtype
        )
        result = evaluate(features, action_tensor)
        if not isinstance(result, EstimatorOutput):
            raise TypeError("Estimator.evaluate() must return EstimatorOutput")
        return result
