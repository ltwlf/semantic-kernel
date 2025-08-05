# Copyright (c) Microsoft. All rights reserved.

"""Reasoning parameters management for O-series models."""

from dataclasses import dataclass
from typing import Any, Dict, Optional

from semantic_kernel.ai.reasoning.detector import OSeriesModelType


@dataclass(frozen=True)
class ReasoningParameters:
    """Immutable reasoning parameters for O-series models."""
    
    reasoning_effort: str
    store: bool
    max_completion_tokens: Optional[int] = None
    temperature: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API calls."""
        params = {
            'reasoning_effort': self.reasoning_effort,
            'store': self.store
        }

        if self.max_completion_tokens is not None:
            params['max_completion_tokens'] = self.max_completion_tokens
        if self.temperature is not None:
            params['temperature'] = self.temperature

        return params


class ReasoningParametersRegistry:
    """Enterprise-grade parameters registry for O-series models."""

    # Evidence-based parameter configurations
    DEFAULT_PARAMETERS = {
        OSeriesModelType.O1_SERIES: ReasoningParameters(
            reasoning_effort='high',
            store=True,
            max_completion_tokens=32768
        ),
        OSeriesModelType.O3_SERIES: ReasoningParameters(
            reasoning_effort='high',
            store=True,
            max_completion_tokens=65536
        ),
        OSeriesModelType.O4_SERIES: ReasoningParameters(
            reasoning_effort='medium',
            store=True,
            max_completion_tokens=16384
        ),
        OSeriesModelType.UNKNOWN_O_SERIES: ReasoningParameters(
            reasoning_effort='high',
            store=True,
            max_completion_tokens=32768
        )
    }

    _custom_parameters: Dict[str, ReasoningParameters] = {}

    @classmethod
    def get_parameters(cls, model_id: str) -> ReasoningParameters:
        """
        Get optimized reasoning parameters for specific model.

        Args:
            model_id: The model identifier

        Returns:
            ReasoningParameters: Optimized parameters for the model
        """
        # Check for custom overrides first
        if model_id in cls._custom_parameters:
            return cls._custom_parameters[model_id]

        # Get model type and return default parameters
        from semantic_kernel.ai.reasoning.detector import OSeriesModelDetector
        model_type = OSeriesModelDetector.get_model_type(model_id)
        return cls.DEFAULT_PARAMETERS[model_type]

    @classmethod
    def register_custom_parameters(
        cls,
        model_id: str,
        parameters: ReasoningParameters
    ) -> None:
        """Register custom parameters for specific model."""
        cls._custom_parameters[model_id] = parameters