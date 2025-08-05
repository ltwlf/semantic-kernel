# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for O-series reasoning parameters management."""

import pytest

from semantic_kernel.ai.reasoning import (
    ReasoningParameters,
    ReasoningParametersRegistry,
)


class TestReasoningParameters:
    """Test suite for ReasoningParameters class."""

    def test_reasoning_parameters_creation(self):
        """Test creating ReasoningParameters with all fields."""
        params = ReasoningParameters(
            reasoning_effort="high",
            store=True,
            max_completion_tokens=32768,
            temperature=0.7
        )
        
        assert params.reasoning_effort == "high"
        assert params.store is True
        assert params.max_completion_tokens == 32768
        assert params.temperature == 0.7

    def test_reasoning_parameters_to_dict(self):
        """Test converting ReasoningParameters to dictionary."""
        params = ReasoningParameters(
            reasoning_effort="medium",
            store=False,
            max_completion_tokens=16384,
            temperature=0.5
        )
        
        result = params.to_dict()
        expected = {
            'reasoning_effort': 'medium',
            'store': False,
            'max_completion_tokens': 16384,
            'temperature': 0.5
        }
        
        assert result == expected

    def test_reasoning_parameters_to_dict_with_none_values(self):
        """Test converting ReasoningParameters to dict with None values."""
        params = ReasoningParameters(
            reasoning_effort="low",
            store=True,
            max_completion_tokens=None,
            temperature=None
        )
        
        result = params.to_dict()
        expected = {
            'reasoning_effort': 'low',
            'store': True
        }
        
        assert result == expected

    def test_reasoning_parameters_immutable(self):
        """Test that ReasoningParameters is immutable."""
        params = ReasoningParameters(
            reasoning_effort="high",
            store=True
        )
        
        with pytest.raises(AttributeError):
            params.reasoning_effort = "low"  # Should raise error due to frozen dataclass


class TestReasoningParametersRegistry:
    """Test suite for ReasoningParametersRegistry class."""

    def test_get_parameters_for_o1_series(self):
        """Test getting parameters for O1 series models."""
        models = ["o1", "o1-mini", "o1-mini"]
        
        for model in models:
            params = ReasoningParametersRegistry.get_parameters(model)
            assert params.reasoning_effort == "high"
            assert params.store is True
            assert params.max_completion_tokens == 32768

    def test_get_parameters_for_o3_series(self):
        """Test getting parameters for O3 series models."""
        models = ["o3", "o3-mini"]
        
        for model in models:
            params = ReasoningParametersRegistry.get_parameters(model)
            assert params.reasoning_effort == "high"
            assert params.store is True
            assert params.max_completion_tokens == 32768

    def test_get_parameters_for_unknown_o_series(self):
        """Test getting parameters for unknown O-series models."""
        models = ["o5", "o10-future", "o99-advanced"]
        
        for model in models:
            params = ReasoningParametersRegistry.get_parameters(model)
            assert params.reasoning_effort == "high"
            assert params.store is True
            assert params.max_completion_tokens == 16384

    def test_get_parameters_for_non_o_series_raises_error(self):
        """Test that getting parameters for non-O-series models raises error."""
        with pytest.raises(ValueError, match="is not an O-series model"):
            ReasoningParametersRegistry.get_parameters("gpt-4o")

    def test_register_custom_parameters(self):
        """Test registering custom parameters for a model."""
        custom_params = ReasoningParameters(
            reasoning_effort="low",
            store=False,
            max_completion_tokens=8192,
            temperature=0.1
        )
        
        # Register custom parameters
        ReasoningParametersRegistry.register_custom_parameters("o1-custom", custom_params)
        
        # Retrieve and verify
        retrieved_params = ReasoningParametersRegistry.get_parameters("o1-custom")
        assert retrieved_params == custom_params
        assert retrieved_params.reasoning_effort == "low"
        assert retrieved_params.store is False
        assert retrieved_params.max_completion_tokens == 8192
        assert retrieved_params.temperature == 0.1

    def test_custom_parameters_override_defaults(self):
        """Test that custom parameters override default parameters."""
        # First verify default parameters
        default_params = ReasoningParametersRegistry.get_parameters("o1")
        assert default_params.reasoning_effort == "high"
        
        # Register custom parameters for the same model
        custom_params = ReasoningParameters(
            reasoning_effort="low",
            store=False
        )
        ReasoningParametersRegistry.register_custom_parameters("o1", custom_params)
        
        # Verify custom parameters are returned
        retrieved_params = ReasoningParametersRegistry.get_parameters("o1")
        assert retrieved_params == custom_params
        assert retrieved_params.reasoning_effort == "low"
        assert retrieved_params.store is False

    def teardown_method(self):
        """Clean up custom parameters after each test."""
        ReasoningParametersRegistry._custom_parameters.clear()