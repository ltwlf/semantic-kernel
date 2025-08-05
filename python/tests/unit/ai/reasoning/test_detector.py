# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for O-series model detection."""

import pytest

from semantic_kernel.ai.reasoning import OSeriesModelDetector, OSeriesModelType


class TestOSeriesModelDetector:
    """Comprehensive test suite for O-series model detection."""

    @pytest.mark.parametrize("model_id,expected", [
        # Current O-series models
        ("o1", True),
        ("o1", True),
        ("o1-mini", True),
        ("o3", True),
        ("o3-mini", True),
        ("o4-mini", True),
        ("o4", True),
        
        # Future O-series models (generic pattern)
        ("o5", True),
        ("o5-turbo", True),
        ("o10-advanced", True),
        ("o99-future", True),
        
        # Non-O-series models
        ("gpt-4o", False),
        ("gpt-4-turbo", False),
        ("gpt-3.5-turbo", False),
        ("claude-3", False),
        ("gemini-pro", False),
        
        # Edge cases
        ("", False),
        (None, False),
        ("O1", True),  # Case insensitive
        ("O1-PREVIEW", True),  # Case insensitive
        ("o1 ", True),  # Whitespace handling
        (" o1 ", True),  # Whitespace handling
        
        # Invalid patterns
        ("o", False),
        ("o-1", False),
        ("1o", False),
        ("o1o", False),
    ])
    def test_o_series_detection_accuracy(self, model_id: str, expected: bool):
        """Test accuracy of O-series model detection."""
        result = OSeriesModelDetector.is_o_series_model(model_id)
        assert result == expected, f"Detection failed for model: {model_id}"

    @pytest.mark.parametrize("model_id,expected_type", [
        ("o1", OSeriesModelType.O1_SERIES),
        ("o1", OSeriesModelType.O1_SERIES),
        ("o1-mini", OSeriesModelType.O1_SERIES),
        ("o3", OSeriesModelType.O3_SERIES),
        ("o3-mini", OSeriesModelType.O3_SERIES),
        ("o4-mini", OSeriesModelType.O4_SERIES),
        ("o4", OSeriesModelType.O4_SERIES),
        ("o5-future", OSeriesModelType.UNKNOWN_O_SERIES),
        ("o99", OSeriesModelType.UNKNOWN_O_SERIES),
    ])
    def test_model_type_classification(self, model_id: str, expected_type: OSeriesModelType):
        """Test model type classification accuracy."""
        result = OSeriesModelDetector.get_model_type(model_id)
        assert result == expected_type, f"Type classification failed for model: {model_id}"

    def test_get_model_type_with_non_o_series_raises_error(self):
        """Test that get_model_type raises ValueError for non-O-series models."""
        with pytest.raises(ValueError, match="is not an O-series model"):
            OSeriesModelDetector.get_model_type("gpt-4o")

    def test_case_insensitive_detection(self):
        """Test that detection is case insensitive."""
        models = ["o1", "O1", "o1-mini", "O1-MINI", "O1-Mini"]
        for model in models:
            assert OSeriesModelDetector.is_o_series_model(model), f"Case insensitive detection failed for: {model}"

    def test_whitespace_handling(self):
        """Test that detection handles whitespace correctly."""
        models = ["o1", " o1 ", "\to1\n", "  o1-mini  "]
        for model in models:
            assert OSeriesModelDetector.is_o_series_model(model), f"Whitespace handling failed for: '{model}'"