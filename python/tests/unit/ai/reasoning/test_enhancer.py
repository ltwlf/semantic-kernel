# Copyright (c) Microsoft. All rights reserved.

"""Unit tests for O-series service enhancement."""

import pytest

from semantic_kernel.ai.reasoning import OSeriesServiceEnhancer
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)


class TestOSeriesServiceEnhancer:
    """Test suite for OSeriesServiceEnhancer class."""

    def test_enhance_settings_for_o1_series(self):
        """Test enhancing settings for O1 series models."""
        settings = OpenAIChatPromptExecutionSettings()
        
        # Verify initial state
        assert settings.reasoning_effort is None
        assert settings.store is None
        assert settings.max_completion_tokens is None
        
        # Enhance settings
        OSeriesServiceEnhancer.enhance_settings_for_o_series(settings, "o1")
        
        # Verify enhancement
        assert settings.reasoning_effort == "high"
        assert settings.store is True
        assert settings.max_completion_tokens == 32768

    def test_enhance_settings_for_o4_series(self):
        """Test enhancing settings for O4 series models."""
        settings = OpenAIChatPromptExecutionSettings()
        
        # Enhance settings
        OSeriesServiceEnhancer.enhance_settings_for_o_series(settings, "o4-mini")
        
        # Verify enhancement
        assert settings.reasoning_effort == "medium"
        assert settings.store is True
        assert settings.max_completion_tokens == 16384

    def test_enhance_settings_preserves_existing_values(self):
        """Test that enhancement preserves existing values."""
        settings = OpenAIChatPromptExecutionSettings(
            reasoning_effort="low",
            store=False,
            max_completion_tokens=1024,
            temperature=0.5
        )
        
        # Enhance settings
        OSeriesServiceEnhancer.enhance_settings_for_o_series(settings, "o1")
        
        # Verify existing values are preserved
        assert settings.reasoning_effort == "low"  # Not changed
        assert settings.store is False  # Not changed
        assert settings.max_completion_tokens == 1024  # Not changed
        assert settings.temperature == 0.5  # Not changed

    def test_enhance_settings_with_partial_existing_values(self):
        """Test enhancement with some existing values."""
        settings = OpenAIChatPromptExecutionSettings(
            reasoning_effort="low",  # This should be preserved
            # store and max_completion_tokens are None, should be enhanced
        )
        
        # Enhance settings
        OSeriesServiceEnhancer.enhance_settings_for_o_series(settings, "o1")
        
        # Verify mixed behavior
        assert settings.reasoning_effort == "low"  # Preserved
        assert settings.store is True  # Enhanced
        assert settings.max_completion_tokens == 32768  # Enhanced

    def test_enhance_settings_for_non_o_series_does_nothing(self):
        """Test that enhancing settings for non-O-series models does nothing."""
        settings = OpenAIChatPromptExecutionSettings()
        
        # Enhance settings for non-O-series model
        OSeriesServiceEnhancer.enhance_settings_for_o_series(settings, "gpt-4o")
        
        # Verify no changes
        assert settings.reasoning_effort is None
        assert settings.store is None
        assert settings.max_completion_tokens is None

    def test_enhance_settings_with_empty_model_id(self):
        """Test enhancement with empty or None model ID."""
        settings = OpenAIChatPromptExecutionSettings()
        
        # Test with None
        OSeriesServiceEnhancer.enhance_settings_for_o_series(settings, None)
        assert settings.reasoning_effort is None
        
        # Test with empty string
        OSeriesServiceEnhancer.enhance_settings_for_o_series(settings, "")
        assert settings.reasoning_effort is None

    def test_enhance_settings_dict_for_o_series(self):
        """Test enhancing settings dictionary for O-series models."""
        settings_dict = {}
        
        # Enhance dictionary
        OSeriesServiceEnhancer.enhance_settings_dict_for_o_series(settings_dict, "o1")
        
        # Verify enhancement
        assert settings_dict["reasoning_effort"] == "high"
        assert settings_dict["store"] is True
        assert settings_dict["max_completion_tokens"] == 32768

    def test_enhance_settings_dict_preserves_existing_values(self):
        """Test that dictionary enhancement preserves existing values."""
        settings_dict = {
            "reasoning_effort": "low",
            "temperature": 0.5
        }
        
        # Enhance dictionary
        OSeriesServiceEnhancer.enhance_settings_dict_for_o_series(settings_dict, "o1")
        
        # Verify mixed behavior
        assert settings_dict["reasoning_effort"] == "low"  # Preserved
        assert settings_dict["store"] is True  # Added
        assert settings_dict["max_completion_tokens"] == 32768  # Added
        assert settings_dict["temperature"] == 0.5  # Preserved

    def test_enhance_settings_dict_for_non_o_series_does_nothing(self):
        """Test that dictionary enhancement for non-O-series models does nothing."""
        settings_dict = {"temperature": 0.7}
        
        # Enhance for non-O-series model
        OSeriesServiceEnhancer.enhance_settings_dict_for_o_series(settings_dict, "gpt-4o")
        
        # Verify no changes
        assert settings_dict == {"temperature": 0.7}