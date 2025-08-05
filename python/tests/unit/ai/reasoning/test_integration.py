# Copyright (c) Microsoft. All rights reserved.

"""Integration tests for O-series models with OpenAI services."""

from unittest.mock import patch

from semantic_kernel.ai.reasoning import OSeriesModelDetector
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import AzureChatCompletion
from semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion import OpenAIChatCompletion


class TestOSeriesServiceIntegration:
    """Integration tests for O-series models across OpenAI services."""

    def test_azure_chat_completion_with_o1_model_initialization(self):
        """Test that Azure chat completion correctly initializes with O1 model."""
        with patch("semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion.AsyncAzureOpenAI"):
            service = AzureChatCompletion(
                deployment_name="o1",
                api_key="test-key",
                endpoint="https://test.openai.azure.com"
            )
            
            # Verify the service recognizes the O-series model
            assert OSeriesModelDetector.is_o_series_model(service.ai_model_id)
            assert service.ai_model_id == "o1"

    def test_openai_chat_completion_with_o_series_initialization(self):
        """Test that OpenAI chat completion correctly initializes with O-series model."""
        with patch("semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion.AsyncOpenAI"):
            service = OpenAIChatCompletion(
                ai_model_id="o3-mini",
                api_key="test-key"
            )
            
            # Verify the service recognizes the O-series model
            assert OSeriesModelDetector.is_o_series_model(service.ai_model_id)
            assert service.ai_model_id == "o3-mini"

    def test_azure_chat_completion_settings_enhancement_o1(self):
        """Test that Azure chat completion enhances settings for O1 models."""
        with patch("semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion.AsyncAzureOpenAI"):
            service = AzureChatCompletion(
                deployment_name="o1",
                api_key="test-key",
                endpoint="https://test.openai.azure.com"
            )
            
            # Create settings without reasoning parameters
            settings = AzureChatPromptExecutionSettings()
            assert settings.reasoning_effort is None
            assert settings.store is None
            
            # Test direct enhancement via the enhancer
            from semantic_kernel.ai.reasoning.enhancer import OSeriesServiceEnhancer
            OSeriesServiceEnhancer.enhance_settings_for_o_series(settings, service.ai_model_id)
            
            # Verify O1-specific parameters were applied
            assert settings.reasoning_effort == "high"
            assert settings.store is True
            assert settings.max_completion_tokens == 16384

    def test_azure_chat_completion_settings_enhancement_o3(self):
        """Test that Azure chat completion enhances settings for O3 models."""
        with patch("semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion.AsyncAzureOpenAI"):
            service = AzureChatCompletion(
                deployment_name="o3-mini",
                api_key="test-key",
                endpoint="https://test.openai.azure.com"
            )
            
            # Create settings
            settings = AzureChatPromptExecutionSettings()
            
            # Test enhancement
            from semantic_kernel.ai.reasoning.enhancer import OSeriesServiceEnhancer
            OSeriesServiceEnhancer.enhance_settings_for_o_series(settings, service.ai_model_id)
            
            # Verify O3-specific parameters
            assert settings.reasoning_effort == "high"
            assert settings.store is True
            assert settings.max_completion_tokens == 32768

    def test_openai_chat_completion_settings_enhancement_o3(self):
        """Test that OpenAI chat completion enhances settings for O3 models."""
        with patch("semantic_kernel.connectors.ai.open_ai.services.open_ai_chat_completion.AsyncOpenAI"):
            service = OpenAIChatCompletion(
                ai_model_id="o3-mini",
                api_key="test-key"
            )
            
            # Create settings
            settings = OpenAIChatPromptExecutionSettings()
            
            # Test enhancement
            from semantic_kernel.ai.reasoning.enhancer import OSeriesServiceEnhancer
            OSeriesServiceEnhancer.enhance_settings_for_o_series(settings, service.ai_model_id)
            
            # Verify O3-specific parameters
            assert settings.reasoning_effort == "high"
            assert settings.store is True
            assert settings.max_completion_tokens == 65536

    def test_non_o_series_model_no_enhancement(self):
        """Test that non-O-series models are not enhanced."""
        with patch("semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion.AsyncAzureOpenAI"):
            service = AzureChatCompletion(
                deployment_name="gpt-4o",
                api_key="test-key",
                endpoint="https://test.openai.azure.com"
            )
            
            # Verify it's not an O-series model
            assert not OSeriesModelDetector.is_o_series_model(service.ai_model_id)
            
            # Create settings
            settings = AzureChatPromptExecutionSettings()
            
            # Test that enhancement doesn't happen
            from semantic_kernel.ai.reasoning.enhancer import OSeriesServiceEnhancer
            OSeriesServiceEnhancer.enhance_settings_for_o_series(settings, service.ai_model_id)
            
            # Verify no enhancement occurred
            assert settings.reasoning_effort is None
            assert settings.store is None
            assert settings.max_completion_tokens is None

    def test_existing_reasoning_parameters_preserved(self):
        """Test that existing reasoning parameters are preserved during enhancement."""
        with patch("semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion.AsyncAzureOpenAI"):
            service = AzureChatCompletion(
                deployment_name="o1",
                api_key="test-key",
                endpoint="https://test.openai.azure.com"
            )
            
            # Create settings with existing parameters
            settings = AzureChatPromptExecutionSettings(
                reasoning_effort="low",
                store=False,
                max_completion_tokens=1024,
                temperature=0.5
            )
            
            # Test enhancement
            from semantic_kernel.ai.reasoning.enhancer import OSeriesServiceEnhancer
            OSeriesServiceEnhancer.enhance_settings_for_o_series(settings, service.ai_model_id)
            
            # Verify existing parameters were preserved
            assert settings.reasoning_effort == "low"  # Not overridden
            assert settings.store is False  # Not overridden
            assert settings.max_completion_tokens == 1024  # Not overridden
            assert settings.temperature == 0.5  # Not overridden

    def test_case_insensitive_model_detection_in_services(self):
        """Test that services work with case-insensitive O-series model names."""
        with patch("semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion.AsyncAzureOpenAI"):
            # Test uppercase model name
            service = AzureChatCompletion(
                deployment_name="O1-PREVIEW",
                api_key="test-key",
                endpoint="https://test.openai.azure.com"
            )
            
            # Verify detection still works
            assert OSeriesModelDetector.is_o_series_model(service.ai_model_id)
            
            # Test enhancement still works
            settings = AzureChatPromptExecutionSettings()
            from semantic_kernel.ai.reasoning.enhancer import OSeriesServiceEnhancer
            OSeriesServiceEnhancer.enhance_settings_for_o_series(settings, service.ai_model_id)
            
            assert settings.reasoning_effort == "high"
            assert settings.store is True