# Copyright (c) Microsoft. All rights reserved.

"""Service enhancement utilities for O-series reasoning models."""

import logging
from typing import Any, Dict

from semantic_kernel.ai.reasoning.detector import OSeriesModelDetector
from semantic_kernel.ai.reasoning.parameters import ReasoningParametersRegistry
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.open_ai_prompt_execution_settings import (
    OpenAIChatPromptExecutionSettings,
)

logger = logging.getLogger(__name__)


class OSeriesServiceEnhancer:
    """Enhances OpenAI services with automatic O-series reasoning support."""

    @staticmethod
    def enhance_settings_for_o_series(
        settings: OpenAIChatPromptExecutionSettings,
        ai_model_id: str,
    ) -> None:
        """Enhance chat settings with O-series reasoning parameters if applicable.
        
        Args:
            settings: The prompt execution settings to enhance
            ai_model_id: The model ID to check for O-series compatibility
        """
        if not ai_model_id or not OSeriesModelDetector.is_o_series_model(ai_model_id):
            return

        try:
            # Get optimal parameters for this O-series model
            reasoning_params = ReasoningParametersRegistry.get_parameters(ai_model_id)
            
            # Only set parameters if they're not already explicitly set
            if settings.reasoning_effort is None:
                settings.reasoning_effort = reasoning_params.reasoning_effort
                
            if settings.store is None:
                settings.store = reasoning_params.store
                
            if settings.max_completion_tokens is None and reasoning_params.max_completion_tokens is not None:
                settings.max_completion_tokens = reasoning_params.max_completion_tokens
                
            if settings.temperature is None and reasoning_params.temperature is not None:
                settings.temperature = reasoning_params.temperature

            logger.debug(f"Enhanced O-series model {ai_model_id} with reasoning parameters")
            
        except Exception as e:
            logger.warning(f"Failed to enhance settings for O-series model {ai_model_id}: {e}")

    @staticmethod
    def enhance_settings_dict_for_o_series(
        settings_dict: Dict[str, Any],
        ai_model_id: str,
    ) -> None:
        """Enhance settings dictionary with O-series reasoning parameters if applicable.
        
        This is a fallback method for cases where settings object enhancement is not sufficient.
        
        Args:
            settings_dict: The settings dictionary to enhance
            ai_model_id: The model ID to check for O-series compatibility
        """
        if not ai_model_id or not OSeriesModelDetector.is_o_series_model(ai_model_id):
            return

        try:
            # Get optimal parameters for this O-series model
            reasoning_params = ReasoningParametersRegistry.get_parameters(ai_model_id)
            reasoning_dict = reasoning_params.to_dict()
            
            # Only add parameters if they're not already set
            for key, value in reasoning_dict.items():
                if key not in settings_dict or settings_dict[key] is None:
                    settings_dict[key] = value

            logger.debug(f"Enhanced settings dict for O-series model {ai_model_id}")
            
        except Exception as e:
            logger.warning(f"Failed to enhance settings dict for O-series model {ai_model_id}: {e}")