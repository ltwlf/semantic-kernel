# Copyright (c) Microsoft. All rights reserved.

"""Universal O-series model detection engine for OpenAI reasoning models."""

import re
from enum import Enum
from typing import Optional


class OSeriesModelType(Enum):
    """Enumeration of O-series model categories."""
    
    O1_SERIES = "o1"
    O3_SERIES = "o3"
    O4_SERIES = "o4"
    UNKNOWN_O_SERIES = "unknown_o"


class OSeriesModelDetector:
    """Production-grade O-series model detection system.
    
    This class provides comprehensive detection for OpenAI's O-series reasoning models,
    including current models (O1, O3, O4-mini) and future-proofing for new O-series releases.
    """

    # Comprehensive O-series model patterns
    O_SERIES_PATTERNS = {
        OSeriesModelType.O1_SERIES: [
            r'^o1$', 
            r'^o1-mini$',
            r'^o1-2024-12-17$'
        ],
        OSeriesModelType.O3_SERIES: [
            r'^o3$', 
            r'^o3-mini$', 
            r'^o3-preview$'
        ],
        OSeriesModelType.O4_SERIES: [
            r'^o4-mini$', 
            r'^o4-preview$', 
            r'^o4$'
        ]
    }

    # Future-proofing pattern for unknown O-series models
    GENERIC_O_PATTERN = r'^o\d+(-\w+)*$'

    @classmethod
    def is_o_series_model(cls, model_id: Optional[str]) -> bool:
        """
        Determine if model is an O-series reasoning model.

        Args:
            model_id: The model identifier to check

        Returns:
            bool: True if model is O-series, False otherwise
        """
        if not model_id:
            return False

        normalized_id = model_id.lower().strip()

        # Check known O-series patterns first
        for model_type, patterns in cls.O_SERIES_PATTERNS.items():
            if any(re.match(pattern, normalized_id) for pattern in patterns):
                return True

        # Check generic O-series pattern for future models
        return bool(re.match(cls.GENERIC_O_PATTERN, normalized_id))

    @classmethod
    def get_model_type(cls, model_id: str) -> OSeriesModelType:
        """
        Get the specific O-series model type.

        Args:
            model_id: The model identifier

        Returns:
            OSeriesModelType: The detected model type
            
        Raises:
            ValueError: If model is not an O-series model
        """
        if not cls.is_o_series_model(model_id):
            raise ValueError(f"Model {model_id} is not an O-series model")

        normalized_id = model_id.lower().strip()

        for model_type, patterns in cls.O_SERIES_PATTERNS.items():
            if any(re.match(pattern, normalized_id) for pattern in patterns):
                return model_type

        return OSeriesModelType.UNKNOWN_O_SERIES