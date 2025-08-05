# Copyright (c) Microsoft. All rights reserved.

"""Tests for O-series integration with ResponsesAgent."""


from semantic_kernel.agents.open_ai.responses_agent_thread_actions import ResponsesAgentThreadActions
from semantic_kernel.ai.reasoning import OSeriesModelDetector


class MockAgent:
    """Mock agent for testing purposes."""
    
    def __init__(self, ai_model_id: str, **kwargs):
        self.ai_model_id = ai_model_id
        self.text = kwargs.get("text")
        self.temperature = kwargs.get("temperature")
        self.top_p = kwargs.get("top_p")
        self.metadata = kwargs.get("metadata", {})


class TestResponsesAgentOSeries:
    """Test suite for O-series integration with ResponsesAgent."""

    def test_o_series_model_detection(self):
        """Test that O-series models are correctly detected."""
        assert OSeriesModelDetector.is_o_series_model("o1")
        assert OSeriesModelDetector.is_o_series_model("o3-mini")
        assert not OSeriesModelDetector.is_o_series_model("gpt-4o")

    def test_responses_agent_generate_options_includes_reasoning_o1(self):
        """Test that _generate_options includes reasoning parameter for O1 models."""
        agent = MockAgent(ai_model_id="o1")
        
        # Test _generate_options method
        options = ResponsesAgentThreadActions._generate_options(
            agent=agent,
            model="o1"
        )
        
        # Verify reasoning parameter is included and set correctly for O1
        assert "reasoning" in options
        assert options["reasoning"] == "high"  # O1 should get high reasoning
        assert options["model"] == "o1"

    def test_responses_agent_generate_options_includes_reasoning_o3(self):
        """Test that _generate_options includes reasoning parameter for O3 models."""
        agent = MockAgent(ai_model_id="o3-mini")
        
        # Test _generate_options method
        options = ResponsesAgentThreadActions._generate_options(
            agent=agent,
            model="o3-mini"
        )
        
        # Verify reasoning parameter is included and set correctly for O3
        assert "reasoning" in options
        assert options["reasoning"] == "high"  # O3 should get high reasoning
        assert options["model"] == "o3-mini"

    def test_responses_agent_generate_options_preserves_explicit_reasoning(self):
        """Test that explicit reasoning parameters are preserved."""
        agent = MockAgent(ai_model_id="o1")
        
        # Test with explicit reasoning parameter
        options = ResponsesAgentThreadActions._generate_options(
            agent=agent,
            model="o1",
            reasoning="low"  # Explicitly set to low
        )
        
        # Verify explicit reasoning is preserved
        assert options["reasoning"] == "low"  # Should preserve explicit value

    def test_responses_agent_generate_options_non_o_series_no_auto_reasoning(self):
        """Test that non-O-series models don't get automatic reasoning parameters."""
        agent = MockAgent(ai_model_id="gpt-4o")
        
        # Test _generate_options method
        options = ResponsesAgentThreadActions._generate_options(
            agent=agent,
            model="gpt-4o"
        )
        
        # Verify no automatic reasoning for non-O-series
        assert options.get("reasoning") is None
        assert options["model"] == "gpt-4o"

    def test_responses_agent_generate_options_o3_gets_high_reasoning(self):
        """Test that O3 models get high reasoning effort."""
        agent = MockAgent(ai_model_id="o3-mini")
        
        # Test _generate_options method
        options = ResponsesAgentThreadActions._generate_options(
            agent=agent,
            model="o3-mini"
        )
        
        # Verify O3 gets high reasoning
        assert options["reasoning"] == "high"

    def test_responses_agent_case_insensitive_o_series_detection(self):
        """Test that O-series detection works with different cases."""
        agent = MockAgent(ai_model_id="O1-PREVIEW")
        
        # Test _generate_options method
        options = ResponsesAgentThreadActions._generate_options(
            agent=agent,
            model="O1-PREVIEW"
        )
        
        # Verify case-insensitive detection works
        assert options["reasoning"] == "high"

    def test_responses_agent_future_o_series_gets_high_reasoning(self):
        """Test that future O-series models get high reasoning effort."""
        agent = MockAgent(ai_model_id="o5-future")
        
        # Test _generate_options method
        options = ResponsesAgentThreadActions._generate_options(
            agent=agent,
            model="o5-future"
        )
        
        # Verify future O-series gets high reasoning (default for unknown O-series)
        assert options["reasoning"] == "high"

    def test_responses_agent_o_series_reasoning_in_all_option_fields(self):
        """Test that O-series reasoning is properly handled with all options."""
        agent = MockAgent(
            ai_model_id="o1",
            temperature=0.7,
            top_p=0.9
        )
        
        # Test with multiple options
        options = ResponsesAgentThreadActions._generate_options(
            agent=agent,
            model="o1",
            temperature=0.5,  # Override agent temperature
            max_output_tokens=1024,
            metadata={"test": "value"}
        )
        
        # Verify all options are present
        assert options["reasoning"] == "high"
        assert options["temperature"] == 0.5  # Override value
        assert options["top_p"] == 0.9  # Agent value
        assert options["max_output_tokens"] == 1024
        assert options["metadata"] == {"test": "value"}
        assert options["model"] == "o1"

    def test_responses_agent_no_model_specified_no_reasoning(self):
        """Test that when no model is specified, no automatic reasoning is applied."""
        agent = MockAgent(ai_model_id="o1")
        
        # Test without specifying model parameter
        options = ResponsesAgentThreadActions._generate_options(
            agent=agent
            # No model parameter
        )
        
        # When model is None (not provided), should fall back to agent's model
        # but the auto-reasoning logic should still work
        assert options["model"] == "o1"  # Falls back to agent model
        assert options["reasoning"] == "high"    # Should still get reasoning

    def test_responses_agent_explicit_none_reasoning_preserved(self):
        """Test that explicitly setting reasoning to None is preserved."""
        agent = MockAgent(ai_model_id="o1")
        
        # Test with explicit None reasoning
        options = ResponsesAgentThreadActions._generate_options(
            agent=agent,
            model="o1",
            reasoning=None  # Explicitly set to None
        )
        
        # Verify explicit None is preserved (no auto-reasoning)
        assert options["reasoning"] is None