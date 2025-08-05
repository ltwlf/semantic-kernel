# Reasoning Models

This folder contains samples demonstrating how to use OpenAI's reasoning models (O-series) with Semantic Kernel.

## Samples

- **simple_reasoning.py** - Basic usage of O1 reasoning models with chat completion
- **simple_reasoning_function_calling.py** - Using reasoning models with function calling
- **o_series_models.py** - Comprehensive demonstration of O-series model integration features

## Prerequisites

To run these samples, you'll need:

- An OpenAI API key or Azure OpenAI endpoint
- Access to O-series reasoning models (o1, o1-mini, o3, o4-mini, etc.)
- Python 3.10+ with Semantic Kernel installed

## Key Features Demonstrated

### Automatic O-Series Detection
The samples show how Semantic Kernel automatically detects O-series models and applies optimal reasoning parameters.

### Reasoning Parameters
O-series models support specialized parameters:
- `reasoning_effort`: Controls the model's reasoning intensity (low, medium, high)
- `max_completion_tokens`: Maximum tokens for the completion
- `store`: Whether to store the conversation

### Developer Messages
O-series models use developer messages instead of system messages for instructions.

## Running the Samples

1. Set up your environment variables (see individual sample files for details)
2. Install dependencies: `pip install semantic-kernel`
3. Run any sample: `python simple_reasoning.py`

## Learn More

For more information about reasoning models, visit:
- [OpenAI O-series Documentation](https://platform.openai.com/docs/guides/reasoning)
- [Semantic Kernel Documentation](https://learn.microsoft.com/semantic-kernel/)