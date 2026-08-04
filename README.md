# Simple Banking Agent

This repository contains a minimal banking assistant agent that classifies customer requests into intents and returns structured JSON output.

## What it does
- Classifies banking requests into supported intents
- Returns structured response data with confidence, reasoning, matched keywords, and metadata
- Separates concerns into an agent interface, classifier, authentication policy, and response generator
- Loads behavior and responses from JSON configuration files

## Supported intents
- `BALANCE_INQUIRY`
- `FUND_TRANSFER`
- `CARD_BLOCK`
- `PIN_RESET`
- `LOAN_QUERY`
- `UNKNOWN`

## Project structure
- `banking_agent.py` - main agent implementation
- `config/keywords.json` - intent keyword patterns
- `config/responses.json` - response templates
- `config/auth_required_intents.json` - authentication rules
- `config/agent.json` - agent metadata and confidence settings, including classifier mode
- `test_banking_agent.py` - unit tests

## Classifier modes
- `rule` (default): pattern-based regex classifier
- `semantic`: sentence-transformers embeddings for intent matching

You can switch to the semantic classifier by updating `config/agent.json`:

```json
{
  "classifier": "semantic",
  "model_name": "all-MiniLM-L6-v2"
}
```

`sentence-transformers` must be installed for the semantic mode.

## Usage

Run the agent from the command line:

```bash
python banking_agent.py "What is my account balance?"
```

### Orchestrator example

The `Orchestrator` can execute any `Agent` implementation that follows the common interface:

```python
from banking_agent import IntentAgent, Orchestrator

agent = IntentAgent()
orchestrator = Orchestrator(agent)
result = orchestrator.execute("What is my account balance?")
print(result)
```

To switch between rule-based and semantic matching, update `config/agent.json` and reload the agent.

Example output:

```json
{
  "primaryIntent": "BALANCE_INQUIRY",
  "secondaryIntent": null,
  "confidence": 0.88,
  "requiresAuthentication": true,
  "response": "I can help with your balance inquiry. Please authenticate to continue.",
  "reason": "Detected 2 matching keyword(s): account balance, balance",
  "matchedKeywords": ["balance", "account balance"],
  "metadata": {
    "agent": "IntentAgent",
    "version": "1.0",
    "timestamp": "2026-08-01T00:00:00+00:00",
    "processingTimeMs": 0
  }
}
```

## Running tests

```bash
python -m unittest -v
```

## Architecture notes
- The core entry point is the `Agent` interface and the concrete `IntentAgent`
- The orchestrator uses `agent.execute(request)` so different agent implementations can be swapped in later
- The current implementation is rule-based, but it is structured so an LLM-based or hybrid implementation can be plugged in later without changing the outer contract
