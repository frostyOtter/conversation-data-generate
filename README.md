# Conversation Data Generator

Generates synthetic conversation data between users and assistants with realistic tool usage patterns.

## Quick Start

1. **Setup environment**:
   ```bash
   uv sync
   echo "GEMINI_API_KEY=your_api_key_here" > .env
   ```

2. **Run generator**:
   ```bash
   uv run main.py
   ```

3. **Follow prompts**:
   - Topic (e.g., "durian cultivation")
   - User persona (e.g., "durian farmer")
   - Number of conversations
   - Turns per conversation

## Output

Generates JSON files in `conversations/` directory with structured conversation data including:
- Multi-turn dialogues with realistic timing
- Tool call simulations with latency metrics
- Token usage statistics
- User metadata and conversation summaries

## Domain Model

The generator models conversations as sequences of turns, where each turn contains either a user message or assistant response with associated metadata (tokens, latency, tool calls).
