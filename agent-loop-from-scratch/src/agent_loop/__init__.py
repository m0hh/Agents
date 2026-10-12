"""
agent_loop: an AI agent built from scratch, with no agent frameworks.

Modules (the phase each one arrives in is in brackets):
    config        [P1]  settings loaded from environment variables / .env
    api_client    [P1]  raw HTTP calls to the Claude Messages API
    hello         [P1]  send one message and dissect the response
    payload       [P2]  builds the request JSON body (shared by every program)
    usage         [P2]  token counts → dollars
    conversation  [P2]  the append-only message history (the agent's memory)
    agent         [P2]  one user turn → API call → result (the loop arrives in P4)
    cli_args      [P2]  command-line flags shared by every program
    cli           [P2]  the interactive chat REPL (`uv run agent`)

See PROJECT_PLAN.md in the project root for the full roadmap.
"""
