"""Strands AgentCore Tools - Complete toolkit for AWS Bedrock AgentCore.

This package provides comprehensive tools for managing AWS Bedrock AgentCore agents:

Core Tools:
-----------
- configure: Configure and prepare agents for deployment
- launch: Deploy agents to AgentCore runtime
- invoke: Invoke deployed agents with payloads
- status: Check agent runtime and endpoint health
- logs: Access CloudWatch logs for debugging
- agents: List and discover deployed agents
- memory: Manage AgentCore Memory resources
- identity: Manage OAuth2, API keys, and credentials
- session: Manage runtime sessions

Quick Start:
-----------
```python
from strands_agentcore_tools import (
    configure, launch, invoke, status, logs,
    agents, memory, identity, session
)

# 1. Configure agent
configure(
    action="configure",
    entrypoint="agent.py",
    agent_name="my-agent"
)

# 2. Deploy to AgentCore
launch(action="launch", agent_name="my-agent")

# 3. Invoke deployed agent
invoke(
    agent_arn="arn:aws:bedrock-agentcore:...",
    payload='{"prompt": "Hello!"}'
)

# 4. Check status
status(agent_id="my-agent-abc123")

# 5. View logs
logs(agent_name="my-agent", action="recent")
```

Installation:
------------
```bash
pip install strands-agentcore-tools
```

Requirements:
------------
- boto3
- pyyaml
- strands-agents

Environment Setup:
-----------------
Configure AWS credentials:
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
export AWS_DEFAULT_REGION=us-west-2
```

Or use AWS CLI:
```bash
aws configure
```

Tool Overview:
-------------

**configure** - Prepare agent for deployment:
```python
configure(
    action="configure",
    entrypoint="agent.py",
    agent_name="my-agent",
    memory_mode="STM_AND_LTM",
    enable_observability=True
)
```

**launch** - Deploy agent to AgentCore:
```python
launch(
    action="launch",
    agent_name="my-agent",
    mode="codebuild"  # Cloud build (default)
)
```

**invoke** - Call deployed agent:
```python
invoke(
    agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/agent-abc",
    payload='{"prompt": "What can you do?"}'
)
```

**status** - Check agent health:
```python
status(agent_id="my-agent-abc123")
```

**logs** - Debug with CloudWatch:
```python
logs(agent_name="my-agent", action="recent")
logs(agent_name="my-agent", action="search", filter_pattern="ERROR")
```

**agents** - Discover deployed agents:
```python
agents(action="list")
agents(action="get", agent_id="my-agent-abc123")
```

**memory** - Manage conversation memory:
```python
memory(action="create", name="my-memory", strategies=[...])
memory(action="retrieve", memory_id="mem-123", namespace="/users/...")
```

**identity** - OAuth2 and credentials:
```python
identity(
    action="create",
    provider_type="oauth2",
    name="slack-oauth",
    vendor="SlackOauth2",
    client_id="...",
    client_secret="..."
)
```

**session** - Manage runtime sessions:
```python
session(
    action="stop",
    agent_arn="arn:aws:bedrock-agentcore:...",
    session_id="session-123"
)
```

Complete Workflow Example:
-------------------------
```python
from strands_agentcore_tools import (
    configure, launch, invoke, status, logs
)

# 1. Configure agent
configure_result = configure(
    action="configure",
    entrypoint="agent.py",
    agent_name="research-agent",
    memory_mode="STM_AND_LTM",
    enable_observability=True
)
print("✅ Configured:", configure_result)

# 2. Deploy to AgentCore
launch_result = launch(
    action="launch",
    agent_name="research-agent"
)
print("✅ Deployed:", launch_result)

# 3. Check deployment status
status_result = status(agent_id="research-agent-abc123")
print("✅ Status:", status_result)

# 4. Invoke agent
invoke_result = invoke(
    agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/research-agent",
    payload='{"prompt": "Summarize latest AI research"}'
)
print("✅ Response:", invoke_result)

# 5. Check logs
logs_result = logs(
    agent_name="research-agent",
    action="recent",
    limit=50
)
print("✅ Logs:", logs_result)
```

Documentation:
-------------
For detailed documentation on each tool, see:
- https://github.com/cagataycali/strands-agentcore-tools
- Tool docstrings: help(configure), help(launch), etc.

Support:
--------
- GitHub Issues: https://github.com/cagataycali/strands-agentcore-tools/issues
- Documentation: https://github.com/cagataycali/strands-agentcore-tools/blob/main/README.md

License:
-------
MIT License - see LICENSE file for details
"""

__version__ = "0.1.2"
__author__ = "Cagatay Cali"
__email__ = "cagataycali@icloud.com"

# Import all tools from their respective modules
from .agents import agents
from .configure import configure
from .identity import identity
from .invoke import invoke
from .launch import launch
from .logs import logs
from .memory import memory
from .session import session
from .status import status

# Public API - tools available when importing package
__all__ = [
    "agents",
    "configure",
    "identity",
    "invoke",
    "launch",
    "logs",
    "memory",
    "session",
    "status",
    "__version__",
    "__author__",
    "__email__",
]


def get_version():
    """Get package version."""
    return __version__


def list_tools():
    """List all available AgentCore tools."""
    return [
        "agents - List and manage agent runtimes",
        "configure - Configure agents for deployment",
        "identity - Manage OAuth2, API keys, credentials",
        "invoke - Invoke deployed agents",
        "launch - Deploy agents to AgentCore",
        "logs - Access CloudWatch logs",
        "memory - Manage AgentCore Memory resources",
        "session - Manage runtime sessions",
        "status - Check agent health and status",
    ]


# Package metadata
__doc_url__ = "https://github.com/cagataycali/strands-agentcore-tools"
__repo_url__ = "https://github.com/cagataycali/strands-agentcore-tools"
__issue_tracker__ = "https://github.com/cagataycali/strands-agentcore-tools/issues"
