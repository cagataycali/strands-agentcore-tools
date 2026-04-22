<div align="center">
  <a href="https://strandsagents.com">
    <img src="https://strandsagents.com/latest/assets/logo-github.svg" alt="Strands Agents" width="55px" height="105px">
  </a>
  <span style="font-size: 48px; margin: 0 20px;">×</span>
  <a href="https://aws.amazon.com/bedrock/agentcore/">
    <img width="150" height="150" alt="Bedrock AgentCore" src="https://github.com/user-attachments/assets/b8b9456d-c9e2-45e1-ac5b-760f21f1ac18" />
  </a>

  <h1>Strands AgentCore Tools</h1>
  <p><strong>Agentic tools that enable AI agents to autonomously deploy, manage, and monitor themselves on AWS Bedrock AgentCore</strong></p>

  <p>
    <a href="https://pypi.org/project/strands-agentcore-tools"><img src="https://img.shields.io/pypi/v/strands-agentcore-tools" alt="PyPI version"></a>
    <a href="https://github.com/cagataycali/strands-agentcore-tools/blob/main/LICENSE"><img src="https://img.shields.io/github/license/cagataycali/strands-agentcore-tools" alt="License"></a>
  </p>
</div>

---

## What This Is

**9 Python functions that wrap AWS Bedrock AgentCore boto3 APIs** for deployment, invocation, monitoring, and lifecycle management. Use these tools to:

- ✅ Deploy agents to AgentCore from code
- ✅ Invoke deployed agents programmatically
- ✅ Monitor CloudWatch logs and metrics
- ✅ Manage memory, sessions, and OAuth
- ✅ Integrate with Strands agents for autonomous deployment

**Lightweight by design:** Import only the tools you need for your use case.

---

## Agent Example

**Copy-paste ready agent with all execution modes:**

```python
# agent.py
[![Awesome Strands Agents](https://img.shields.io/badge/Awesome-Strands%20Agents-00FF77?style=flat-square&logo=data:image/svg+xml;base64,PHN2ZyB3aWR0aD0iMjkwIiBoZWlnaHQ9IjQ2MyIgdmlld0JveD0iMCAwIDI5MCA0NjMiIGZpbGw9Im5vbmUiIHhtbG5zPSJodHRwOi8vd3d3LnczLm9yZy8yMDAwL3N2ZyI+CjxwYXRoIGQ9Ik05Ny4yOTAyIDUyLjc4ODRDODUuMDY3NCA0OS4xNjY3IDcyLjIyMzQgNTYuMTM4OSA2OC42MDE3IDY4LjM2MTZDNjQuOTgwMSA4MC41ODQzIDcxLjk1MjQgOTMuNDI4MyA4NC4xNzQ5IDk3LjA1MDFMMjM1LjExNyAxMzkuNzc1QzI0NS4yMjMgMTQyLjc2OSAyNDYuMzU3IDE1Ni42MjggMjM2Ljg3NCAxNjEuMjI2TDMyLjU0NiAyNjAuMjkxQy0xNC45NDM5IDI4My4zMTYgLTkuMTYxMDcgMzUyLjc0IDQxLjQ4MzUgMzY3LjU5MUwxODkuNTUxIDQxMS4wMDlMMTkwLjEyNSA0MTEuMTY5QzIwMi4xODMgNDE0LjM3NiAyMTQuNjY1IDQwNy4zOTYgMjE4LjE5NiAzOTUuMzU1QzIyMS43ODQgMzgzLjEyMiAyMTQuNzc0IDM3MC4yOTYgMjAyLjU0MSAzNjYuNzA5TDU0LjQ3MzggMzIzLjI5MUM0NC4zNDQ3IDMyMC4zMjEgNDMuMTg3OSAzMDYuNDM2IDUyLjY4NTcgMzAxLjgzMUwyNTcuMDE0IDIwMi43NjZDMzA0LjQzMiAxNzkuNzc2IDI5OC43NTggMTEwLjQ4MyAyNDguMjMzIDk1LjUxMkw5Ny4yOTAyIDUyLjc4ODRaIiBmaWxsPSIjRkZGRkZGIi8+CjxwYXRoIGQ9Ik0yNTkuMTQ3IDAuOTgxODEyQzI3MS4zODkgLTIuNTc0OTggMjg0LjE5NyA0LjQ2NTcxIDI4Ny43NTQgMTYuNzA3NEMyOTEuMzExIDI4Ljk0OTIgMjg0LjI3IDQxLjc1NyAyNzIuMDI4IDQ1LjMxMzhMNzEuMTcyNyAxMDMuNjcxQzQwLjcxNDIgMTEyLjUyMSAzNy4xOTc2IDE1NC4yNjIgNjUuNzQ1OSAxNjguMDgzTDI0MS4zNDMgMjUzLjA5M0MzMDcuODcyIDI4NS4zMDIgMjk5Ljc5NCAzODIuNTQ2IDIyOC44NjIgNDAzLjMzNkwzMC40MDQxIDQ2MS41MDJDMTguMTcwNyA0NjUuMDg4IDUuMzQ3MDggNDU4LjA3OCAxLjc2MTUzIDQ0NS44NDRDLTEuODIzOSA0MzMuNjExIDUuMTg2MzcgNDIwLjc4NyAxNy40MTk3IDQxNy4yMDJMMjE1Ljg3OCAzNTkuMDM1QzI0Ni4yNzcgMzUwLjEyNSAyNDkuNzM5IDMwOC40NDkgMjIxLjIyNiAyOTQuNjQ1TDQ1LjYyOTcgMjA5LjYzNUMtMjAuOTgzNCAxNzcuMzg2IC0xMi43NzcyIDc5Ljk4OTMgNTguMjkyOCA1OS4zNDAyTDI1OS4xNDcgMC45ODE4MTJaIiBmaWxsPSIjRkZGRkZGIi8+Cjwvc3ZnPgo=&logoColor=white)](https://github.com/cagataycali/awesome-strands-agents)

import threading
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent

app = BedrockAgentCoreApp()

@app.entrypoint
async def invoke(payload, context):
    """Async entrypoint with yield support for streaming"""
    agent = Agent()
    prompt = payload.get("prompt", "")
    mode = payload.get("mode", "streaming")  # streaming | sync | fire_and_forget
    
    if mode == "fire_and_forget":
        # Start background thread
        task_id = app.add_async_task("agent_processing", payload)
        thread = threading.Thread(
            target=run_agent_background,
            args=(agent, prompt, task_id),
            daemon=True
        )
        thread.start()
        yield {"status": "started", "content": [{"text": "Agent running in background"}]}
    
    elif mode == "sync":
        # Single blocking response (wait for completion)
        result = agent(prompt)
        yield {"status": "success", "content": [{"text": str(result)}]}
    
    else:
        # Streaming response
        for event in agent.stream_async(prompt):
            yield event

def run_agent_background(agent, prompt, task_id):
    """Background worker (sync function, runs in separate thread)"""
    try:
        for event in agent.stream_async(prompt):
            print(event)  # Log events in background
        app.complete_async_task(task_id)
    except Exception as e:
        app.logger.error(f"Background task failed: {e}")
        app.complete_async_task(task_id)

app.run()
```

**Deploy this agent:**
```python
from strands_agentcore_tools import configure, launch, invoke

configure(action="configure", entrypoint="agent.py", agent_name="my-agent")
launch(action="launch", agent_name="my-agent")
invoke(agent_arn="arn:...", payload='{"prompt": "Hello!", "mode": "streaming"}')
```

<details>
<summary><b>📝 Advanced: Agent with Memory (STM + LTM)</b></summary>

```python
# agent.py - Agent with persistent memory
import os
from bedrock_agentcore import BedrockAgentCoreApp
from strands import Agent
from bedrock_agentcore.memory.integrations.strands.config import (
    AgentCoreMemoryConfig, RetrievalConfig
)
from bedrock_agentcore.memory.integrations.strands.session_manager import (
    AgentCoreMemorySessionManager
)
from strands_tools.agent_core_memory import AgentCoreMemoryToolProvider

app = BedrockAgentCoreApp()

MEMORY_ID = os.getenv("BEDROCK_AGENTCORE_MEMORY_ID")
REGION = os.getenv("AWS_REGION", "us-west-2")

@app.entrypoint
async def invoke(payload, context):
    """Async entrypoint with yield support"""
    session_id = context.session_id
    actor_id = context.headers.get(
        "X-Amzn-Bedrock-AgentCore-Runtime-Custom-Actor-Id", 
        "user"
    )
    
    # Configure memory with semantic search
    memory_config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        session_id=session_id,
        actor_id=actor_id,
        retrieval_config={
            f"/users/{actor_id}/facts": RetrievalConfig(
                top_k=3, 
                relevance_score=0.5
            )
        }
    )
    
    # Memory tools provider
    memory_provider = AgentCoreMemoryToolProvider(
        memory_id=MEMORY_ID,
        session_id=session_id,
        actor_id=actor_id,
        namespace="default",
        region=REGION
    )
    
    # Create agent with memory
    agent = Agent(
        tools=memory_provider.tools,
        session_manager=AgentCoreMemorySessionManager(
            agentcore_memory_config=memory_config,
            region=REGION
        ),
        system_prompt="You have persistent memory across conversations."
    )
    
    # Stream responses
    for event in agent.stream_async(payload.get("prompt")):
        yield event

app.run()
```

**Deploy with memory:**
```python
from strands_agentcore_tools import memory, configure, launch

# 1. Create memory
memory(
    action="create",
    name="my-memory",
    strategies=[{
        "semanticMemoryStrategy": {
            "name": "Facts",
            "namespaces": ["/users/{actorId}/facts"]
        }
    }],
    wait_for_active=True
)

# 2. Configure with memory
configure(
    action="configure",
    entrypoint="agent.py",
    agent_name="my-agent",
    memory_mode="STM_AND_LTM"
)

# 3. Launch
launch(action="launch", agent_name="my-agent")
```

</details>

---

## Installation

```bash
pip install strands-agentcore-tools
```

**Requirements:**
- Python 3.10+
- AWS credentials configured
- IAM permissions for `bedrock-agentcore:*`, `ecr:*`, `codebuild:*`, `iam:*`, `logs:*`

---

## The 9 Tools

| Tool | Purpose | Key Parameters |
|------|---------|----------------|
| **`configure`** | Generate Dockerfile, IAM roles, config YAML | `entrypoint`, `agent_name`, `memory_mode` |
| **`launch`** | Build & deploy to AgentCore via CodeBuild | `agent_name`, `mode`, `env_vars` |
| **`invoke`** | Execute deployed agent with payload | `agent_arn`, `payload`, `session_id` |
| **`agents`** | List/get agent runtimes | `action` ("list", "get", "find_by_name") |
| **`status`** | Check agent health & endpoint status | `agent_id` or `agent_name` |
| **`logs`** | Query CloudWatch logs | `agent_name`, `action`, `filter_pattern` |
| **`memory`** | Manage AgentCore memories (STM/LTM) | `action`, `memory_id`, `strategies` |
| **`identity`** | OAuth provider management | `action`, `provider_type`, `vendor` |
| **`session`** | Stop active runtime sessions | `agent_arn`, `session_id` |

---

## Quick Reference

### 1. Configure

**Generates deployment files:** Dockerfile, `.bedrock_agentcore.yaml`, IAM roles

```python
from strands_agentcore_tools import configure

configure(
    action="configure",
    entrypoint="agent.py",
    agent_name="my-agent",
    memory_mode="STM_AND_LTM",          # or "STM_ONLY", "NO_MEMORY"
    enable_observability=True,
    idle_timeout=1800,                  # seconds
    max_lifetime=7200,                  # seconds
    region="us-west-2"
)
```

**Actions:**
- `configure` - Generate deployment files
- `status` - Check configuration status
- `list` - List all configured agents

**Output:**
- `.bedrock_agentcore/<agent_name>/Dockerfile`
- `.bedrock_agentcore/<agent_name>/.dockerignore`
- `.bedrock_agentcore.yaml`

---

### 2. Launch

**Builds ARM64 container on CodeBuild, pushes to ECR, deploys to AgentCore**

```python
from strands_agentcore_tools import launch

result = launch(
    action="launch",
    agent_name="my-agent",
    mode="codebuild",                   # or "local" (requires Docker/Finch/Podman)
    auto_update_on_conflict=True,
    env_vars={
        "MODEL_ID": "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "LOG_LEVEL": "DEBUG"
    },
    region="us-west-2"
)

print(result["agent_arn"])  # arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc
```

**Actions:**
- `launch` - Deploy agent
- `status` - Check deployment status
- `stop` - Stop active deployment

**Returns:**
```python
{
    "agent_id": "my-agent-abc123",
    "agent_arn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc123",
    "status": "READY"
}
```

---

### 3. Invoke

**Execute deployed agent, supports streaming responses**

```python
from strands_agentcore_tools import invoke

# Streaming invocation (default)
invoke(
    agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc",
    payload='{"prompt": "Analyze this dataset"}',
    session_id="session-123",           # Optional: for conversation continuity
    user_id="user-abc",                 # Optional: for OAuth flows
    headers={                           # Optional: custom headers
        "X-Custom-Header": "value"
    },
    region="us-west-2"
)
```

**Parameters:**
- `agent_arn` (required) - Full ARN of deployed agent
- `payload` (required) - JSON string payload
- `session_id` (optional) - Min 33 chars for session tracking
- `user_id` (optional) - Triggers OAuth if configured
- `headers` (optional) - Custom HTTP headers
- `region` (optional) - AWS region

**Returns:** SSE streaming events from agent

---

### 4. Agents

**List, get, and search agent runtimes**

```python
from strands_agentcore_tools import agents

# List all agents
agents(action="list", region="us-west-2")

# Get specific agent
agents(action="get", agent_id="my-agent-abc123", region="us-west-2")

# Find by name
result = agents(action="find_by_name", agent_name="my-agent", region="us-west-2")
print(result["agent_arn"])
```

**Actions:**
- `list` - List all agent runtimes
- `get` - Get agent by ID
- `find_by_name` - Find agent by name (returns first match)

**Returns:**
```python
{
    "agent_id": "my-agent-abc123",
    "agent_arn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc123",
    "status": "READY",
    "created": "2024-10-24T14:20:00Z"
}
```

---

### 5. Status

**Check agent health and endpoint status**

```python
from strands_agentcore_tools import status

status(
    agent_id="my-agent-abc123",         # Can use agent_id
    agent_name="my-agent",              # Or agent_name (auto-lookup)
    endpoint="DEFAULT",                 # Optional: specific endpoint
    region="us-west-2"
)
```

**Returns:**
```python
{
    "agent_id": "my-agent-abc123",
    "status": "READY",                  # READY, CREATING, UPDATING, DELETING, FAILED
    "endpoint": "DEFAULT",
    "created": "2024-10-24T14:20:00Z",
    "updated": "2024-10-24T15:30:00Z"
}
```

---

### 6. Logs

**Query CloudWatch logs for deployed agents**

```python
from strands_agentcore_tools import logs

# Recent logs
logs(
    agent_name="my-agent",
    action="recent",
    limit=50,
    region="us-west-2"
)

# Search for errors
logs(
    agent_name="my-agent",
    action="search",
    filter_pattern="ERROR",
    limit=100,
    region="us-west-2"
)

# Tail logs (real-time)
logs(
    agent_name="my-agent",
    action="tail",
    region="us-west-2"
)

# List log streams
logs(
    agent_name="my-agent",
    action="streams",
    region="us-west-2"
)
```

**Actions:**
- `recent` - Get recent log events
- `search` - Filter logs by pattern
- `tail` - Stream logs in real-time
- `streams` - List available log streams

**Parameters:**
- `agent_name` - Agent name (auto-lookups runtime ID)
- `agent_id` - Or use runtime ID directly
- `endpoint` - Default: "DEFAULT"
- `limit` - Max events to return
- `filter_pattern` - CloudWatch filter pattern

---

### 7. Memory

**Manage AgentCore memories (STM/LTM)**

```python
from strands_agentcore_tools import memory

# Create memory
memory(
    action="create",
    name="research-memory",
    strategies=[{
        "semanticMemoryStrategy": {
            "name": "Facts",
            "namespaces": ["/users/{actorId}/facts"]
        }
    }],
    wait_for_active=True,               # Block until ACTIVE
    region="us-west-2"
)

# Get memory status
memory(
    action="get_status",
    memory_id="memory-abc123",
    region="us-west-2"
)

# Retrieve from memory
memory(
    action="retrieve",
    memory_id="memory-abc123",
    namespace="/users/user-abc/facts",
    search_query="What is the capital of France?",
    top_k=5,
    min_score=0.5,
    region="us-west-2"
)

# Create memory event
memory(
    action="create_event",
    memory_id="memory-abc123",
    session_id="session-123",
    actor_id="user-abc",
    namespace="/users/user-abc/facts",
    content="Paris is the capital of France",
    region="us-west-2"
)

# List all memories
memory(action="list", region="us-west-2")
```

**Actions:**
- `create` - Create new memory
- `get_status` - Check memory status
- `retrieve` - Semantic search
- `create_event` - Add memory event
- `list` - List all memories

---

### 8. Identity

**OAuth provider management**

```python
from strands_agentcore_tools import identity

# Create OAuth provider
identity(
    action="create",
    provider_type="oauth2",
    name="slack-oauth",
    vendor="SlackOauth2",               # SlackOauth2, GitHubOAuth2, GoogleOAuth2
    client_id="your-client-id",
    client_secret="your-client-secret",
    region="us-west-2"
)

# Get provider
identity(
    action="get",
    provider_id="provider-abc123",
    region="us-west-2"
)

# List providers
identity(action="list", region="us-west-2")

# Delete provider
identity(
    action="delete",
    provider_id="provider-abc123",
    region="us-west-2"
)
```

**Actions:**
- `create` - Create OAuth provider
- `get` - Get provider details
- `list` - List all providers
- `delete` - Delete provider

**Supported Vendors:**
- `SlackOauth2`
- `GitHubOAuth2`
- `GoogleOAuth2`

---

### 9. Session

**Stop active runtime sessions**

```python
from strands_agentcore_tools import session

session(
    action="stop",
    agent_arn="arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc",
    session_id="session-123",
    region="us-west-2"
)
```

**Actions:**
- `stop` - Terminate active session

---

## Complete Workflow Example

```python
from strands_agentcore_tools import (
    configure, launch, invoke, status, logs, memory
)

# 1. Configure deployment
configure(
    action="configure",
    entrypoint="agent.py",
    agent_name="research-agent",
    memory_mode="STM_AND_LTM"
)

# 2. Create memory
memory(
    action="create",
    name="research-memory",
    strategies=[{
        "semanticMemoryStrategy": {
            "name": "Facts",
            "namespaces": ["/users/{actorId}/facts"]
        }
    }],
    wait_for_active=True
)

# 3. Deploy to AWS
result = launch(
    action="launch",
    agent_name="research-agent",
    mode="codebuild"
)

agent_arn = result["agent_arn"]

# 4. Check deployment status
status(agent_name="research-agent")

# 5. Invoke agent
invoke(
    agent_arn=agent_arn,
    payload='{"prompt": "Analyze this data"}',
    session_id="session-abc-123-very-long-session-id-here"
)

# 6. Check logs
logs(agent_name="research-agent", action="recent", limit=50)
```

---

## Use with Strands Agents

**Enable autonomous deployment by giving agents these tools:**

```python
from strands import Agent
from strands_agentcore_tools import (
    configure, launch, invoke, status, logs
)

agent = Agent(
    tools=[configure, launch, invoke, status, logs],
    system_prompt="You can deploy yourself to AWS AgentCore."
)

# Agent deploys itself
response = agent("""
Deploy yourself to AWS:
1. Configure deployment
2. Launch to production
3. Check status
4. Show logs
""")
```

**Agent will autonomously:**
- ✅ Generate Dockerfile and config
- ✅ Trigger CodeBuild deployment
- ✅ Monitor deployment status
- ✅ Validate with logs
- ✅ Report results

---

## Configuration File

**`.bedrock_agentcore.yaml`** (auto-generated by `configure`):

```yaml
agents:
  my-agent:
    name: my-agent
    entrypoint: agent.py
    platform: linux/arm64
    aws:
      execution_role: arn:aws:iam::123:role/AgentCoreRuntime-us-west-2-abc
      region: us-west-2
      ecr_repository: 123.dkr.ecr.us-west-2.amazonaws.com/bedrock-agentcore-my-agent
    memory:
      mode: STM_AND_LTM
      memory_id: memory-abc123
    bedrock_agentcore:
      agent_id: my-agent-abc123
      agent_arn: arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc123
```

---

## IAM Permissions

**Runtime Execution Role** (auto-created by `configure`):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      "logs:CreateLogGroup",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "bedrock-agentcore:CreateMemoryEvent",
      "bedrock-agentcore:RetrieveMemoryRecords",
      "ecr:GetAuthorizationToken",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "xray:PutTraceSegments",
      "xray:PutTelemetryRecords"
    ],
    "Resource": "*"
  }]
}
```

**Your IAM User/Role** (attach manually):

```json
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": [
      "bedrock-agentcore-control:*",
      "bedrock-agentcore:*",
      "ecr:*",
      "codebuild:*",
      "iam:CreateRole",
      "iam:GetRole",
      "iam:PassRole",
      "iam:PutRolePolicy",
      "logs:*",
      "s3:PutObject",
      "s3:GetObject"
    ],
    "Resource": "*"
  }]
}
```

---

## Error Handling

All tools return structured responses:

```python
# Success
{
    "status": "success",
    "agent_arn": "arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent-abc",
    "message": "Agent deployed successfully"
}

# Error
{
    "status": "error",
    "error": "ConflictException",
    "message": "Agent already exists. Use auto_update_on_conflict=True"
}
```

**Common Errors:**
- `ConflictException` - Agent exists, use `auto_update_on_conflict=True` in `launch()`
- `ValidationException` - Invalid parameters (check agent_name format)
- `AccessDeniedException` - IAM permissions missing
- `ResourceNotFoundException` - Agent/memory not found

---

## Observability

**View logs programmatically:**

```python
from strands_agentcore_tools import logs

# Recent logs
logs(agent_name="my-agent", action="recent", limit=50)

# Search for errors
logs(agent_name="my-agent", action="search", filter_pattern="ERROR")

# Tail in real-time
logs(agent_name="my-agent", action="tail")
```

**Add ADOT instrumentation for tracing:**

```dockerfile
# In Dockerfile (auto-generated by configure)
RUN pip install aws-opentelemetry-distro>=0.10.1
CMD ["opentelemetry-instrument", "python", "agent.py"]
```

**CloudWatch Logs Location:**
```
/aws/bedrock-agentcore/runtimes/<agent_id>-<endpoint>
```

---

## Development

**Local Testing:**

```bash
# Install package locally
pip install -e .

# Run tests
pytest tests/

# Format code
black strands_agentcore_tools/
```

**Project Structure:**

```
strands-agentcore-tools/
├── strands_agentcore_tools/
│   ├── __init__.py
│   ├── configure.py        # Deployment configuration
│   ├── launch.py           # CodeBuild deployment
│   ├── invoke.py           # Agent invocation
│   ├── agents.py           # Agent discovery
│   ├── status.py           # Health checks
│   ├── logs.py             # CloudWatch logs
│   ├── memory.py           # Memory management
│   ├── identity.py         # OAuth providers
│   └── session.py          # Session control
├── setup.py
├── pyproject.toml
└── README.md
```

---

## Resources

**Documentation:**
- [Strands Agents Docs](https://strandsagents.com/)
- [AWS Bedrock AgentCore Docs](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/)
- [AgentCore API Reference](https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/bedrock-agentcore-control.html)

**Related Projects:**
- [bedrock-agentcore-starter-toolkit](https://github.com/aws/bedrock-agentcore-starter-toolkit) - Official AWS toolkit
- [strands-agents](https://github.com/strands-agents/sdk-python) - Strands SDK

---

## License

Apache 2.0 License - see [LICENSE](LICENSE) file for details.

---

## Contributing

Issues and PRs welcome at [github.com/cagataycali/strands-agentcore-tools](https://github.com/cagataycali/strands-agentcore-tools)
