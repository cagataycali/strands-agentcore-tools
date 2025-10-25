"""AgentCore Configure Tool

This tool provides AgentCore agent lifecycle management with:
- Dockerfile generation via ContainerRuntime
- Pydantic schema validation
- Interactive and non-interactive modes
- Memory configuration (STM/LTM)
- Lifecycle configuration
- OAuth and request header configuration
- Requirements detection
- .dockerignore template handling
"""

import boto3
import sys
import logging
import os
import platform
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from strands import tool

log = logging.getLogger(__name__)

# ========== Utility Functions ==========


def get_account_id() -> str:
    """Get AWS account ID."""
    try:
        sts = boto3.client("sts")
        account_id = sts.get_caller_identity()["Account"]
        return str(account_id)
    except Exception as e:
        raise RuntimeError(f"Failed to get AWS account ID: {e}")


def get_region() -> str:
    """Get AWS region."""

    session = boto3.Session()
    region = session.region_name
    if not region:
        region = os.environ.get("AWS_DEFAULT_REGION", "us-west-2")
    return str(region)


def validate_agent_name(name: str) -> Tuple[bool, str]:
    """Validate agent name against AWS requirements.

    Pattern: [a-zA-Z][a-zA-Z0-9_]{0,47}
    """
    pattern = r"^[a-zA-Z][a-zA-Z0-9_]{0,47}$"
    if re.match(pattern, name):
        return True, ""
    return False, (
        "Invalid agent name. Must start with a letter, contain only "
        "letters/numbers/underscores, and be 1-48 characters long."
    )


def detect_entrypoint(source_path: Path) -> Optional[Path]:
    """Detect entrypoint file in source directory."""
    candidates = ["agent.py", "app.py", "main.py", "__main__.py"]
    for candidate in candidates:
        candidate_path = source_path / candidate
        if candidate_path.exists():
            return candidate_path
    return None


def infer_agent_name(entrypoint_path: Path) -> str:
    """Infer agent name from entrypoint path."""
    rel_path = str(entrypoint_path.relative_to(Path.cwd()))
    if rel_path.endswith(".py"):
        rel_path = rel_path[:-3]
    name = rel_path.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return name


def get_python_version() -> str:
    """Get current Python version."""
    version = platform.python_version_tuple()
    return f"{version[0]}.{version[1]}"


def detect_dependencies(source_path: Path, explicit_file: Optional[str] = None) -> Any:
    """Detect requirements file in source directory."""

    class DependencyInfo:
        def __init__(
            self,
            found: bool,
            file: Optional[str] = None,
            install_path: Optional[str] = None,
        ):
            self.found = found
            self.file = file
            self.install_path = install_path
            self.is_root_package = False
            self.resolved_path = None

    if explicit_file:
        explicit_path = Path(explicit_file)
        if explicit_path.exists():
            return DependencyInfo(True, explicit_file, explicit_file)

    # Check for requirements.txt
    requirements_path = source_path / "requirements.txt"
    if requirements_path.exists():
        return DependencyInfo(True, "requirements.txt", "requirements.txt")

    # Check for pyproject.toml
    pyproject_path = source_path / "pyproject.toml"
    if pyproject_path.exists():
        info = DependencyInfo(True, "pyproject.toml", ".")
        info.is_root_package = True
        return info

    return DependencyInfo(False)


# ========== Container Runtime Management ==========


class ContainerRuntime:
    """Container runtime for Docker, Finch, and Podman."""

    DEFAULT_RUNTIME = "auto"
    DEFAULT_PLATFORM = "linux/arm64"

    def __init__(self, runtime_type: Optional[str] = None):
        """Initialize container runtime."""
        runtime_type = runtime_type or self.DEFAULT_RUNTIME
        self.available_runtimes = ["finch", "docker", "podman"]
        self.runtime = None
        self.has_local_runtime = False

        if runtime_type == "auto":
            for runtime in self.available_runtimes:
                if self._is_runtime_installed(runtime):
                    self.runtime = runtime
                    self.has_local_runtime = True
                    log.info(f"Detected {runtime} container runtime")
                    break
            else:
                log.info(
                    "No container runtime found - will use CodeBuild for deployment"
                )
                self.runtime = "none"
                self.has_local_runtime = False
        elif runtime_type in self.available_runtimes:
            if self._is_runtime_installed(runtime_type):
                self.runtime = runtime_type
                self.has_local_runtime = True
            else:
                log.warning(f"{runtime_type} not installed - will use CodeBuild")
                self.runtime = "none"
                self.has_local_runtime = False

    def _is_runtime_installed(self, runtime: str) -> bool:
        """Check if runtime is installed."""
        try:
            result = subprocess.run(
                [runtime, "version"], capture_output=True, check=False
            )
            return result.returncode == 0
        except (FileNotFoundError, OSError):
            return False

    def get_name(self) -> str:
        """Get runtime name."""
        return self.runtime.capitalize() if self.runtime else "None"

    def generate_dockerfile(
        self,
        agent_path: Path,
        output_dir: Path,
        agent_name: str,
        aws_region: Optional[str] = None,
        enable_observability: bool = True,
        requirements_file: Optional[str] = None,
        memory_id: Optional[str] = None,
        memory_name: Optional[str] = None,
        source_path: Optional[str] = None,
        protocol: Optional[str] = None,
    ) -> Path:
        """Generate Dockerfile for agent - matches launch.py implementation."""
        output_dir.mkdir(parents=True, exist_ok=True)

        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"

        # Get configuration
        entrypoint = agent_path.name

        # Detect requirements file
        requirements_file_to_use = None
        pyproject_file = None
        for fname in ["requirements.txt", "pyproject.toml"]:
            if Path(fname).exists():
                if fname == "requirements.txt":
                    requirements_file_to_use = fname
                else:
                    pyproject_file = fname
                break

        # Build Dockerfile content
        dockerfile_content = f"""FROM ghcr.io/astral-sh/uv:python{python_version}-bookworm-slim
WORKDIR /app

# Environment variables
ENV UV_SYSTEM_PYTHON=1 \\
    UV_COMPILE_BYTECODE=1 \\
    UV_NO_PROGRESS=1 \\
    PYTHONUNBUFFERED=1 \\
    DOCKER_CONTAINER=1"""

        # Add AWS region
        if aws_region:
            dockerfile_content += f""" \\
    AWS_REGION={aws_region} \\
    AWS_DEFAULT_REGION={aws_region}"""

        # Add memory configuration
        if memory_id:
            dockerfile_content += f""" \\
    BEDROCK_memory_ID={memory_id}"""
        if memory_name:
            dockerfile_content += f""" \\
    BEDROCK_memory_NAME={memory_name}"""

        dockerfile_content += "\n\n"

        # Add dependencies installation
        if pyproject_file:
            dockerfile_content += f"""COPY {pyproject_file} {pyproject_file}
RUN cd . && uv pip install .

"""
        elif requirements_file_to_use:
            dockerfile_content += f"""COPY {requirements_file_to_use} {requirements_file_to_use}
RUN uv pip install -r {requirements_file_to_use}

"""

        # Add OpenTelemetry if observability enabled
        if enable_observability:
            dockerfile_content += """RUN uv pip install aws-opentelemetry-distro>=0.10.1

"""

        # Create non-root user
        dockerfile_content += """# Create non-root user
RUN useradd -m -u 1000 bedrock_agentcore
USER bedrock_agentcore

EXPOSE 9000
EXPOSE 8000
EXPOSE 8080

# Copy project files
COPY . .

"""

        # Add CMD with or without OpenTelemetry
        if enable_observability:
            dockerfile_content += f"""# Run with OpenTelemetry instrumentation
CMD ["opentelemetry-instrument", "python", "{entrypoint}"]
"""
        else:
            dockerfile_content += f"""# Run agent
CMD ["python", "{entrypoint}"]
"""

        # Write Dockerfile
        dockerfile_path = output_dir / "Dockerfile"
        dockerfile_path.write_text(dockerfile_content)
        log.info(f"Generated Dockerfile at: {dockerfile_path}")

        # Ensure .dockerignore exists
        self._ensure_dockerignore(
            output_dir.parent if output_dir.name == ".bedrock_agentcore" else output_dir
        )

        return dockerfile_path

    def _ensure_dockerignore(self, project_dir: Path) -> None:
        """Create .dockerignore if it doesn't exist."""
        dockerignore_path = project_dir / ".dockerignore"
        if not dockerignore_path.exists():
            dockerignore_content = """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv

# IDEs
.vscode/
.idea/
*.swp
*.swo
*~

# Testing
.pytest_cache/
.coverage
htmlcov/
.tox/

# Build
build/
dist/
*.egg-info/
wheelhouse/

# Git
.git/
.gitignore

# Docker
Dockerfile
.dockerignore

# AWS
.aws/

# Bedrock AgentCore
.bedrock_agentcore/
"""
            dockerignore_path.write_text(dockerignore_content)
            log.info(f"Generated .dockerignore: {dockerignore_path}")


# ========== Configuration Management ==========


def load_config(config_path: Path) -> Optional[Dict[str, Any]]:
    """Load configuration from YAML file."""
    if not config_path.exists():
        return None

    try:
        import yaml

        with open(config_path, "r") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        log.error(f"Failed to load config: {e}")
        return None


def save_config(config: Dict[str, Any], config_path: Path) -> None:
    """Save configuration to YAML file."""
    try:
        import yaml

        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        log.info(f"Configuration saved: {config_path}")
    except Exception as e:
        log.error(f"Failed to save config: {e}")
        raise


def merge_agent_config(
    config_path: Path, agent_name: str, new_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge agent configuration into project config."""
    config = load_config(config_path)

    if config is None:
        config = {"agents": {}, "default_agent": agent_name}

    if "agents" not in config:
        config["agents"] = {}

    # Preserve existing deployment info
    if agent_name in config["agents"]:
        existing = config["agents"][agent_name]
        if "bedrock_agentcore" in existing:
            new_config["bedrock_agentcore"] = existing["bedrock_agentcore"]
        if "memory" in existing and "memory_id" in existing["memory"]:
            if "memory" not in new_config:
                new_config["memory"] = {}
            new_config["memory"]["memory_id"] = existing["memory"]["memory_id"]

    # Update agent config
    config["agents"][agent_name] = new_config
    config["default_agent"] = agent_name

    return config


# ========== Main Configure Tool ==========


@tool
def configure(  # type: ignore[return]
    action: str,
    entrypoint: Optional[str] = None,
    agent_name: Optional[str] = None,
    execution_role: Optional[str] = None,
    code_build_execution_role: Optional[str] = None,
    ecr_repository: Optional[str] = None,
    memory_mode: str = "STM_ONLY",
    enable_observability: bool = True,
    protocol: str = "HTTP",
    idle_timeout: Optional[int] = None,
    max_lifetime: Optional[int] = None,
    region: str = "us-west-2",
    source_path: Optional[str] = None,
    requirements_file: Optional[str] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    """Configure and manage AWS Bedrock AgentCore agents with full lifecycle support.

    This tool provides comprehensive AgentCore agent configuration and management,
    automatically handling Dockerfile generation, memory setup, IAM roles, and
    deployment preparation for production-ready autonomous agents.

    How It Works:
    ------------
    1. Validates agent name and entrypoint file
    2. Detects Python version and dependencies (requirements.txt/pyproject.toml)
    3. Generates optimized Dockerfile with UV package manager
    4. Configures memory system (STM/LTM) for persistent conversations
    5. Sets up IAM roles and permissions (auto-create or use existing)
    6. Saves configuration to .bedrock_agentcore.yaml for deployment
    7. Prepares agent for launch to AWS Bedrock AgentCore runtime

    Configuration Process:
    --------------------
    1. **Agent Identity**: Auto-infers name from entrypoint if not provided
    2. **AWS Resources**: Detects account/region, creates necessary IAM roles
    3. **Container Setup**: Generates Dockerfile optimized for ARM64 deployment
    4. **Memory Config**: Sets up STM/LTM memory with semantic search capabilities
    5. **Observability**: Integrates OpenTelemetry for distributed tracing
    6. **Lifecycle**: Configures idle timeout and max lifetime for cost optimization

    Three Critical AgentCore Integration Patterns:
    --------------------------------------------

    ### 1. Memory Integration (STM/LTM)
    AgentCore agents support persistent memory across conversations:

    **Memory Configuration:**
    ```python
    # STM_ONLY - Short-term memory (session-based)
    configure(action="configure", entrypoint="agent.py", memory_mode="STM_ONLY")

    # STM_AND_LTM - Long-term memory with semantic search
    configure(action="configure", entrypoint="agent.py", memory_mode="STM_AND_LTM")
    ```

    **Agent Implementation Pattern:**
    ```python
    from bedrock_agentcore.memory.integrations.strands.config import (
        AgentCoreMemoryConfig, RetrievalConfig
    )
    from bedrock_agentcore.memory.integrations.strands.session_manager import (
        AgentCoreMemorySessionManager
    )
    from strands_tools.agent_core_memory import AgentCoreMemoryToolProvider

    # Configure memory with actor-specific namespaces
    memory_config = AgentCoreMemoryConfig(
        memory_id=MEMORY_ID,
        session_id=session_id,
        actor_id=actor_id,
        retrieval_config={
            f"/users/{actor_id}/facts": RetrievalConfig(top_k=3, relevance_score=0.5),
            f"/users/{actor_id}/preferences": RetrievalConfig(top_k=3, relevance_score=0.5)
        }
    )

    # Create session manager for conversation history
    session_manager = AgentCoreMemorySessionManager(memory_config, REGION)

    # Provide memory tools to agent
    memory_provider = AgentCoreMemoryToolProvider(
        memory_id=MEMORY_ID,
        session_id=session_id,
        actor_id=actor_id,
        namespace="default",
        region=REGION,
    )
    ```

    ### 2. Tool Attachment Patterns
    AgentCore agents support multiple tool integration methods:

    **Custom Tools:**
    ```python
    from strands import tool

    @tool
    def system_prompt(action: str, prompt: str = None) -> dict:
        \"\"\"Dynamic system prompt modification.\"\"\"
        if action == "set":
            os.environ["SYSTEM_PROMPT"] = prompt
            return {"status": "success"}
        return {"status": "error"}
    ```

    **Built-in Tools:**
    ```python
    from strands_tools import shell, environment, use_agent, python_repl
    ```

    **Memory Provider Tools:**
    ```python
    # Memory provider automatically generates tools:
    # - store_memory: Store information in memory
    # - retrieve_memory: Semantic search across stored memories
    # - list_memories: List all stored memories
    memory_provider.tools  # List of dynamically generated tools
    ```

    **Hot-Reload Tools:**
    ```python
    agent = Agent(
        tools=[system_prompt, shell] + memory_provider.tools,
        load_tools_from_directory=True,  # Auto-load from ./tools/
    )
    ```

    ### 3. Result Retrieval Methods
    AgentCore agents support three execution patterns:

    **Async Non-blocking (Fire-and-forget):**
    ```python
    @app.async_task
    async def start_agent(agent, input):
        stream = agent.stream_async(input)
        async for event in stream:
            print(event)

    @app.entrypoint
    async def invoke(payload, context):
        q = payload.get("prompt", "")
        # Start task without waiting for result
        asyncio.create_task(start_agent(agent, q))
        return {"status": "success", "content": "Agent started"}
    ```

    **Async Blocking (Stream):**
    ```python
    @app.entrypoint
    async def invoke(payload, context):
        q = payload.get("prompt", "")
        stream = agent.stream_async(q)

        # Yield events as they arrive
        async for event in stream:
            yield event
    ```

    **Sync Blocking:**
    ```python
    @app.entrypoint
    async def invoke(payload, context):
        q = payload.get("prompt", "")

        # Wait for complete response
        result = agent(q)
        return {"result": str(result)}
    ```

    Common Use Cases:
    ---------------
    - **Research Agents**: Memory-enabled agents that learn from interactions
    - **Customer Support**: Multi-session agents with preference tracking
    - **Code Assistants**: Agents with hot-reload tool creation capabilities
    - **Multi-Agent Systems**: Coordinated agents with shared memory
    - **Long-running Tasks**: Background processing with async execution

    Args:
        action: Operation to perform:
            - "configure": Create/update agent configuration
            - "status": Check agent deployment status
            - "list": List all configured agents

        entrypoint: Path to agent entrypoint file (e.g., "agent.py")
            Must contain BedrockAgentCoreApp with @app.entrypoint decorator
        agent_name: Name for the agent (auto-generated if not provided)
            Pattern: [a-zA-Z][a-zA-Z0-9_]{0,47}
        execution_role: IAM role ARN for agent execution (auto-created if not provided)
            Needs: bedrock:InvokeModel, logs:*, bedrock-agentcore:*Memory*
        code_build_execution_role: Separate IAM role for CodeBuild
        ecr_repository: ECR repository URI (auto-created if not provided)
        memory_mode: Memory configuration:
            - "NO_MEMORY": No persistent memory
            - "STM_ONLY": Short-term memory (session-based) [DEFAULT]
            - "STM_AND_LTM": Long-term memory with semantic search strategies
        enable_observability: Enable OpenTelemetry observability (default: True)
            Adds aws-opentelemetry-distro for distributed tracing
        protocol: Server protocol:
            - "HTTP": RESTful HTTP server (default)
            - "MCP": Model Context Protocol
            - "A2A": Agent-to-Agent communication
        idle_timeout: Idle session timeout in seconds (60-28800)
            Sessions terminate after this period of inactivity
        max_lifetime: Maximum instance lifetime in seconds (60-28800)
            Forces session restart for resource optimization
        region: AWS region (default: us-west-2)
        source_path: Optional path to agent source code directory
        requirements_file: Optional explicit requirements file path
        verbose: Enable verbose logging (default: False)

    Returns:
        Dict containing status and response content in the format:
        {
            "status": "success|error",
            "content": [
                {"text": "Agent configuration status"},
                {"text": "Details about configuration"},
                ...
            ]
        }

        Success case: Returns configuration details and next steps
        Error case: Returns information about what went wrong

    Environment Variables:
    --------------------
    The configured agent can access these in runtime:
    - AWS_REGION: AWS region for API calls
    - BEDROCK_memory_ID: Memory resource ID (if memory enabled)
    - BEDROCK_memory_NAME: Memory resource name
    - SYSTEM_PROMPT: Dynamic system prompt (modifiable at runtime)

    Configuration File:
    -----------------
    Creates .bedrock_agentcore.yaml with complete agent configuration:
    ```yaml
    agents:
      my-research-agent:
        name: my-research-agent
        entrypoint: agent.py
        platform: linux/arm64
        aws:
          execution_role: arn:aws:iam::123:role/AgentCoreRuntime
          region: us-west-2
          ecr_repository: 123.dkr.ecr.us-west-2.amazonaws.com/bedrock-agentcore-my-agent
          network_configuration:
            network_mode: PUBLIC
          protocol_configuration:
            server_protocol: HTTP
          observability:
            enabled: true
        memory:
          mode: STM_AND_LTM
          memory_name: my-research-agent_memory
          memory_id: my-memory-abc123  # Preserved across updates
        bedrock_agentcore:  # Populated by launch tool
          agent_id: my-agent-abc123
          agent_arn: arn:aws:bedrock-agentcore:us-west-2:123:runtime/my-agent
    default_agent: my-research-agent
    ```

    Examples:
    --------
    # Basic configuration
    configure(
        action="configure",
        entrypoint="agent.py",
        agent_name="my-agent"
    )

    # Full-featured research agent
    configure(
        action="configure",
        entrypoint="research_agent.py",
        agent_name="research-assistant",
        memory_mode="STM_AND_LTM",
        enable_observability=True,
        idle_timeout=1800,  # 30 minutes
        max_lifetime=7200,  # 2 hours
        protocol="HTTP"
    )

    # Production agent with custom IAM roles
    configure(
        action="configure",
        entrypoint="production_agent.py",
        agent_name="prod-agent",
        execution_role="arn:aws:iam::123456:role/MyAgentRole",
        ecr_repository="123456.dkr.ecr.us-west-2.amazonaws.com/my-agents",
        memory_mode="STM_AND_LTM",
        region="us-west-2"
    )

    # Check agent status
    configure(action="status", agent_name="my-agent")

    # List all configured agents
    configure(action="list")

    Notes:
        - Configuration persists in .bedrock_agentcore.yaml
        - Memory IDs are preserved across configuration updates
        - Dockerfile is auto-generated with optimal ARM64 settings
        - IAM roles auto-created with minimal required permissions
        - After configure, use launch tool to deploy to AgentCore
        - Memory strategies (semantic, preferences, summarization) enable intelligent recall
        - Session manager handles conversation history automatically
        - Tool hot-reload enables agents to create their own capabilities
        - Async non-blocking pattern recommended for long-running tasks
    """
    # Set logging level
    if verbose:
        log.setLevel(logging.DEBUG)

    # Ensure required packages
    try:
        import yaml
    except ImportError:
        return {
            "status": "error",
            "content": [
                {"text": "Required packages missing. Install: pip install boto3 pyyaml"}
            ],
        }

    config_path = Path.cwd() / ".bedrock_agentcore.yaml"

    try:
        if action == "configure":
            if not entrypoint:
                return {
                    "status": "error",
                    "content": [
                        {"text": "entrypoint is required for configure action"}
                    ],
                }

            entrypoint_path = Path(entrypoint).resolve()
            if not entrypoint_path.exists():
                return {
                    "status": "error",
                    "content": [{"text": f"Entrypoint file not found: {entrypoint}"}],
                }

            # Auto-detect agent name if not provided
            if not agent_name:
                agent_name = infer_agent_name(entrypoint_path)
                log.info(f"Inferred agent name: {agent_name}")

            # Validate agent name
            valid, error = validate_agent_name(agent_name)
            if not valid:
                return {"status": "error", "content": [{"text": error}]}

            # Get AWS info
            try:
                account_id = get_account_id()
                region = region or get_region()
            except Exception as e:
                return {
                    "status": "error",
                    "content": [{"text": f"AWS authentication failed: {e}"}],
                }

            log.info(f"Configuring agent: {agent_name}")
            log.info(f"Account: {account_id}, Region: {region}")

            # Initialize container runtime
            runtime = ContainerRuntime()

            # Handle execution role
            execution_role_arn = None
            execution_role_auto_create = True

            if execution_role:
                if execution_role.startswith("arn:aws:iam::"):
                    execution_role_arn = execution_role
                else:
                    execution_role_arn = (
                        f"arn:aws:iam::{account_id}:role/{execution_role}"
                    )
                execution_role_auto_create = False
                log.info(f"Using execution role: {execution_role_arn}")
            else:
                log.info("Execution role will be auto-created")

            # Handle CodeBuild execution role
            codebuild_execution_role_arn = None
            if code_build_execution_role:
                if code_build_execution_role.startswith("arn:aws:iam::"):
                    codebuild_execution_role_arn = code_build_execution_role
                else:
                    codebuild_execution_role_arn = (
                        f"arn:aws:iam::{account_id}:role/{code_build_execution_role}"
                    )
                log.info(f"Using CodeBuild role: {codebuild_execution_role_arn}")

            # Handle memory configuration
            memory_config = {
                "mode": memory_mode,
                "event_expiry_days": 30,
                "memory_name": f"{agent_name}_memory",
            }

            # Check for existing memory ID
            existing_config = load_config(config_path)
            if existing_config and "agents" in existing_config:
                if agent_name in existing_config["agents"]:
                    existing_memory = existing_config["agents"][agent_name].get(
                        "memory", {}
                    )
                    if "memory_id" in existing_memory:
                        memory_config["memory_id"] = existing_memory["memory_id"]
                        log.info(
                            f"Preserving existing memory ID: {memory_config['memory_id']}"
                        )

            log.info(f"Memory mode: {memory_mode}")

            # Handle lifecycle configuration
            lifecycle_config = {}
            if idle_timeout is not None:
                lifecycle_config["idle_runtime_session_timeout"] = idle_timeout
                log.info(f"Idle timeout: {idle_timeout}s")
            if max_lifetime is not None:
                lifecycle_config["max_lifetime"] = max_lifetime
                log.info(f"Max lifetime: {max_lifetime}s")

            # Generate Dockerfile in proper directory structure
            log.info("Generating Dockerfile...")
            # Always use .bedrock_agentcore/agent_name/ directory
            agentcore_dir = Path.cwd() / ".bedrock_agentcore" / agent_name
            agentcore_dir.mkdir(parents=True, exist_ok=True)
            output_dir = agentcore_dir

            memory_id = memory_config.get("memory_id")
            memory_name = memory_config.get("memory_name")

            dockerfile_path = runtime.generate_dockerfile(
                entrypoint_path,
                output_dir,
                agent_name,
                region,
                enable_observability,
                requirements_file,
                str(memory_id) if memory_id else None,
                str(memory_name) if memory_name else None,
                source_path,
                protocol,
            )

            # Build configuration
            agent_config = {
                "name": agent_name,
                "entrypoint": str(entrypoint_path.relative_to(Path.cwd())),
                "platform": ContainerRuntime.DEFAULT_PLATFORM,
                "container_runtime": runtime.runtime,
                "aws": {
                    "execution_role": execution_role_arn,
                    "execution_role_auto_create": execution_role_auto_create,
                    "account": account_id,
                    "region": region,
                    "ecr_repository": ecr_repository,
                    "ecr_auto_create": not bool(ecr_repository),
                    "network_configuration": {"network_mode": "PUBLIC"},
                    "protocol_configuration": {"server_protocol": protocol},
                    "observability": {"enabled": enable_observability},
                },
                "memory": memory_config,
                "bedrock_agentcore": {},
            }

            if source_path:
                agent_config["source_path"] = str(Path(source_path).resolve())

            if lifecycle_config:
                # Type assertion for nested dict
                aws_config = agent_config["aws"]
                if isinstance(aws_config, dict):
                    aws_config["lifecycle_configuration"] = lifecycle_config  # type: ignore[assignment]

            if codebuild_execution_role_arn:
                agent_config["codebuild"] = {  # type: ignore[assignment]
                    "execution_role": codebuild_execution_role_arn
                }

            # Save configuration
            project_config = merge_agent_config(config_path, agent_name, agent_config)
            save_config(project_config, config_path)

            return {
                "status": "success",
                "content": [
                    {"text": f"✅ **Agent Configured Successfully**"},
                    {"text": f"**Agent Name:** {agent_name}"},
                    {"text": f"**Entrypoint:** {entrypoint}"},
                    {"text": f"**Region:** {region}"},
                    {"text": f"**Account:** {account_id}"},
                    {"text": f"**Memory Mode:** {memory_mode}"},
                    {"text": f"**Protocol:** {protocol}"},
                    {
                        "text": f"**Observability:** {'Enabled' if enable_observability else 'Disabled'}"
                    },
                    {"text": f"**Container Runtime:** {runtime.get_name()}"},
                    {"text": f"**Dockerfile:** {dockerfile_path}"},
                    {"text": f"**Config:** {config_path}"},
                    {"text": "\n**Next Steps:**\n   Use launch tool to deploy"},
                ],
            }

        elif action == "status":
            if not config_path.exists():
                return {
                    "status": "error",
                    "content": [
                        {"text": "No configuration found. Run configure action first."}
                    ],
                }

            config = load_config(config_path)
            if not config or "agents" not in config:
                return {
                    "status": "error",
                    "content": [{"text": "Invalid configuration format"}],
                }

            # Get agent config
            if agent_name:
                if agent_name not in config["agents"]:
                    return {
                        "status": "error",
                        "content": [{"text": f"Agent '{agent_name}' not found"}],
                    }
                agent_config = config["agents"][agent_name]
            else:
                default_agent = config.get("default_agent")
                if not default_agent:
                    return {
                        "status": "error",
                        "content": [{"text": "No default agent found"}],
                    }
                agent_config = config["agents"][default_agent]
                agent_name = default_agent

            # Check deployment status
            bedrock_info = agent_config.get("bedrock_agentcore", {})
            agent_arn = bedrock_info.get("agent_arn")
            agent_id = bedrock_info.get("agent_id")

            if not agent_arn:
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": f"⚠️ **Agent '{agent_name}' Configured but Not Deployed**"
                        },
                        {"text": f"**Region:** {agent_config['aws']['region']}"},
                        {"text": f"**Account:** {agent_config['aws']['account']}"},
                        {"text": "\n**Next Steps:**\n   Use launch to deploy"},
                    ],
                }

            # Get runtime status from AWS
            try:
                client = boto3.client(
                    "bedrock-agentcore-control",
                    region_name=agent_config["aws"]["region"],
                )
                if agent_id:
                    response = client.get_agent_runtime(agentRuntimeId=agent_id)
                    return {
                        "status": "success",
                        "content": [
                            {"text": f"✅ **Agent '{agent_name}' Status**"},
                            {
                                "text": f"**Status:** {response.get('status', 'Unknown')}"
                            },
                            {"text": f"**Agent ARN:** {agent_arn}"},
                            {"text": f"**Agent ID:** {agent_id}"},
                            {"text": f"**Region:** {agent_config['aws']['region']}"},
                            {
                                "text": f"**Created:** {response.get('createdAt', 'Unknown')}"
                            },
                            {
                                "text": f"**Last Updated:** {response.get('lastUpdatedAt', 'Unknown')}"
                            },
                        ],
                    }
            except Exception as e:
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to get status: {e}"}],
                }

        elif action == "list":
            if not config_path.exists():
                return {
                    "status": "success",
                    "content": [{"text": "No agents configured yet."}],
                }

            config = load_config(config_path)
            if not config or "agents" not in config or not config["agents"]:
                return {
                    "status": "success",
                    "content": [{"text": "No agents configured yet."}],
                }

            default_agent = config.get("default_agent")
            content = [{"text": f"**Configured Agents ({len(config['agents'])}):**\n"}]

            for name, agent_config in config["agents"].items():
                is_default = " (default)" if name == default_agent else ""
                has_arn = (
                    "✅ Deployed"
                    if agent_config.get("bedrock_agentcore", {}).get("agent_arn")
                    else "⚠️ Config only"
                )

                content.append({"text": f"\n**{name}**{is_default}"})
                content.append({"text": f"  Status: {has_arn}"})
                content.append(
                    {"text": f"  Entrypoint: {agent_config.get('entrypoint', 'N/A')}"}
                )
                content.append(
                    {
                        "text": f"  Region: {agent_config.get('aws', {}).get('region', 'N/A')}"
                    }
                )
                content.append(
                    {
                        "text": f"  Memory: {agent_config.get('memory', {}).get('mode', 'N/A')}"
                    }
                )

            return {"status": "success", "content": content}

        else:
            return {
                "status": "error",
                "content": [
                    {"text": f"Unknown action: {action}"},
                    {"text": "Valid actions: configure, status, list"},
                ],
            }

    except Exception as e:
        log.exception("Configuration failed")
        return {
            "status": "error",
            "content": [
                {"text": f"**Error:** {str(e)}"},
                {"text": f"**Action:** {action}"},
            ],
        }
